from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from features.embeddings import EmbeddingConfig, SimilarityConfig, build_embedder
from models import Direction, Market, Side, TradeEvent, TraderMarketSettlement


def side_to_yes_shares(side: Side, size: float) -> float:
    if side == Side.BUY_YES:
        return size
    if side == Side.SELL_YES:
        return -size
    if side == Side.BUY_NO:
        return -size
    if side == Side.SELL_NO:
        return size
    return 0.0


def side_to_cashflow(side: Side, size: float, yes_price: float) -> float:
    no_price = 1.0 - yes_price
    if side == Side.BUY_YES:
        return -size * yes_price
    if side == Side.SELL_YES:
        return size * yes_price
    if side == Side.BUY_NO:
        return -size * no_price
    if side == Side.SELL_NO:
        return size * no_price
    return 0.0


@dataclass(frozen=True)
class SkillEstimate:
    trader_id: str
    market_id: str
    as_of: datetime
    weighted_score: float
    weighted_history_notional: float


class TraderSkillEstimator:
    def __init__(
        self,
        markets: dict[str, Market],
        settlements: list[TraderMarketSettlement],
        min_weighted_history: float = 0.0,
        similarity_config: SimilarityConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        embedding_cache_dir: str | None = None,
    ) -> None:
        self.markets = markets
        self.min_weighted_history = min_weighted_history
        self.similarity_config = similarity_config or SimilarityConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.embedder = build_embedder(self.embedding_config)
        self.embedding_cache_dir = embedding_cache_dir
        self.market_ids = list(markets.keys())
        questions = [markets[mid].question for mid in self.market_ids]
        self.market_vectors = self._load_or_build_market_vectors(questions)
        self.market_index = {mid: idx for idx, mid in enumerate(self.market_ids)}
        self.by_trader: dict[str, list[TraderMarketSettlement]] = {}
        for row in settlements:
            self.by_trader.setdefault(row.trader_id, []).append(row)
        for rows in self.by_trader.values():
            rows.sort(key=lambda x: x.settled_at)

    def _embedding_signature(self, questions: list[str]) -> str:
        payload = {
            "backend": self.embedding_config.backend,
            "hashing_n_features": self.embedding_config.hashing_n_features,
            "st_model_name": self.embedding_config.st_model_name,
            "st_normalize_embeddings": self.embedding_config.st_normalize_embeddings,
            "market_ids": self.market_ids,
            "questions": questions,
        }
        raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _load_or_build_market_vectors(self, questions: list[str]) -> np.ndarray:
        if not self.embedding_cache_dir:
            return self.embedder.embed_texts(questions)
        cache_root = Path(self.embedding_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        signature = self._embedding_signature(questions)
        cache_path = cache_root / f"market_embeddings_{signature}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            cached_ids = cached["market_ids"].tolist()
            if cached_ids == self.market_ids:
                return cached["vectors"]
        vectors = self.embedder.embed_texts(questions)
        np.savez_compressed(
            cache_path,
            market_ids=np.array(self.market_ids, dtype=object),
            vectors=vectors,
        )
        return vectors

    def estimate(self, trader_id: str, target_market_id: str, as_of: datetime) -> SkillEstimate:
        rows = self.by_trader.get(trader_id, [])
        visible = [x for x in rows if x.settled_at <= as_of and x.market_id != target_market_id]
        if not visible:
            return SkillEstimate(
                trader_id=trader_id,
                market_id=target_market_id,
                as_of=as_of,
                weighted_score=0.0,
                weighted_history_notional=0.0,
            )

        target_idx = self.market_index.get(target_market_id)
        if target_idx is None:
            return SkillEstimate(trader_id, target_market_id, as_of, 0.0, 0.0)
        target_vec = self.market_vectors[target_idx]

        usable_rows = [row for row in visible if row.market_id in self.market_index]
        if not usable_rows:
            return SkillEstimate(trader_id, target_market_id, as_of, 0.0, 0.0)
        history_idx = [self.market_index[row.market_id] for row in usable_rows]
        history_vecs = self.market_vectors[history_idx]
        sim = self.embedder.pairwise_similarity(
            query_vec=target_vec,
            history_vecs=history_vecs,
            config=self.similarity_config,
        )

        history_notional = np.array([row.notional for row in usable_rows], dtype=float)
        history_score = np.array([row.score for row in usable_rows], dtype=float)

        weights = sim * history_notional
        denom = float(weights.sum())
        if denom <= 0.0:
            return SkillEstimate(trader_id, target_market_id, as_of, 0.0, 0.0)
        weighted_score = float((weights * history_score).sum() / denom)
        if denom < self.min_weighted_history:
            weighted_score = 0.0
        return SkillEstimate(
            trader_id=trader_id,
            market_id=target_market_id,
            as_of=as_of,
            weighted_score=weighted_score,
            weighted_history_notional=denom,
        )


def build_trader_market_settlements(
    trades: list[TradeEvent],
    resolutions: dict[str, Direction],
    resolution_ts: dict[str, datetime],
    score_clip: float = 1.0,
) -> list[TraderMarketSettlement]:
    """
    Computes per-trader per-market score using final settlement only.
    score = clip(pnl / notional, -score_clip, score_clip)
    """
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for t in trades:
        key = (t.trader_id, t.market_id)
        state = grouped.setdefault(key, {"yes_shares": 0.0, "cashflow": 0.0, "notional": 0.0})
        state["yes_shares"] += side_to_yes_shares(t.side, t.size)
        state["cashflow"] += side_to_cashflow(t.side, t.size, t.price_yes)
        state["notional"] += abs(side_to_cashflow(t.side, t.size, t.price_yes))

    rows: list[TraderMarketSettlement] = []
    for (trader_id, market_id), state in grouped.items():
        if market_id not in resolutions or market_id not in resolution_ts:
            continue
        notional = state["notional"]
        if notional <= 0:
            continue
        payout_per_yes = 1.0 if resolutions[market_id] == Direction.YES else 0.0
        payout = state["yes_shares"] * payout_per_yes
        pnl = state["cashflow"] + payout
        score = float(np.clip(pnl / notional, -score_clip, score_clip))
        rows.append(
            TraderMarketSettlement(
                trader_id=trader_id,
                market_id=market_id,
                score=score,
                notional=notional,
                settled_at=resolution_ts[market_id],
            )
        )
    rows.sort(key=lambda x: x.settled_at)
    return rows
