from __future__ import annotations

from collections.abc import Iterable
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    supporting_markets: int = 0
    effective_history_markets: float = 0.0
    mean_similarity: float = 0.0
    positive_history_weight_fraction: float = 0.0
    weighted_score_std: float = 0.0


class TraderSkillEstimator:
    def __init__(
        self,
        markets: dict[str, Market],
        settlements: list[TraderMarketSettlement],
        min_weighted_history: float = 0.0,
        similarity_config: SimilarityConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        embedding_cache_dir: str | None = None,
        similarity_mode: str = "semantic",
        embedding_bootstrap_path: str | None = None,
    ) -> None:
        self.markets = markets
        self.min_weighted_history = min_weighted_history
        self.similarity_config = similarity_config or SimilarityConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.similarity_mode = similarity_mode
        if self.similarity_mode not in {"semantic", "uniform", "category"}:
            raise ValueError(f"Unsupported similarity mode: {self.similarity_mode}")
        # Construct a text model only when vectors must actually be encoded.
        # Reproduction runs already have cached market vectors and only need
        # deterministic cosine similarity at inference time.
        self.embedder = None
        self.embedding_cache_dir = embedding_cache_dir
        self.embedding_bootstrap_path = embedding_bootstrap_path
        self.market_ids = list(markets.keys())
        questions = [markets[mid].question for mid in self.market_ids]
        self.market_vectors = (
            self._load_or_build_market_vectors(questions)
            if self.similarity_mode == "semantic"
            else np.zeros((len(questions), 0), dtype=float)
        )
        self.market_index = {mid: idx for idx, mid in enumerate(self.market_ids)}
        self.by_trader: dict[str, list[TraderMarketSettlement]] = {}
        for row in settlements:
            self.by_trader.setdefault(row.trader_id, []).append(row)
        for rows in self.by_trader.values():
            rows.sort(key=lambda x: x.settled_at)
        self.settlement_times = {
            trader_id: [row.settled_at for row in rows]
            for trader_id, rows in self.by_trader.items()
        }
        self._estimate_cache: dict[
            tuple[str, str, int],
            tuple[float, float, int, float, float, float, float],
        ] = {}
        self._estimate_cache_max_entries = 500_000

    @staticmethod
    def _empty_estimate(
        trader_id: str,
        target_market_id: str,
        as_of: datetime,
    ) -> SkillEstimate:
        return SkillEstimate(trader_id, target_market_id, as_of, 0.0, 0.0)

    def _embedding_signature(self, questions: list[str], include_runtime_options: bool = True) -> str:
        payload = {
            "backend": self.embedding_config.backend,
            "hashing_n_features": self.embedding_config.hashing_n_features,
            "st_model_name": self.embedding_config.st_model_name,
            "st_normalize_embeddings": self.embedding_config.st_normalize_embeddings,
            "market_ids": self.market_ids,
            "questions": questions,
        }
        if include_runtime_options:
            payload["st_trust_remote_code"] = self.embedding_config.st_trust_remote_code
        raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _load_or_build_market_vectors(self, questions: list[str]) -> np.ndarray:
        if not self.embedding_cache_dir:
            self.embedder = build_embedder(self.embedding_config)
            return self.embedder.embed_texts(questions)
        cache_root = Path(self.embedding_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        signature = self._embedding_signature(questions)
        cache_path = cache_root / f"market_embeddings_{signature}.npz"
        legacy_signature = self._embedding_signature(questions, include_runtime_options=False)
        legacy_cache_path = cache_root / f"market_embeddings_{legacy_signature}.npz"
        for candidate_path in [cache_path, legacy_cache_path]:
            if candidate_path.exists():
                cached = np.load(candidate_path, allow_pickle=True)
                cached_ids = cached["market_ids"].tolist()
                if cached_ids == self.market_ids:
                    return cached["vectors"]

        if self.embedding_bootstrap_path:
            bootstrap = np.load(self.embedding_bootstrap_path, allow_pickle=True)
            bootstrap_ids = bootstrap["market_ids"].tolist()
            bootstrap_index = {mid: idx for idx, mid in enumerate(bootstrap_ids)}
            if all(mid in bootstrap_index for mid in self.market_ids):
                indices = [bootstrap_index[mid] for mid in self.market_ids]
                vectors = bootstrap["vectors"][indices].astype(np.float32, copy=False)
                np.savez(
                    cache_path,
                    market_ids=np.array(self.market_ids, dtype=object),
                    vectors=vectors,
                )
                print(
                    f"Embedding cache: bootstrapped {len(self.market_ids)} markets "
                    f"from {self.embedding_bootstrap_path}",
                    flush=True,
                )
                return vectors

        if self.embedder is None:
            self.embedder = build_embedder(self.embedding_config)

        chunk_size = max(1, int(self.embedding_config.st_cache_chunk_size))
        parts_dir = cache_root / f"market_embeddings_{signature}_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_paths: list[Path] = []
        total = len(questions)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            part_path = parts_dir / f"part_{start:08d}_{end:08d}.npz"
            part_paths.append(part_path)
            if part_path.exists():
                cached_part = np.load(part_path, allow_pickle=True)
                if cached_part["market_ids"].tolist() == self.market_ids[start:end]:
                    print(f"Embedding cache: loaded chunk {start}:{end}", flush=True)
                    continue
            print(f"Embedding cache: encoding chunk {start}:{end} of {total}", flush=True)
            part_vectors = self.embedder.embed_texts(questions[start:end])
            np.savez(
                part_path,
                market_ids=np.array(self.market_ids[start:end], dtype=object),
                vectors=part_vectors,
            )

        vectors = np.concatenate(
            [np.load(path, allow_pickle=True)["vectors"] for path in part_paths],
            axis=0,
        )
        np.savez(
            cache_path,
            market_ids=np.array(self.market_ids, dtype=object),
            vectors=vectors,
        )
        return vectors

    def estimate(self, trader_id: str, target_market_id: str, as_of: datetime) -> SkillEstimate:
        rows = self.by_trader.get(trader_id, [])
        visible_count = bisect_right(self.settlement_times.get(trader_id, []), as_of)
        if visible_count == 0:
            return self._empty_estimate(trader_id, target_market_id, as_of)
        cache_key = (trader_id, target_market_id, visible_count)
        cached = self._estimate_cache.get(cache_key)
        if cached is not None:
            return SkillEstimate(trader_id, target_market_id, as_of, *cached)
        visible = [x for x in rows[:visible_count] if x.market_id != target_market_id]
        if not visible:
            return self._empty_estimate(trader_id, target_market_id, as_of)

        target_idx = self.market_index.get(target_market_id)
        if target_idx is None:
            return self._empty_estimate(trader_id, target_market_id, as_of)
        usable_rows = [row for row in visible if row.market_id in self.market_index]
        if not usable_rows:
            return self._empty_estimate(trader_id, target_market_id, as_of)
        if self.similarity_mode == "uniform":
            sim = np.ones(len(usable_rows), dtype=float)
        elif self.similarity_mode == "category":
            target_category = self.markets[target_market_id].category
            sim = np.array(
                [1.0 if self.markets[row.market_id].category == target_category else 0.0 for row in usable_rows],
                dtype=float,
            )
        else:
            target_vec = self.market_vectors[target_idx]
            history_idx = [self.market_index[row.market_id] for row in usable_rows]
            history_vecs = self.market_vectors[history_idx]
            sim = self._pairwise_similarity(target_vec, history_vecs)

        history_notional = np.array([row.notional for row in usable_rows], dtype=float)
        history_score = np.array([row.score for row in usable_rows], dtype=float)

        weights = sim * history_notional
        denom = float(weights.sum())
        if denom <= 0.0:
            return self._empty_estimate(trader_id, target_market_id, as_of)
        weighted_score = float((weights * history_score).sum() / denom)
        positive_weights = weights[weights > 0.0]
        supporting_markets = int(len(positive_weights))
        effective_history_markets = (
            float(denom**2 / np.square(positive_weights).sum())
            if supporting_markets > 0
            else 0.0
        )
        history_notional_sum = float(history_notional.sum())
        mean_similarity = (
            float((sim * history_notional).sum() / history_notional_sum)
            if history_notional_sum > 0.0
            else 0.0
        )
        positive_history_weight_fraction = float(weights[history_score > 0.0].sum() / denom)
        weighted_score_std = float(
            np.sqrt((weights * np.square(history_score - weighted_score)).sum() / denom)
        )
        if denom < self.min_weighted_history:
            weighted_score = 0.0
        result = SkillEstimate(
            trader_id=trader_id,
            market_id=target_market_id,
            as_of=as_of,
            weighted_score=weighted_score,
            weighted_history_notional=denom,
            supporting_markets=supporting_markets,
            effective_history_markets=effective_history_markets,
            mean_similarity=mean_similarity,
            positive_history_weight_fraction=positive_history_weight_fraction,
            weighted_score_std=weighted_score_std,
        )
        if len(self._estimate_cache) >= self._estimate_cache_max_entries:
            self._estimate_cache.clear()
        self._estimate_cache[cache_key] = (
            weighted_score,
            denom,
            supporting_markets,
            effective_history_markets,
            mean_similarity,
            positive_history_weight_fraction,
            weighted_score_std,
        )
        return result

    def market_similarity(self, left_market_id: str, right_market_id: str) -> float:
        """Return non-negative semantic similarity for point-in-time risk grouping."""
        if left_market_id == right_market_id:
            return 1.0
        if self.similarity_mode != "semantic":
            return 0.0
        left_idx = self.market_index.get(left_market_id)
        right_idx = self.market_index.get(right_market_id)
        if left_idx is None or right_idx is None:
            return 0.0
        return float(
            self._pairwise_similarity(
                self.market_vectors[left_idx],
                self.market_vectors[[right_idx]],
            )[0]
        )

    def _pairwise_similarity(
        self,
        query_vec: np.ndarray,
        history_vecs: np.ndarray,
    ) -> np.ndarray:
        if history_vecs.size == 0:
            return np.array([], dtype=float)
        sim = cosine_similarity(query_vec.reshape(1, -1), history_vecs).reshape(-1)
        if self.similarity_config.positive_similarity_only:
            sim = np.maximum(sim, self.similarity_config.similarity_floor)
        return sim


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
        state = grouped.setdefault(
            key,
            {"yes_shares": 0.0, "no_shares": 0.0, "cashflow": 0.0, "notional": 0.0},
        )
        if t.side == Side.BUY_YES:
            state["yes_shares"] += t.size
        elif t.side == Side.SELL_YES:
            state["yes_shares"] -= t.size
        elif t.side == Side.BUY_NO:
            state["no_shares"] += t.size
        elif t.side == Side.SELL_NO:
            state["no_shares"] -= t.size
        state["cashflow"] += side_to_cashflow(t.side, t.size, t.price_yes)
        state["notional"] += abs(side_to_cashflow(t.side, t.size, t.price_yes))

    rows: list[TraderMarketSettlement] = []
    for (trader_id, market_id), state in grouped.items():
        if market_id not in resolutions or market_id not in resolution_ts:
            continue
        notional = state["notional"]
        if notional <= 0:
            continue
        payout = (
            state["yes_shares"]
            if resolutions[market_id] == Direction.YES
            else state["no_shares"]
        )
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


def build_trader_market_settlements_from_records(
    records: Iterable[dict[str, Any]],
    resolutions: dict[str, Direction],
    resolution_ts: dict[str, datetime],
    score_clip: float = 1.0,
) -> list[TraderMarketSettlement]:
    """Memory-conscious settlement aggregation for large normalized datasets.

    This is equivalent to :func:`build_trader_market_settlements`, but avoids
    materializing one ``TradeEvent`` dataclass per source row.  Only resolved
    markets are aggregated because unresolved positions cannot contribute a
    historically observable skill label.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    valid_sides = {side.value: side for side in Side}
    for raw in records:
        market_id = str(raw.get("market_id") or "")
        if market_id not in resolutions or market_id not in resolution_ts:
            continue
        side = valid_sides.get(str(raw.get("side") or ""))
        if side is None:
            continue
        trader_id = str(raw.get("trader_id") or "")
        if not trader_id:
            continue
        size = float(raw.get("size") or 0.0)
        if size <= 0.0:
            continue
        yes_price = float(raw.get("price_yes", 0.5))
        # State layout: yes shares, no shares, cashflow, absolute notional.
        state = grouped.setdefault((trader_id, market_id), [0.0, 0.0, 0.0, 0.0])
        if side == Side.BUY_YES:
            state[0] += size
        elif side == Side.SELL_YES:
            state[0] -= size
        elif side == Side.BUY_NO:
            state[1] += size
        elif side == Side.SELL_NO:
            state[1] -= size
        cashflow = side_to_cashflow(side, size, yes_price)
        state[2] += cashflow
        state[3] += abs(cashflow)

    rows: list[TraderMarketSettlement] = []
    for (trader_id, market_id), state in grouped.items():
        notional = state[3]
        if notional <= 0.0:
            continue
        payout = state[0] if resolutions[market_id] == Direction.YES else state[1]
        pnl = state[2] + payout
        rows.append(
            TraderMarketSettlement(
                trader_id=trader_id,
                market_id=market_id,
                score=float(np.clip(pnl / notional, -score_clip, score_clip)),
                notional=notional,
                settled_at=resolution_ts[market_id],
            )
        )
    rows.sort(key=lambda row: row.settled_at)
    return rows
