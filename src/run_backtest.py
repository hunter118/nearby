from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import yaml

from alpha.trader_skill import TraderSkillEstimator, build_trader_market_settlements
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_markets, build_resolution_events, build_timeline, build_trade_events
from data.polymarket_client import PolymarketClient
from eval.metrics import evaluate_closed_positions
from features.embeddings import EmbeddingConfig, SimilarityConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_load(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _cache_save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Polymarket event-driven backtest.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    cache_cfg = cfg.get("cache", {})
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_dir = Path(cache_cfg.get("dir", ".cache/backtest"))
    client = PolymarketClient(
        host=data_cfg["host"],
        gamma_host=data_cfg["gamma_host"],
        data_api_host=data_cfg["data_api_host"],
    )

    markets_cache_key = _cache_key(
        {
            "markets_limit": int(data_cfg["markets_limit"]),
            "open_market_pages": int(data_cfg["open_market_pages"]),
            "closed_market_pages": int(data_cfg["closed_market_pages"]),
            "gamma_host": data_cfg["gamma_host"],
        }
    )
    markets_cache_path = cache_dir / f"markets_{markets_cache_key}.pkl"
    market_records = _cache_load(markets_cache_path) if cache_enabled else None
    if market_records is None:
        market_records = {}
        for row in client.iter_markets(
            page_size=int(data_cfg["markets_limit"]),
            max_pages=int(data_cfg["open_market_pages"]),
            closed=False,
        ):
            market_records[row["market_id"]] = row
        for row in client.iter_markets(
            page_size=int(data_cfg["markets_limit"]),
            max_pages=int(data_cfg["closed_market_pages"]),
            closed=True,
        ):
            market_records[row["market_id"]] = row
        if cache_enabled:
            _cache_save(markets_cache_path, market_records)
            print(f"Cache: saved markets to {markets_cache_path}")
    else:
        print(f"Cache: loaded markets from {markets_cache_path}")
    markets = build_markets(market_records.values())

    trade_fetch_mode = data_cfg.get("trade_fetch_mode", "global")
    trades_limit = int(data_cfg["trades_limit"])
    trades_cache_key = _cache_key(
        {
            "trade_fetch_mode": trade_fetch_mode,
            "trades_limit": trades_limit,
            "trade_market_count": int(data_cfg.get("trade_market_count", 200)),
            "per_market_trades_limit": int(data_cfg.get("per_market_trades_limit", 300)),
            "data_api_host": data_cfg["data_api_host"],
            "markets_scope_key": markets_cache_key,
        }
    )
    trades_cache_path = cache_dir / f"normalized_trades_{trades_cache_key}.pkl"
    normalized_trades = _cache_load(trades_cache_path) if cache_enabled else None
    if normalized_trades is None:
        if trade_fetch_mode == "by_markets":
            market_count = int(data_cfg.get("trade_market_count", 200))
            per_market_limit = int(data_cfg.get("per_market_trades_limit", 300))
            ranked_market_ids = [
                m.market_id
                for m in sorted(
                    markets.values(),
                    key=lambda x: x.volume,
                    reverse=True,
                )
            ]
            trade_raw: list[dict] = []
            for market_id in ranked_market_ids[:market_count]:
                remaining = trades_limit - len(trade_raw)
                if remaining <= 0:
                    break
                rows = client.fetch_trades(
                    market_id=market_id,
                    limit=min(per_market_limit, remaining),
                    offset=0,
                )
                trade_raw.extend(rows)
        else:
            trade_raw = client.fetch_trades(limit=trades_limit)
        keyed_trades: dict[str, dict] = {}
        for row in trade_raw:
            key = str(
                row.get("id")
                or row.get("transactionHash")
                or f"{row.get('proxyWallet')}_{row.get('conditionId')}_{row.get('timestamp')}_{row.get('size')}"
            )
            keyed_trades[key] = row
        deduped_trade_raw = list(keyed_trades.values())
        normalized_trades = [client.normalize_trade(x) for x in deduped_trade_raw]
        if cache_enabled:
            _cache_save(trades_cache_path, normalized_trades)
            print(f"Cache: saved normalized trades to {trades_cache_path}")
    else:
        print(f"Cache: loaded normalized trades from {trades_cache_path}")
    trade_events = build_trade_events(normalized_trades)
    resolution_events = build_resolution_events(markets)
    timeline = build_timeline(trade_events, resolution_events)

    resolution_map = {
        x.market_id: x.resolution
        for x in resolution_events
    }
    resolution_ts = {x.market_id: x.resolved_at for x in resolution_events}
    settlements = build_trader_market_settlements(
        trades=trade_events,
        resolutions=resolution_map,
        resolution_ts=resolution_ts,
    )
    trade_ts_min = min((x.timestamp for x in trade_events), default=None)
    trade_ts_max = max((x.timestamp for x in trade_events), default=None)
    resolution_ts_min = min((x.resolved_at for x in resolution_events), default=None)
    resolution_ts_max = max((x.resolved_at for x in resolution_events), default=None)
    unique_trade_markets = len({x.market_id for x in trade_events})
    unique_traders = len({x.trader_id for x in trade_events})
    covered_markets_with_resolution = len({x.market_id for x in trade_events if x.market_id in resolution_map})
    print(
        "Dataset stats:",
        f"markets={len(markets)}",
        f"trades={len(trade_events)}",
        f"resolutions={len(resolution_events)}",
        f"settlements={len(settlements)}",
    )
    print(
        "Coverage:",
        f"trade_time_range=[{trade_ts_min} -> {trade_ts_max}]",
        f"resolution_time_range=[{resolution_ts_min} -> {resolution_ts_max}]",
        f"unique_trade_markets={unique_trade_markets}",
        f"unique_traders={unique_traders}",
        f"trade_markets_with_resolution={covered_markets_with_resolution}",
    )
    similarity_cfg = SimilarityConfig(
        positive_similarity_only=cfg["strategy"]["positive_similarity_only"],
        similarity_floor=float(cfg["strategy"]["similarity_floor"]),
    )
    embedding_cfg_raw = cfg.get("embeddings", {})
    embedding_cfg = EmbeddingConfig(
        backend=str(embedding_cfg_raw.get("backend", "hashing")),
        hashing_n_features=int(embedding_cfg_raw.get("hashing_n_features", 2**12)),
        st_model_name=str(embedding_cfg_raw.get("st_model_name", "BAAI/bge-large-en-v1.5")),
        st_device=str(embedding_cfg_raw.get("st_device", "auto")),
        st_batch_size=int(embedding_cfg_raw.get("st_batch_size", 64)),
        st_normalize_embeddings=bool(embedding_cfg_raw.get("st_normalize_embeddings", True)),
        st_trust_remote_code=bool(embedding_cfg_raw.get("st_trust_remote_code", False)),
        st_cache_chunk_size=int(embedding_cfg_raw.get("st_cache_chunk_size", 4096)),
    )
    skill_estimator = TraderSkillEstimator(
        markets=markets,
        settlements=settlements,
        min_weighted_history=float(cfg["strategy"]["min_weighted_history"]),
        similarity_config=similarity_cfg,
        embedding_config=embedding_cfg,
        embedding_cache_dir=str(cache_dir / "embeddings") if cache_enabled else None,
    )
    backtest_cfg = BacktestConfig(
        delay_seconds=int(cfg["execution"]["delay_seconds"]),
        skill_threshold=float(cfg["strategy"]["skill_threshold"]),
        consensus_threshold=float(cfg["strategy"]["consensus_threshold"]),
        min_skilled_traders=int(cfg["strategy"]["min_skilled_traders"]),
        max_single_trader_weight=float(cfg["strategy"]["max_single_trader_weight"]),
        min_edge=float(cfg["strategy"]["min_edge"]),
        min_user_volume=float(cfg["strategy"]["min_user_volume"]),
        max_trades_per_market=int(cfg["strategy"]["max_trades_per_market"]),
        stable_min_price=float(cfg["execution"]["stable_min_price"]),
        lottery_min_price=float(cfg["execution"]["lottery_min_price"]),
        lottery_max_price=float(cfg["execution"]["lottery_max_price"]),
        stable_balance_fraction=float(cfg["risk"]["stable_balance_fraction"]),
        lottery_lot_size=float(cfg["risk"]["lottery_lot_size"]),
        lottery_max_exposure_fraction=float(cfg["risk"]["lottery_max_exposure_fraction"]),
        min_days_to_resolution=float(cfg["risk"]["min_days_to_resolution"]),
        max_days_to_resolution=float(cfg["risk"]["max_days_to_resolution"]),
        trade_fee_bps=float(cfg["execution"]["trade_fee_bps"]),
        slippage_bps=float(cfg["execution"]["slippage_bps"]),
        min_entry_price=float(cfg["execution"]["min_entry_price"]),
        max_entry_price=float(cfg["execution"]["max_entry_price"]),
        dynamic_price_at_consensus=float(cfg["execution"]["dynamic_price_at_consensus"]),
        dynamic_price_at_high_confidence=float(cfg["execution"]["dynamic_price_at_high_confidence"]),
        dynamic_high_confidence=float(cfg["execution"]["dynamic_high_confidence"]),
        max_market_fraction=float(cfg["risk"]["max_market_fraction"]),
        max_balance_fraction=float(cfg["risk"]["max_balance_fraction"]),
        max_loss_per_trade_fraction=float(cfg["risk"]["max_loss_per_trade_fraction"]),
        min_ticket_size=float(cfg["risk"]["min_ticket_size"]),
        initial_balance=float(cfg["risk"]["initial_balance"]),
        position_sizing=str(cfg["risk"].get("position_sizing", "fixed_fraction")),
        target_exposure_fraction=float(cfg["risk"].get("target_exposure_fraction", 0.0)),
        cash_buffer_fraction=float(cfg["risk"].get("cash_buffer_fraction", 0.0)),
        min_target_order_fraction=float(cfg["risk"].get("min_target_order_fraction", 0.0)),
        max_target_order_fraction=float(cfg["risk"].get("max_target_order_fraction", 1.0)),
        annualized_edge_multiplier=float(cfg["risk"].get("annualized_edge_multiplier", 1.0)),
    )
    engine = EventDrivenBacktester(
        markets=markets,
        timeline=timeline,
        skill_estimator=skill_estimator,
        config=backtest_cfg,
    )
    result = engine.run()
    metrics = evaluate_closed_positions(
        closed_positions=result["closed_positions"],
        initial_balance=backtest_cfg.initial_balance,
        final_balance=result["balance"],
    )

    print("Backtest finished")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    stable_closed = [x for x in result["closed_positions"] if x.avg_entry_price > backtest_cfg.stable_min_price]
    lottery_closed = [
        x
        for x in result["closed_positions"]
        if backtest_cfg.lottery_min_price <= x.avg_entry_price <= backtest_cfg.lottery_max_price
    ]
    stable_win_rate = (
        sum(1 for x in stable_closed if x.pnl > 0) / len(stable_closed)
        if stable_closed
        else 0.0
    )
    lottery_win_rate = (
        sum(1 for x in lottery_closed if x.pnl > 0) / len(lottery_closed)
        if lottery_closed
        else 0.0
    )
    print(
        "Bucket win rates:",
        f"stable_gt_{backtest_cfg.stable_min_price}={stable_win_rate}",
        f"lottery_{backtest_cfg.lottery_min_price}_to_{backtest_cfg.lottery_max_price}={lottery_win_rate}",
        f"stable_trades={len(stable_closed)}",
        f"lottery_trades={len(lottery_closed)}",
    )
    print(
        "Portfolio snapshot:",
        f"cash_balance={result['balance']}",
        f"open_positions={len(result['open_positions'])}",
        f"open_notional={result['open_notional']}",
        f"open_market_value={result['open_market_value']}",
        f"open_unrealized_pnl={result['open_unrealized_pnl']}",
        f"total_equity={result['total_equity']}",
    )


if __name__ == "__main__":
    main()
