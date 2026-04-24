from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from alpha.trader_skill import TraderSkillEstimator, build_trader_market_settlements
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_markets, build_resolution_events, build_timeline, build_trade_events
from data.polymarket_client import PolymarketClient
from features.embeddings import EmbeddingConfig, SimilarityConfig
from run_backtest import _cache_key, _cache_load, _cache_save


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot total equity over time.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--csv-out", type=str, default="artifacts/equity_curve.csv")
    parser.add_argument("--png-out", type=str, default="artifacts/equity_curve.png")
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

    trade_events = build_trade_events(normalized_trades)
    resolution_events = build_resolution_events(markets)
    timeline = build_timeline(trade_events, resolution_events)

    resolution_map = {x.market_id: x.resolution for x in resolution_events}
    resolution_ts = {x.market_id: x.resolved_at for x in resolution_events}
    settlements = build_trader_market_settlements(
        trades=trade_events,
        resolutions=resolution_map,
        resolution_ts=resolution_ts,
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
    )

    result = EventDrivenBacktester(
        markets=markets,
        timeline=timeline,
        skill_estimator=skill_estimator,
        config=backtest_cfg,
    ).run()

    curve = pd.DataFrame(result["equity_curve"])
    curve["ts"] = pd.to_datetime(curve["ts"])
    # Plot from the strategy's first actual fill time.
    fills = result.get("fills", [])
    if fills:
        first_fill_ts = min(x.filled_at for x in fills)
        curve = curve[curve["ts"] >= pd.Timestamp(first_fill_ts)].copy()
    if curve.empty:
        raise RuntimeError("Equity curve is empty after applying first-fill filter.")
    csv_out = Path(args.csv_out)
    png_out = Path(args.png_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    png_out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(csv_out, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(curve["ts"], curve["total_equity"], label="Total Equity")
    plt.plot(curve["ts"], curve["cash_balance"], label="Cash Balance", alpha=0.6)
    plt.title("Backtest Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(png_out, dpi=150)
    print(f"Saved equity curve csv: {csv_out}")
    print(f"Saved equity curve chart: {png_out}")


if __name__ == "__main__":
    main()
