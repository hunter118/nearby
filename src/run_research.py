from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import gc
import hashlib
from itertools import chain
import json
import pickle
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from alpha.trader_skill import (
    TraderSkillEstimator,
    build_trader_market_settlements_from_records,
)
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_markets, build_resolution_events, build_timeline, build_trade_events
from data.polymarket_client import PolymarketClient
from features.embeddings import EmbeddingConfig, SimilarityConfig


THREAD_LOCAL = threading.local()


def _load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(path)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp_path.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client(data_cfg: dict[str, Any]) -> PolymarketClient:
    client = getattr(THREAD_LOCAL, "polymarket_client", None)
    if client is None:
        client = PolymarketClient(
            host=data_cfg["host"],
            gamma_host=data_cfg["gamma_host"],
            data_api_host=data_cfg["data_api_host"],
            timeout_seconds=float(data_cfg.get("timeout_seconds", 30)),
        )
        THREAD_LOCAL.polymarket_client = client
    return client


def _with_retries(fn: Callable[[], Any], attempts: int = 6):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network errors are recorded in the manifest
            last_error = exc
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def _fetch_metadata(
    market_ids: list[str],
    data_cfg: dict[str, Any],
    workers: int,
    cache_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    cached: dict[str, dict[str, Any]] = _load_pickle(cache_path) if cache_path.exists() else {}
    errors: dict[str, str] = {}
    missing = [mid for mid in market_ids if mid not in cached]
    if not missing:
        return cached, errors

    def fetch_one(mid: str):
        return _with_retries(lambda: _client(data_cfg).fetch_market_by_condition(mid))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, mid): mid for mid in missing}
        for completed, future in enumerate(as_completed(futures), 1):
            mid = futures[future]
            try:
                row = future.result()
                if row:
                    cached[mid] = row
                else:
                    errors[mid] = "not_found"
            except Exception as exc:
                errors[mid] = repr(exc)
            if completed % 250 == 0:
                _save_pickle(cache_path, cached)
                print(f"Metadata: {completed}/{len(missing)} fetched", flush=True)
    _save_pickle(cache_path, cached)
    return cached, errors


def _raw_trade_key(row: dict[str, Any]) -> str:
    payload = (
        row.get("transactionHash") or row.get("transaction_hash") or "",
        row.get("proxyWallet") or "",
        row.get("asset") or "",
        row.get("conditionId") or "",
        row.get("timestamp") or "",
        row.get("side") or "",
        row.get("size") or "",
    )
    return "|".join(str(x) for x in payload)


def _fetch_market_window(
    market_id: str,
    start_epoch: int,
    end_epoch: int,
    data_cfg: dict[str, Any],
    max_rows: int,
) -> tuple[list[dict[str, Any]], bool]:
    client = _client(data_cfg)

    def fetch_window(start: int, end: int, budget: int) -> tuple[list[dict[str, Any]], bool]:
        if budget <= 0 or end < start:
            return [], True
        request_limit = min(20000, budget)
        rows = _with_retries(
            lambda: client.fetch_trades(
                market_id=market_id,
                limit=request_limit,
                start=start,
                end=end,
                taker_only=True,
            )
        )
        if len(rows) < request_limit or end - start <= 1:
            return rows, False
        midpoint = (start + end) // 2
        left, left_truncated = fetch_window(start, midpoint, budget)
        remaining = budget - len(left)
        right, right_truncated = fetch_window(midpoint + 1, end, remaining)
        return left + right, left_truncated or right_truncated or remaining <= 0

    rows, truncated = fetch_window(start_epoch, end_epoch, max_rows)
    deduped = {_raw_trade_key(row): row for row in rows}
    ordered = sorted(deduped.values(), key=lambda row: int(row.get("timestamp") or 0))
    return ordered[:max_rows], truncated or len(ordered) > max_rows


def _fetch_incremental_trades(
    market_ids: list[str],
    last_timestamp_by_market: dict[str, int],
    snapshot_end_epoch: int,
    data_cfg: dict[str, Any],
    workers: int,
    max_rows_per_market: int,
    cache_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], list[str]]:
    cached: dict[str, list[dict[str, Any]]] = _load_pickle(cache_path) if cache_path.exists() else {}
    errors: dict[str, str] = {}
    truncated: list[str] = []
    missing = [mid for mid in market_ids if mid not in cached]
    if not missing:
        return cached, errors, truncated

    def fetch_one(mid: str):
        start = last_timestamp_by_market.get(mid, 0) + 1
        return _fetch_market_window(mid, start, snapshot_end_epoch, data_cfg, max_rows_per_market)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, mid): mid for mid in missing}
        for completed, future in enumerate(as_completed(futures), 1):
            mid = futures[future]
            try:
                rows, was_truncated = future.result()
                cached[mid] = rows
                if was_truncated:
                    truncated.append(mid)
            except Exception as exc:
                errors[mid] = repr(exc)
            if completed % 200 == 0:
                _save_pickle(cache_path, cached)
                row_count = sum(len(rows) for rows in cached.values())
                print(
                    f"Incremental trades: {completed}/{len(missing)} markets, {row_count} rows",
                    flush=True,
                )
    _save_pickle(cache_path, cached)
    return cached, errors, truncated


def _dt_to_epoch(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp())


def _utc_naive_from_epoch(value: int) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)


def _build_backtest_config(cfg: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(
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
        signal_weighting=str(cfg["strategy"].get("signal_weighting", "skill_volume")),
        flow_lookback_seconds=cfg["strategy"].get("flow_lookback_seconds"),
        max_fill_participation=cfg["execution"].get("max_fill_participation"),
        enforce_risk_caps=bool(cfg["risk"].get("enforce_risk_caps", False)),
        equity_record_interval=int(cfg["research"].get("equity_record_interval", 1000)),
        min_directional_traders=int(cfg["strategy"].get("min_directional_traders", 1)),
        min_effective_directional_traders=float(
            cfg["strategy"].get("min_effective_directional_traders", 1.0)
        ),
        max_directional_trader_weight=float(
            cfg["strategy"].get("max_directional_trader_weight", 1.0)
        ),
        min_expert_effective_history_markets=float(
            cfg["strategy"].get("min_expert_effective_history_markets", 0.0)
        ),
        min_expert_mean_similarity=float(
            cfg["strategy"].get("min_expert_mean_similarity", 0.0)
        ),
        min_expert_positive_history_fraction=float(
            cfg["strategy"].get("min_expert_positive_history_fraction", 0.0)
        ),
        max_expert_score_std=float(cfg["strategy"].get("max_expert_score_std", float("inf"))),
        semantic_cluster_similarity_threshold=cfg["risk"].get(
            "semantic_cluster_similarity_threshold"
        ),
        max_semantic_cluster_exposure_fraction=float(
            cfg["risk"].get("max_semantic_cluster_exposure_fraction", 1.0)
        ),
        apply_market_volume_cap=bool(cfg["risk"].get("apply_market_volume_cap", True)),
        apply_balance_cap=bool(cfg["risk"].get("apply_balance_cap", True)),
        apply_loss_cap=bool(cfg["risk"].get("apply_loss_cap", True)),
        min_signal_mean_expert_history_markets=float(
            cfg["strategy"].get("min_signal_mean_expert_history_markets", 0.0)
        ),
        max_competitive_event_exposure_fraction=cfg["risk"].get(
            "max_competitive_event_exposure_fraction"
        ),
        max_position_exposure_fraction=cfg["risk"].get("max_position_exposure_fraction"),
    )


def _bootstrap_mean_return(closed_positions, draws: int = 2000) -> tuple[float | None, float | None]:
    returns = np.array(
        [position.pnl / position.notional for position in closed_positions if position.notional > 0],
        dtype=float,
    )
    if len(returns) < 2:
        return None, None
    rng = np.random.default_rng(20260808)
    means = np.empty(draws, dtype=float)
    for idx in range(draws):
        means[idx] = rng.choice(returns, size=len(returns), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_result(name: str, result: dict[str, Any], initial_balance: float) -> dict[str, Any]:
    closed = result["closed_positions"]
    fees = float(sum(fill.fee for fill in result["fills"]))
    gross_closed_pnl = float(sum(position.pnl for position in closed))
    trade_returns = [position.pnl / position.notional for position in closed if position.notional > 0]
    trade_pnls = np.array([position.pnl for position in closed], dtype=float)
    losing_pnls = trade_pnls[trade_pnls < 0.0]
    gross_profit = float(trade_pnls[trade_pnls > 0.0].sum()) if len(trade_pnls) else 0.0
    gross_loss = float(-losing_pnls.sum()) if len(losing_pnls) else 0.0
    tail_count = max(1, int(np.ceil(0.05 * len(trade_pnls)))) if len(trade_pnls) else 0
    expected_shortfall_5pct = (
        float(np.sort(trade_pnls)[:tail_count].mean()) if tail_count else 0.0
    )
    curve = pd.DataFrame(result["equity_curve"])
    if curve.empty:
        max_drawdown = 0.0
    else:
        running_max = curve["total_equity"].cummax()
        max_drawdown = float((curve["total_equity"] / running_max - 1.0).min())
    ci_low, ci_high = _bootstrap_mean_return(closed)
    return {
        "experiment": name,
        "fills": len(result["fills"]),
        "closed_positions": len(closed),
        "open_positions": len(result["open_positions"]),
        "win_rate": sum(position.pnl > 0 for position in closed) / len(closed) if closed else None,
        "loss_count": int(len(losing_pnls)),
        "gross_closed_pnl": gross_closed_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "fees": fees,
        "net_realized_pnl": gross_closed_pnl - fees,
        "mean_trade_return": float(np.mean(trade_returns)) if trade_returns else None,
        "median_trade_return": float(np.median(trade_returns)) if trade_returns else None,
        "worst_trade_pnl": float(min((p.pnl for p in closed), default=0.0)),
        "worst_trade_return": float(min(trade_returns, default=0.0)),
        "expected_shortfall_5pct_pnl": expected_shortfall_5pct,
        "largest_notional": float(max((p.notional for p in closed), default=0.0)),
        "mean_trade_return_ci_low": ci_low,
        "mean_trade_return_ci_high": ci_high,
        "cash": float(result["balance"]),
        "open_notional": float(result["open_notional"]),
        "open_market_value": float(result["open_market_value"]),
        "total_equity": float(result["total_equity"]),
        "total_return": float(result["total_equity"] / initial_balance - 1.0),
        "max_drawdown": max_drawdown,
    }


def _write_main_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fills = pd.DataFrame([asdict(row) for row in result["fills"]])
    closed = pd.DataFrame([asdict(row) for row in result["closed_positions"]])
    fills.to_csv(output_dir / "main_fills.csv", index=False)
    closed.to_csv(output_dir / "main_closed_positions.csv", index=False)

    curve = pd.DataFrame(result["equity_curve"])
    if curve.empty:
        return
    curve["ts"] = pd.to_datetime(curve["ts"])
    daily = curve.set_index("ts").resample("1D").last().dropna(how="all").reset_index()
    daily.to_csv(output_dir / "main_equity_daily.csv", index=False)
    plt.figure(figsize=(10, 5.5))
    plt.plot(daily["ts"], daily["total_equity"], label="Total equity", linewidth=2)
    plt.plot(daily["ts"], daily["cash_balance"], label="Cash", alpha=0.65)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USDC)")
    plt.title("Out-of-sample equity: frozen 2026-04-27 market cohort")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "main_equity.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen-cohort Polymarket research reproduction.")
    parser.add_argument("--config", default="config/research_2026_08_08.yaml")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    cache_dir = Path(cfg["research"]["cache_dir"])
    output_dir = Path(cfg["research"]["output_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_markets_path = Path(cfg["research"]["legacy_markets_cache"])
    old_trades_path = Path(cfg["research"]["legacy_trades_cache"])
    old_market_records: dict[str, dict[str, Any]] = _load_pickle(old_markets_path)
    old_normalized_trades: list[dict[str, Any]] = _load_pickle(old_trades_path)
    legacy_index_path = cache_dir / "legacy_market_index.pkl"
    if legacy_index_path.exists():
        legacy_index = _load_pickle(legacy_index_path)
        cohort_ids = legacy_index["cohort_ids"]
        last_timestamp_by_market = legacy_index["last_timestamp_by_market"]
    else:
        cohort_ids = sorted({row["market_id"] for row in old_normalized_trades})
        last_timestamp_by_market: dict[str, int] = {}
        for row in old_normalized_trades:
            market_id = row["market_id"]
            timestamp = row.get("timestamp")
            if timestamp is None:
                continue
            last_timestamp_by_market[market_id] = max(
                last_timestamp_by_market.get(market_id, 0),
                _dt_to_epoch(timestamp),
            )
        _save_pickle(
            legacy_index_path,
            {
                "cohort_ids": cohort_ids,
                "last_timestamp_by_market": last_timestamp_by_market,
            },
        )
    legacy_cutoff_epoch = max(last_timestamp_by_market.values())

    manifest_path = output_dir / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Early fetch checkpoints stored full error dictionaries in the main
        # manifest.  Keep only counts here and the detailed dictionaries in the
        # dedicated data_errors.json file.
        manifest.pop("metadata_errors", None)
        manifest.pop("incremental_trade_errors", None)
        snapshot_end_epoch = int(manifest["snapshot_end_epoch"])
    else:
        snapshot_end_epoch = int(time.time())
        manifest = {
            "snapshot_id": cfg["research"]["snapshot_id"],
            "snapshot_end_epoch": snapshot_end_epoch,
            "snapshot_end_utc": _utc_naive_from_epoch(snapshot_end_epoch).isoformat() + "Z",
            "legacy_cutoff_epoch": legacy_cutoff_epoch,
            "legacy_cutoff_utc": _utc_naive_from_epoch(legacy_cutoff_epoch).isoformat() + "Z",
            "legacy_market_cache": str(old_markets_path),
            "legacy_trade_cache": str(old_trades_path),
            "legacy_trade_cache_sha256": _sha256_file(old_trades_path),
            "cohort_definition": "2,993 markets selected before the holdout in the 2026-04-27 cache",
            "taker_only": True,
        }
        _save_json(manifest_path, manifest)

    metadata_raw, metadata_errors = _fetch_metadata(
        cohort_ids,
        data_cfg,
        int(cfg["research"]["fetch_workers"]),
        cache_dir / "gamma_markets_raw.pkl",
    )
    incremental_raw_path = cache_dir / "data_api_incremental_raw.pkl"
    incremental_normalized_path = cache_dir / "incremental_normalized.pkl"
    if incremental_raw_path.exists() and incremental_normalized_path.exists():
        incremental_raw = None
        incremental_normalized = None if args.fetch_only else _load_pickle(incremental_normalized_path)
        trade_errors: dict[str, str] = {}
        truncated_markets = list(manifest.get("incremental_truncated_markets", []))
    else:
        incremental_raw, trade_errors, truncated_markets = _fetch_incremental_trades(
            cohort_ids,
            last_timestamp_by_market,
            snapshot_end_epoch,
            data_cfg,
            int(cfg["research"]["fetch_workers"]),
            int(cfg["research"]["max_incremental_trades_per_market"]),
            incremental_raw_path,
        )
        incremental_normalized = []

    normalized_market_records: dict[str, dict[str, Any]] = {}
    asset_outcome_index: dict[str, int] = {}
    outcome_labels_by_market: dict[str, list[str]] = {}
    fallback_metadata = 0
    for mid in cohort_ids:
        if mid in metadata_raw:
            row = PolymarketClient.normalize_market(metadata_raw[mid])
            old_row = old_market_records.get(mid)
            if old_row and row.get("resolution") is None and metadata_raw[mid].get("closed") is True:
                row["resolution"] = old_row.get("resolution")
            if not row.get("created_at") and old_row:
                row["created_at"] = old_row.get("created_at")
        else:
            fallback_metadata += 1
            row = dict(old_market_records[mid])
            last_trade = _utc_naive_from_epoch(last_timestamp_by_market.get(mid, 0))
            scheduled = row.get("close_time")
            row["resolved_at"] = max(x for x in [scheduled, last_trade] if x is not None)
            row.setdefault("outcome_labels", [])
            row.setdefault("clob_token_ids", [])
        normalized_market_records[mid] = row
        outcome_labels_by_market[mid] = row.get("outcome_labels", [])
        for index, asset in enumerate(row.get("clob_token_ids", [])[:2]):
            asset_outcome_index[str(asset)] = index

    unknown_outcome_rows = 0
    if incremental_raw is not None:
        assert incremental_normalized is not None
        for rows in incremental_raw.values():
            for raw in rows:
                normalized = PolymarketClient.normalize_trade(
                    raw,
                    asset_outcome_index=asset_outcome_index,
                    outcome_labels_by_market=outcome_labels_by_market,
                )
                if normalized["side"] is None:
                    unknown_outcome_rows += 1
                    continue
                incremental_normalized.append(normalized)
        _save_pickle(incremental_normalized_path, incremental_normalized)
    elif incremental_normalized is not None:
        unknown_outcome_rows = int(manifest.get("unknown_outcome_rows_dropped", 0))

    incremental_trade_count = (
        len(incremental_normalized)
        if incremental_normalized is not None
        else int(manifest.get("incremental_trades", 0))
    )

    manifest.update(
        {
            "cohort_markets": len(cohort_ids),
            "metadata_fetched": len(metadata_raw),
            "metadata_fallbacks": fallback_metadata,
            "metadata_error_count": len(metadata_errors),
            "incremental_trades": incremental_trade_count,
            "incremental_trade_error_count": len(trade_errors),
            "incremental_truncated_markets": truncated_markets,
            "unknown_outcome_rows_dropped": unknown_outcome_rows,
        }
    )
    _save_json(
        output_dir / "data_errors.json",
        {"metadata_errors": metadata_errors, "incremental_trade_errors": trade_errors},
    )
    _save_json(manifest_path, manifest)
    if args.fetch_only:
        print(json.dumps(manifest, indent=2, default=str), flush=True)
        return

    assert incremental_normalized is not None
    excluded_market_ids = set(truncated_markets)
    latest_trade_by_market: dict[str, datetime] = {}
    for raw in chain(old_normalized_trades, incremental_normalized):
        market_id = str(raw.get("market_id") or "")
        timestamp = raw.get("timestamp")
        if market_id in excluded_market_ids or timestamp is None:
            continue
        latest_trade_by_market[market_id] = max(
            latest_trade_by_market.get(market_id, timestamp), timestamp
        )

    resolution_time_adjustments = 0
    for mid, row in normalized_market_records.items():
        resolved_at = row.get("resolved_at")
        last_trade = latest_trade_by_market.get(mid)
        if row.get("resolution") and resolved_at and last_trade and resolved_at < last_trade:
            row["resolved_at"] = last_trade + timedelta(seconds=1)
            resolution_time_adjustments += 1

    analysis_market_records = [
        row
        for mid, row in normalized_market_records.items()
        if mid not in excluded_market_ids
    ]
    markets = build_markets(analysis_market_records)
    resolution_events = build_resolution_events(markets)
    test_start = _utc_naive_from_epoch(legacy_cutoff_epoch + 1)
    snapshot_end = _utc_naive_from_epoch(snapshot_end_epoch)
    max_wait_seconds = float(cfg["risk"]["max_days_to_resolution"]) * 86400.0
    eligible_holdout_records = []
    for raw in incremental_normalized:
        market = markets.get(str(raw.get("market_id") or ""))
        timestamp = raw.get("timestamp")
        if market is None or timestamp is None or market.close_time is None:
            continue
        if timestamp < test_start or timestamp > snapshot_end:
            continue
        wait_seconds = (market.close_time - timestamp).total_seconds()
        if 0.0 < wait_seconds < max_wait_seconds:
            eligible_holdout_records.append(raw)
    holdout_trade_events = build_trade_events(eligible_holdout_records)
    timeline = build_timeline(
        holdout_trade_events,
        resolution_events,
        start_ts=test_start,
        end_ts=snapshot_end,
    )

    resolution_map = {event.market_id: event.resolution for event in resolution_events}
    resolution_ts = {event.market_id: event.resolved_at for event in resolution_events}
    settlements = build_trader_market_settlements_from_records(
        (
            raw
            for raw in chain(old_normalized_trades, incremental_normalized)
            if str(raw.get("market_id") or "") not in excluded_market_ids
        ),
        resolution_map,
        resolution_ts,
    )

    manifest.update(
        {
            "resolution_time_adjustments": resolution_time_adjustments,
            "combined_trades": len(old_normalized_trades) + len(incremental_normalized),
            "analysis_excluded_truncated_markets": sorted(excluded_market_ids),
            "analysis_markets": len(markets),
            "eligible_holdout_trades": len(holdout_trade_events),
            "holdout_timeline_events": len(timeline),
            "resolved_markets": len(resolution_events),
            "trader_market_settlements": len(settlements),
        }
    )
    _save_json(manifest_path, manifest)

    embedding_raw = cfg["embeddings"]
    embedding_cfg = EmbeddingConfig(
        backend=embedding_raw["backend"],
        hashing_n_features=int(embedding_raw["hashing_n_features"]),
        st_model_name=embedding_raw["st_model_name"],
        st_device=embedding_raw["st_device"],
        st_batch_size=int(embedding_raw["st_batch_size"]),
        st_normalize_embeddings=bool(embedding_raw["st_normalize_embeddings"]),
        st_trust_remote_code=bool(embedding_raw.get("st_trust_remote_code", False)),
        st_cache_chunk_size=int(embedding_raw.get("st_cache_chunk_size", 4096)),
    )
    similarity_cfg = SimilarityConfig(True, 0.0)
    semantic_estimator = TraderSkillEstimator(
        markets,
        settlements,
        min_weighted_history=float(cfg["strategy"]["min_weighted_history"]),
        similarity_config=similarity_cfg,
        embedding_config=embedding_cfg,
        embedding_cache_dir=str(cache_dir / "embeddings"),
        similarity_mode="semantic",
        embedding_bootstrap_path=cfg["research"].get("legacy_embedding_cache"),
    )
    uniform_estimator = TraderSkillEstimator(
        markets,
        settlements,
        min_weighted_history=float(cfg["strategy"]["min_weighted_history"]),
        embedding_config=embedding_cfg,
        similarity_mode="uniform",
    )
    randomized_semantic_estimator = copy.copy(semantic_estimator)
    permutation = np.random.default_rng(20260808).permutation(len(markets))
    randomized_semantic_estimator.market_vectors = semantic_estimator.market_vectors[permutation]
    randomized_semantic_estimator._estimate_cache = {}
    base = _build_backtest_config(cfg)
    experiments = [
        ("semantic_main", semantic_estimator, base),
        (
            "semantic_legacy_entry_rule",
            semantic_estimator,
            replace(base, min_edge=-1.0),
        ),
        ("global_skill_no_semantics", uniform_estimator, base),
        ("randomized_semantic_assignment", randomized_semantic_estimator, base),
        ("favorite_price_only", semantic_estimator, replace(base, signal_mode="favorite")),
        ("volume_weighted_consensus", semantic_estimator, replace(base, signal_weighting="volume")),
        ("skill_only_consensus", semantic_estimator, replace(base, signal_weighting="skill")),
        ("equal_vote_consensus", semantic_estimator, replace(base, signal_weighting="equal")),
        ("delay_30_minutes", semantic_estimator, replace(base, delay_seconds=1800)),
        ("slippage_100_bps", semantic_estimator, replace(base, slippage_bps=100.0)),
        (
            "two_traders_concentration_cap",
            semantic_estimator,
            replace(base, min_skilled_traders=2, max_single_trader_weight=0.5),
        ),
        (
            "costs_capacity_risk_caps",
            semantic_estimator,
            replace(
                base,
                enforce_risk_caps=True,
                max_fill_participation=0.1,
                trade_fee_bps=50.0,
                slippage_bps=50.0,
            ),
        ),
        (
            "fixed_fraction_2pct",
            semantic_estimator,
            replace(
                base,
                position_sizing="fixed_fraction",
                stable_balance_fraction=0.02,
            ),
        ),
        ("delay_24_hours", semantic_estimator, replace(base, delay_seconds=86400)),
        ("rolling_24h_flow", semantic_estimator, replace(base, flow_lookback_seconds=86400)),
    ]

    summaries: list[dict[str, Any]] = []
    for name, estimator, experiment_cfg in experiments:
        print(f"Experiment: {name}", flush=True)
        result = EventDrivenBacktester(markets, timeline, estimator, experiment_cfg).run()
        summary = _summarize_result(name, result, experiment_cfg.initial_balance)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        if name == "semantic_main":
            _write_main_artifacts(result, output_dir)
        del result
        gc.collect()

    results = pd.DataFrame(summaries)
    results.to_csv(output_dir / "experiment_results.csv", index=False)
    _save_json(output_dir / "experiment_results.json", summaries)


if __name__ == "__main__":
    main()
