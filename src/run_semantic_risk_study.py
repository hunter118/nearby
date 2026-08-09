from __future__ import annotations

import argparse
import gc
import importlib
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from run_research import _build_backtest_config, _load_pickle, _save_json, _summarize_result


DEFAULT_START = datetime(2023, 3, 16)
OUTPUT_DIR = Path("artifacts/semantic-risk-2026-08-08")


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}. This study is offline-only; run the main "
            "reproduction/fetch step first to populate the cache."
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_naive_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _normalized_trade_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Cross-cache execution key for the short legacy/incremental overlap.

    The legacy activity cache normalized its transaction hash to the literal
    string ``"None"``, while the newer Data API cache retained the hash.  The
    shared execution fields are therefore the strongest available cross-cache
    identity.  This key is used only inside the roughly 24-minute overlap.
    """
    return (
        str(row.get("market_id") or ""),
        str(row.get("trader_id") or ""),
        str(row.get("side") or ""),
        float(row.get("price_yes") or 0.0),
        float(row.get("size") or 0.0),
        row.get("timestamp"),
    )


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _period_label(timestamp: datetime, frequency: str) -> str:
    if frequency == "year":
        return str(timestamp.year)
    if frequency == "month":
        return f"{timestamp.year}-{timestamp.month:02d}"
    return f"{timestamp.year}Q{((timestamp.month - 1) // 3) + 1}"


def _period_results(
    result: dict[str, Any],
    initial_balance: float,
    frequency: str,
) -> pd.DataFrame:
    curve = pd.DataFrame(result["equity_curve"])
    if curve.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "start_equity",
                "end_equity",
                "portfolio_return",
                "closed_positions",
                "wins",
                "win_rate",
                "gross_realized_pnl",
                "fees_on_fills",
                "net_realized_pnl_less_period_fees",
                "closed_notional",
                "return_on_closed_notional",
                "worst_trade_pnl",
            ]
        )

    curve["ts"] = pd.to_datetime(curve["ts"], format="mixed")
    curve["period"] = curve["ts"].map(lambda ts: _period_label(ts.to_pydatetime(), frequency))
    endpoints = curve.groupby("period", sort=False)["total_equity"].last()

    closed_by_period: dict[str, list[Any]] = {}
    for position in result["closed_positions"]:
        closed_by_period.setdefault(_period_label(position.resolved_at, frequency), []).append(position)
    fees_by_period: Counter[str] = Counter()
    for fill in result["fills"]:
        fees_by_period[_period_label(fill.filled_at, frequency)] += float(fill.fee)

    rows: list[dict[str, Any]] = []
    prior_equity = float(initial_balance)
    for period, end_equity_raw in endpoints.items():
        end_equity = float(end_equity_raw)
        closed = closed_by_period.get(period, [])
        gross_pnl = float(sum(position.pnl for position in closed))
        closed_notional = float(sum(position.notional for position in closed))
        wins = sum(position.pnl > 0 for position in closed)
        fees = float(fees_by_period[period])
        rows.append(
            {
                "period": period,
                "start_equity": prior_equity,
                "end_equity": end_equity,
                "portfolio_return": end_equity / prior_equity - 1.0 if prior_equity else None,
                "closed_positions": len(closed),
                "wins": wins,
                "win_rate": wins / len(closed) if closed else None,
                "gross_realized_pnl": gross_pnl,
                "fees_on_fills": fees,
                "net_realized_pnl_less_period_fees": gross_pnl - fees,
                "closed_notional": closed_notional,
                "return_on_closed_notional": gross_pnl / closed_notional if closed_notional else None,
                "worst_trade_pnl": float(min((position.pnl for position in closed), default=0.0)),
            }
        )
        prior_equity = end_equity
    return pd.DataFrame(rows)


def _write_experiment_artifacts(
    name: str,
    result: dict[str, Any],
    summary: dict[str, Any],
    initial_balance: float,
    output_dir: Path,
) -> None:
    prefix = _safe_name(name)
    pd.DataFrame([asdict(row) for row in result["fills"]]).to_csv(
        output_dir / f"{prefix}_fills.csv", index=False
    )
    pd.DataFrame([asdict(row) for row in result["closed_positions"]]).to_csv(
        output_dir / f"{prefix}_closed_positions.csv", index=False
    )
    curve = pd.DataFrame(result["equity_curve"])
    if not curve.empty:
        curve["ts"] = pd.to_datetime(curve["ts"], format="mixed")
        daily = curve.set_index("ts").resample("1D").last().dropna(how="all").reset_index()
        daily.to_csv(output_dir / f"{prefix}_equity_daily.csv", index=False)
    _period_results(result, initial_balance, "year").to_csv(
        output_dir / f"{prefix}_annual_results.csv", index=False
    )
    _period_results(result, initial_balance, "quarter").to_csv(
        output_dir / f"{prefix}_quarterly_results.csv", index=False
    )
    _period_results(result, initial_balance, "month").to_csv(
        output_dir / f"{prefix}_monthly_results.csv", index=False
    )
    _save_json(output_dir / f"{prefix}_summary.json", summary)


def _execution_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    requests = {
        str(market_id): float(notional)
        for market_id, notional in result.get("execution_order_requests", {}).items()
    }
    fills = list(result["fills"])
    filled_notional: Counter[str] = Counter()
    child_fills: Counter[str] = Counter()
    last_fill_time: dict[str, datetime] = {}
    signal_time: dict[str, datetime] = {}
    participation: list[float] = []
    price_drift_bps: list[float] = []
    for fill in fills:
        filled_notional[fill.market_id] += float(fill.notional)
        child_fills[fill.market_id] += 1
        last_fill_time[fill.market_id] = max(
            last_fill_time.get(fill.market_id, fill.filled_at),
            fill.filled_at,
        )
        signal_time.setdefault(fill.market_id, fill.signal_time)
        if fill.source_trade_size > 0:
            participation.append(float(fill.quantity) / float(fill.source_trade_size))
        if fill.signal_token_price > 0:
            price_drift_bps.append(
                10000.0 * (float(fill.fill_price) / float(fill.signal_token_price) - 1.0)
            )

    tolerance = 1e-8
    orders_with_fill = sum(filled_notional[market_id] > tolerance for market_id in requests)
    fully_filled = sum(
        filled_notional[market_id] >= requested - tolerance
        for market_id, requested in requests.items()
    )
    partially_filled = sum(
        tolerance < filled_notional[market_id] < requested - tolerance
        for market_id, requested in requests.items()
    )
    completion_seconds = [
        (last_fill_time[market_id] - signal_time[market_id]).total_seconds()
        for market_id in last_fill_time
    ]

    def quantile(values: list[float], probability: float) -> float | None:
        if not values:
            return None
        return float(pd.Series(values).quantile(probability))

    requested_total = float(sum(requests.values()))
    filled_total = float(sum(filled_notional.values()))
    return {
        "execution_order_submissions": len(requests),
        "execution_orders_with_fill": orders_with_fill,
        "execution_order_fill_rate": (
            orders_with_fill / len(requests) if requests else None
        ),
        "execution_fully_filled_orders": fully_filled,
        "execution_full_completion_rate": (
            fully_filled / len(requests) if requests else None
        ),
        "execution_partially_filled_orders": partially_filled,
        "execution_unfilled_orders": len(requests) - orders_with_fill,
        "execution_requested_notional": requested_total,
        "execution_filled_notional": filled_total,
        "execution_requested_fill_ratio": (
            filled_total / requested_total if requested_total else None
        ),
        "execution_median_child_fills": quantile(
            [float(value) for value in child_fills.values()], 0.5
        ),
        "execution_median_completion_seconds": quantile(completion_seconds, 0.5),
        "execution_p90_completion_seconds": quantile(completion_seconds, 0.9),
        "execution_median_print_participation": quantile(participation, 0.5),
        "execution_p90_print_participation": quantile(participation, 0.9),
        "execution_median_price_drift_bps": quantile(price_drift_bps, 0.5),
        "execution_p90_price_drift_bps": quantile(price_drift_bps, 0.9),
        "execution_rejected_markets": len(result.get("execution_rejected_markets", [])),
        "execution_pending_orders_at_end": int(result.get("execution_pending_orders", 0)),
    }


def _coerce_presets(raw: Any, base: BacktestConfig) -> dict[str, BacktestConfig]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        try:
            items = iter(raw)
        except TypeError as exc:
            raise TypeError(
                "Risk presets must be a mapping or an iterable of (name, config) pairs."
            ) from exc
    presets: dict[str, BacktestConfig] = {}
    for name, value in items:
        if isinstance(value, BacktestConfig):
            presets[str(name)] = value
        elif isinstance(value, Mapping):
            presets[str(name)] = replace(base, **dict(value))
        else:
            raise TypeError(f"Unsupported preset {name!r}: {type(value).__name__}")
    return presets


def _risk_presets(base: BacktestConfig) -> tuple[dict[str, BacktestConfig], str]:
    """Load optional project presets, with a deliberately small offline fallback."""
    try:
        module = importlib.import_module("alpha.risk_presets")
    except ModuleNotFoundError as exc:
        if exc.name != "alpha.risk_presets":
            raise
        fallback = {
            "loss_cap_2pct": replace(
                base,
                enforce_risk_caps=True,
                apply_market_volume_cap=False,
                apply_balance_cap=False,
                apply_loss_cap=True,
                max_loss_per_trade_fraction=0.02,
            )
        }
        return fallback, "built_in_fallback"

    for factory_name in (
        "build_risk_candidate_grid",
        "build_risk_presets",
        "get_risk_presets",
        "risk_presets",
    ):
        factory = getattr(module, factory_name, None)
        if callable(factory):
            return _coerce_presets(factory(base), base), f"alpha.risk_presets.{factory_name}"
    for attribute_name in ("RISK_PRESETS", "PRESETS"):
        if hasattr(module, attribute_name):
            return _coerce_presets(getattr(module, attribute_name), base), (
                f"alpha.risk_presets.{attribute_name}"
            )
    raise AttributeError(
        "alpha.risk_presets exists but exposes none of build_risk_candidate_grid(base), "
        "build_risk_presets(base), get_risk_presets(base), risk_presets(base), "
        "RISK_PRESETS, or PRESETS."
    )


def _execution_presets(base: BacktestConfig) -> tuple[dict[str, BacktestConfig], str]:
    module = importlib.import_module("alpha.execution_presets")
    factory = getattr(module, "build_execution_candidate_grid", None)
    if not callable(factory):
        raise AttributeError(
            "alpha.execution_presets must expose "
            "build_execution_candidate_grid(base)."
        )
    return (
        _coerce_presets(factory(base), base),
        "alpha.execution_presets.build_execution_candidate_grid",
    )


def _normalize_markets(
    cohort_ids: Iterable[str],
    metadata_raw: Mapping[str, dict[str, Any]],
    old_market_records: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for market_id in cohort_ids:
        if market_id in metadata_raw:
            row = PolymarketClient.normalize_market(metadata_raw[market_id])
            old_row = old_market_records.get(market_id)
            if old_row and row.get("resolution") is None and metadata_raw[market_id].get("closed") is True:
                row["resolution"] = old_row.get("resolution")
            if old_row and not row.get("created_at"):
                row["created_at"] = old_row.get("created_at")
        else:
            row = dict(old_market_records[market_id])
        normalized[market_id] = row
    return normalized


def _embedding_config(cfg: dict[str, Any]) -> EmbeddingConfig:
    raw = cfg["embeddings"]
    return EmbeddingConfig(
        backend=raw["backend"],
        hashing_n_features=int(raw["hashing_n_features"]),
        st_model_name=raw["st_model_name"],
        st_device=raw["st_device"],
        st_batch_size=int(raw["st_batch_size"]),
        st_normalize_embeddings=bool(raw["st_normalize_embeddings"]),
        st_trust_remote_code=bool(raw.get("st_trust_remote_code", False)),
        st_cache_chunk_size=int(raw.get("st_cache_chunk_size", 4096)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline long-window semantic-risk study over the frozen Polymarket cohort. "
            "The legacy portion is incomplete and is intended as a robustness/stability window."
        )
    )
    parser.add_argument("--config", default="config/research_2026_08_08.yaml")
    parser.add_argument(
        "--start",
        default=DEFAULT_START.isoformat(),
        help="UTC-naive ISO start (default: 2023-03-16T00:00:00)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="UTC-naive ISO end; defaults to the frozen snapshot timestamp",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the original semantic baseline and write its diagnostics",
    )
    parser.add_argument(
        "--presets",
        default=None,
        help=(
            "Comma-separated presets from --preset-family to run after baseline; "
            "defaults to that family's full grid. "
            "Ignored with --baseline-only."
        ),
    )
    parser.add_argument(
        "--preset-family",
        choices=("risk", "execution"),
        default="risk",
        help="Preset collection used by --presets (default: risk).",
    )
    parser.add_argument(
        "--base-preset",
        default=None,
        help=(
            "Optional risk preset applied before the baseline and selected preset family. "
            "Execution studies use tiered_position_cap_15pct."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline when running a selected preset subset",
    )
    parser.add_argument(
        "--capacity-initial-balances",
        default=None,
        help=(
            "Optional comma-separated initial balances. Each selected specification "
            "is replayed at every balance to measure execution capacity."
        ),
    )
    args = parser.parse_args()

    config_path = _require_file(Path(args.config), "research config")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_dir = Path(cfg["research"]["cache_dir"])
    source_output_dir = Path(cfg["research"]["output_dir"])
    source_manifest_path = _require_file(source_output_dir / "data_manifest.json", "data manifest")
    source_manifest = _read_json(source_manifest_path)

    old_markets_path = _require_file(
        Path(cfg["research"]["legacy_markets_cache"]), "legacy market cache"
    )
    old_trades_path = _require_file(
        Path(cfg["research"]["legacy_trades_cache"]), "legacy trade cache"
    )
    metadata_path = _require_file(cache_dir / "gamma_markets_raw.pkl", "Gamma metadata cache")
    incremental_path = _require_file(
        cache_dir / "incremental_normalized.pkl", "normalized incremental trade cache"
    )

    requested_start = _parse_naive_utc(args.start)
    snapshot_end = _parse_naive_utc(source_manifest["snapshot_end_utc"])
    requested_end = _parse_naive_utc(args.end) if args.end else snapshot_end
    study_end = min(requested_end, snapshot_end)
    if requested_start >= study_end:
        raise ValueError(f"Study start {requested_start} must be earlier than end {study_end}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached legacy and incremental data (no network access) ...", flush=True)
    old_market_records: dict[str, dict[str, Any]] = _load_pickle(old_markets_path)
    old_trades: list[dict[str, Any]] = _load_pickle(old_trades_path)
    incremental_trades: list[dict[str, Any]] = _load_pickle(incremental_path)
    metadata_raw: dict[str, dict[str, Any]] = _load_pickle(metadata_path)

    legacy_cutoff = max(
        row["timestamp"] for row in old_trades if row.get("timestamp") is not None
    )
    incremental_start = min(
        row["timestamp"] for row in incremental_trades if row.get("timestamp") is not None
    )
    incremental_overlap_rows = sum(
        row.get("timestamp") is not None and row["timestamp"] <= legacy_cutoff
        for row in incremental_trades
    )
    incremental_overlap_keys = {
        _normalized_trade_key(row)
        for row in incremental_trades
        if row.get("timestamp") is not None and row["timestamp"] <= legacy_cutoff
    }
    legacy_overlap_rows = 0
    cross_cache_duplicate_rows_removed = 0
    for row in old_trades:
        timestamp = row.get("timestamp")
        if timestamp is None or timestamp < incremental_start:
            continue
        legacy_overlap_rows += 1
        if _normalized_trade_key(row) in incremental_overlap_keys:
            cross_cache_duplicate_rows_removed += 1

    def combined_records():
        # Prefer the newer incremental copy in the roughly 24-minute cache overlap.
        for row in old_trades:
            timestamp = row.get("timestamp")
            if (
                timestamp is not None
                and timestamp >= incremental_start
                and _normalized_trade_key(row) in incremental_overlap_keys
            ):
                continue
            yield row
        yield from incremental_trades

    legacy_index_path = cache_dir / "legacy_market_index.pkl"
    if legacy_index_path.exists():
        cohort_ids = list(_load_pickle(legacy_index_path)["cohort_ids"])
    else:
        cohort_ids = sorted({str(row["market_id"]) for row in old_trades})
    excluded_market_ids = set(
        source_manifest.get("analysis_excluded_truncated_markets")
        or source_manifest.get("incremental_truncated_markets")
        or []
    )
    normalized_market_records = _normalize_markets(cohort_ids, metadata_raw, old_market_records)

    latest_trade_by_market: dict[str, datetime] = {}
    legacy_rows_by_market: Counter[str] = Counter()
    for row in old_trades:
        legacy_rows_by_market[str(row.get("market_id") or "")] += 1
    for row in combined_records():
        market_id = str(row.get("market_id") or "")
        timestamp = row.get("timestamp")
        if market_id in excluded_market_ids or timestamp is None:
            continue
        latest_trade_by_market[market_id] = max(latest_trade_by_market.get(market_id, timestamp), timestamp)

    resolution_time_adjustments = 0
    for market_id, row in normalized_market_records.items():
        resolved_at = row.get("resolved_at")
        last_trade = latest_trade_by_market.get(market_id)
        if row.get("resolution") and resolved_at and last_trade and resolved_at < last_trade:
            row["resolved_at"] = last_trade + timedelta(seconds=1)
            resolution_time_adjustments += 1

    markets = build_markets(
        row
        for market_id, row in normalized_market_records.items()
        if market_id not in excluded_market_ids
    )
    resolution_events = build_resolution_events(markets)
    max_wait_seconds = float(cfg["risk"]["max_days_to_resolution"]) * 86400.0
    eligible_records: list[dict[str, Any]] = []
    for row in combined_records():
        timestamp = row.get("timestamp")
        market = markets.get(str(row.get("market_id") or ""))
        if timestamp is None or market is None or market.close_time is None:
            continue
        if timestamp < requested_start or timestamp > study_end:
            continue
        wait_seconds = (market.close_time - timestamp).total_seconds()
        if 0.0 < wait_seconds < max_wait_seconds:
            eligible_records.append(row)
    trade_events = build_trade_events(eligible_records)
    timeline = build_timeline(
        trade_events,
        resolution_events,
        start_ts=requested_start,
        end_ts=study_end,
    )
    if not trade_events:
        raise RuntimeError("No eligible trades were found in the requested long window.")

    resolution_map = {event.market_id: event.resolution for event in resolution_events}
    resolution_ts = {event.market_id: event.resolved_at for event in resolution_events}
    settlements = build_trader_market_settlements_from_records(
        (
            row
            for row in combined_records()
            if str(row.get("market_id") or "") not in excluded_market_ids
        ),
        resolution_map,
        resolution_ts,
    )

    warnings = [
        (
            "The pre-2026-04-27 legacy cache is capped at 1,000 rows per selected market; "
            "its market-level trade history is incomplete."
        ),
        (
            "The long window uses the frozen 2,993-market cohort selected for the original "
            "study, so markets outside that cohort are absent and the long-window result is "
            "a robustness/stability test rather than a clean out-of-sample estimate."
        ),
        (
            "Markets whose incremental download hit its row cap are excluded from trades, "
            "resolutions, and trader-settlement histories."
        ),
        (
            "The legacy and incremental caches overlap in time; exact normalized duplicates "
            "are removed with the incremental record taking precedence."
        ),
    ]
    study_manifest = {
        "offline_only": True,
        "source_config": str(config_path),
        "source_manifest": str(source_manifest_path),
        "snapshot_id": source_manifest.get("snapshot_id"),
        "snapshot_end_utc": source_manifest["snapshot_end_utc"],
        "requested_start_utc": requested_start.isoformat() + "Z",
        "requested_end_utc": requested_end.isoformat() + "Z",
        "effective_end_utc": study_end.isoformat() + "Z",
        "actual_trade_start_utc": trade_events[0].timestamp.isoformat() + "Z",
        "actual_trade_end_utc": trade_events[-1].timestamp.isoformat() + "Z",
        "legacy_trades": len(old_trades),
        "incremental_trades": len(incremental_trades),
        "legacy_cutoff_utc": legacy_cutoff.isoformat() + "Z",
        "incremental_start_utc": incremental_start.isoformat() + "Z",
        "cache_overlap_seconds": max(
            0.0, (legacy_cutoff - incremental_start).total_seconds()
        ),
        "legacy_rows_in_cache_overlap": legacy_overlap_rows,
        "incremental_rows_in_cache_overlap": incremental_overlap_rows,
        "cross_cache_duplicate_rows_removed": cross_cache_duplicate_rows_removed,
        "combined_deduplicated_trades": (
            len(old_trades) + len(incremental_trades) - cross_cache_duplicate_rows_removed
        ),
        "cohort_markets": len(cohort_ids),
        "analysis_markets": len(markets),
        "excluded_truncated_markets": sorted(excluded_market_ids),
        "legacy_markets_with_exactly_1000_cached_rows": sum(
            count == 1000 for count in legacy_rows_by_market.values()
        ),
        "eligible_trades": len(trade_events),
        "timeline_events": len(timeline),
        "resolved_markets": len(resolution_events),
        "trader_market_settlements": len(settlements),
        "resolution_time_adjustments": resolution_time_adjustments,
        "warnings": warnings,
    }
    _save_json(output_dir / "study_manifest.json", study_manifest)
    print(json.dumps(study_manifest, indent=2), flush=True)

    estimator = TraderSkillEstimator(
        markets,
        settlements,
        min_weighted_history=float(cfg["strategy"]["min_weighted_history"]),
        similarity_config=SimilarityConfig(True, 0.0),
        embedding_config=_embedding_config(cfg),
        embedding_cache_dir=str(cache_dir / "embeddings"),
        similarity_mode="semantic",
        embedding_bootstrap_path=cfg["research"].get("legacy_embedding_cache"),
    )
    base = _build_backtest_config(cfg)
    base_preset_source = None
    if args.base_preset:
        risk_base_candidates, risk_base_source = _risk_presets(base)
        if args.base_preset not in risk_base_candidates:
            raise KeyError(
                f"Unknown base preset {args.base_preset!r}; available: "
                f"{sorted(risk_base_candidates)}"
            )
        base = risk_base_candidates[args.base_preset]
        base_preset_source = risk_base_source
    experiments: dict[str, BacktestConfig] = {} if args.skip_baseline else {"baseline": base}
    preset_source = None
    if not args.baseline_only:
        if args.preset_family == "execution":
            presets, preset_source = _execution_presets(base)
        else:
            presets, preset_source = _risk_presets(base)
        if args.presets:
            requested_presets = [name.strip() for name in args.presets.split(",") if name.strip()]
            unknown_presets = [name for name in requested_presets if name not in presets]
            if unknown_presets:
                raise KeyError(
                    f"Unknown presets {unknown_presets}; available: {sorted(presets)}"
                )
            experiments.update((name, presets[name]) for name in requested_presets)
        else:
            experiments.update(presets)
    capacity_initial_balances = None
    if args.capacity_initial_balances:
        capacity_initial_balances = [
            float(value.strip())
            for value in args.capacity_initial_balances.split(",")
            if value.strip()
        ]
        if not capacity_initial_balances or any(
            value <= 0.0 for value in capacity_initial_balances
        ):
            raise ValueError("--capacity-initial-balances must contain positive values.")
        experiments = {
            f"{name}_capital_{balance:g}": replace(config, initial_balance=balance)
            for name, config in experiments.items()
            for balance in capacity_initial_balances
        }
    study_manifest["baseline_only"] = bool(args.baseline_only)
    study_manifest["base_preset"] = args.base_preset
    study_manifest["base_preset_source"] = base_preset_source
    study_manifest["preset_family"] = args.preset_family
    study_manifest["preset_source"] = preset_source
    study_manifest["capacity_initial_balances"] = capacity_initial_balances
    study_manifest["experiments"] = list(experiments)
    _save_json(output_dir / "study_manifest.json", study_manifest)
    _save_json(
        output_dir / "experiment_configs.json",
        {name: asdict(config) for name, config in experiments.items()},
    )

    summaries: list[dict[str, Any]] = []
    for name, experiment_config in experiments.items():
        print(f"Experiment: {name}", flush=True)
        result = EventDrivenBacktester(markets, timeline, estimator, experiment_config).run()
        summary = _summarize_result(name, result, experiment_config.initial_balance)
        summary.update(_execution_diagnostics(result))
        filled_notional = summary.get("execution_filled_notional")
        summary["net_pnl_per_execution_filled_notional"] = (
            float(summary["net_realized_pnl"]) / float(filled_notional)
            if filled_notional
            else None
        )
        summary.update(
            {
                "requested_start_utc": requested_start.isoformat() + "Z",
                "effective_end_utc": study_end.isoformat() + "Z",
                "actual_trade_start_utc": trade_events[0].timestamp.isoformat() + "Z",
                "actual_trade_end_utc": trade_events[-1].timestamp.isoformat() + "Z",
                "data_quality_warning": " | ".join(warnings),
            }
        )
        summaries.append(summary)
        _write_experiment_artifacts(
            name,
            result,
            summary,
            experiment_config.initial_balance,
            output_dir,
        )
        print(json.dumps(summary, indent=2), flush=True)
        del result
        gc.collect()

    pd.DataFrame(summaries).to_csv(output_dir / "experiment_results.csv", index=False)
    _save_json(output_dir / "experiment_results.json", summaries)


if __name__ == "__main__":
    main()
