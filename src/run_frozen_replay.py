"""Replay the paper's frozen public-data snapshot without API/model downloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_resolution_events, build_timeline
from data.replay_snapshot import load_snapshot
from features.embeddings import SimilarityConfig
from run_research import _save_json, _summarize_result
from run_semantic_risk_study import (
    _execution_diagnostics, _parse_naive_utc, _write_experiment_artifacts,
)


def verify_metrics(actual, expected):
    checks = []
    for key, value in expected.items():
        measured = actual.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        passed = measured is not None and math.isclose(
            measured, value, rel_tol=1e-8, abs_tol=1e-8)
        checks.append(dict(metric=key, expected=value, actual=measured, passed=passed))
    if not checks:
        raise ValueError("No reference metrics to verify")
    return checks


def replay_group(root, group, names, mode, output):
    start, end = map(_parse_naive_utc, (group["start"], group["end"]))
    print(f"Loading frozen inputs for {start} -- {end} ({mode})", flush=True)
    markets, trades, history, vectors, metadata = load_snapshot(root, start, end)
    timeline = build_timeline(trades, build_resolution_events(markets), start, end)
    estimator = TraderSkillEstimator(
        markets, history, similarity_config=SimilarityConfig(True, 0.0),
        similarity_mode=mode, precomputed_market_vectors=vectors)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "replay_manifest.json", {
        "start": group["start"], "end": group["end"], "similarity_mode": mode,
        "eligible_trades": len(trades), "settlements": len(history),
        "snapshot_sha256": metadata["archive_sha256"],
        "verification": "reference metrics" if mode == "semantic" else "new ablation; no reference target",
    })
    summaries, verification = [], []
    for name, specification in group["experiments"].items():
        if names is not None and name not in names:
            continue
        config = BacktestConfig(**specification["config"])
        print(f"Running {name}", flush=True)
        begun = time.monotonic()
        result = EventDrivenBacktester(markets, timeline, estimator, config).run()
        summary = _summarize_result(name, result, config.initial_balance)
        summary.update(_execution_diagnostics(result))
        notional = summary.get("execution_filled_notional")
        summary["net_pnl_per_execution_filled_notional"] = (
            summary["net_realized_pnl"] / notional if notional else None)
        summary["similarity_mode"] = mode
        summaries.append(summary)
        _write_experiment_artifacts(name, result, summary, config.initial_balance, output)
        _save_json(output / f"{name}_config.json", asdict(config))
        if mode == "semantic":
            checks = verify_metrics(summary, specification["expected"])
            verification.extend(dict(experiment=name, **check) for check in checks)
            _save_json(output / "verification.json", verification)
            failures = [check for check in checks if not check["passed"]]
            if failures:
                raise AssertionError(f"Numerical reproduction failed: {name}: {failures}")
        _save_json(output / "experiment_results.json", summaries)
        print(json.dumps({k: summary[k] for k in (
            "experiment", "closed_positions", "total_return", "max_drawdown")}) +
            f"; {time.monotonic()-begun:.1f}s", flush=True)
        del result
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/frozen"))
    parser.add_argument("--specifications", type=Path,
                        default=Path("config/paper_experiments.json"))
    parser.add_argument("--group", default="all",
                        help="all, full_window, complete_tape, or tail_diagnostic")
    parser.add_argument("--experiments", help="Comma-separated IDs; defaults to three headline replays.")
    parser.add_argument("--all-experiments", action="store_true", help="Replay all appendix grids too.")
    parser.add_argument("--similarity-mode", choices=("semantic", "uniform"), default="semantic")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reproduced"))
    args = parser.parse_args()
    specification = json.loads(args.specifications.read_text())
    groups = specification["groups"]
    if args.group != "all" and args.group not in groups:
        parser.error(f"Unknown group: {args.group}")
    selected = {key: value for key, value in groups.items()
                if args.group == "all" or key == args.group}
    names = (set(args.experiments.split(",")) if args.experiments else
             None if args.all_experiments else set(specification["headline_experiments"]))
    if names is not None:
        known = {name for group in selected.values() for name in group["experiments"]}
        if names - known and args.experiments:
            parser.error(f"Unknown experiments in selected groups: {sorted(names-known)}")
    for key, group in selected.items():
        if names is not None and not names.intersection(group["experiments"]):
            continue
        replay_group(args.snapshot, group, names, args.similarity_mode, args.output_dir / key)
        gc.collect()


if __name__ == "__main__":
    main()
