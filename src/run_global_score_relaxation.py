"""Research-only paired relaxation study; never changes the frozen ICLR files.

Each pair differs only in historical similarity weights. Thresholds are fixed
before running, not chosen from the resulting returns. All runs retain the
original snapshot, horizon, position caps, sizing formula and execution rules.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_resolution_events, build_timeline
from data.replay_snapshot import load_snapshot
from features.embeddings import SimilarityConfig
from run_frozen_replay import verify_metrics
from run_research import _save_json, _summarize_result
from run_semantic_risk_study import _parse_naive_utc, _write_experiment_artifacts


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_NAME = "tiered_position_cap_15pct"
SUPPORT = dict(min_effective_directional_traders=1.1,
               max_directional_trader_weight=.90,
               min_signal_mean_expert_history_markets=1.0)
EXPERT = dict(skill_threshold=.01, min_user_volume=1.0)
CASES = {
    "original": {},
    "support_loose": SUPPORT,
    "expert_loose": EXPERT,
    "joint_loose": {**SUPPORT, **EXPERT},
    "single_wallet_stress": {**EXPERT, "min_directional_traders": 1,
                             "min_effective_directional_traders": 1.0,
                             "max_directional_trader_weight": 1.0,
                             "min_signal_mean_expert_history_markets": 0.0},
    "joint_price_60": {**SUPPORT, **EXPERT, "stable_min_price": .60,
                       "consensus_threshold": .60},
}


def compare_positions(semantic, uniform):
    """Match actual held markets and options; reject duplicate market records."""
    s = {x["market_id"]: x for x in semantic}
    u = {x["market_id"]: x for x in uniform}
    if len(s) != len(semantic) or len(u) != len(uniform):
        raise ValueError("Multiple closed positions per market need explicit aggregation")
    shared, union = set(s) & set(u), set(s) | set(u)
    same = {mid for mid in shared if s[mid]["direction"] == u[mid]["direction"]}
    opposite = shared - same
    return {
        "semantic_markets": len(s), "uniform_markets": len(u),
        "shared_markets": len(shared), "union_markets": len(union),
        "same_option_markets": len(same), "opposite_option_markets": len(opposite),
        "semantic_only_markets": len(s) - len(shared),
        "uniform_only_markets": len(u) - len(shared),
        "market_jaccard": len(shared) / len(union) if union else None,
        "option_pair_jaccard": len(same) / (len(s) + len(u) - len(same)) if union else None,
        "semantic_overlap_fraction": len(shared) / len(s) if s else None,
        "uniform_overlap_fraction": len(shared) / len(u) if u else None,
        "opposite_market_ids": sorted(opposite),
        "semantic_only_market_ids": sorted(set(s) - set(u)),
        "uniform_only_market_ids": sorted(set(u) - set(s)),
        "same_option_same_entry_time": sum(s[m]["opened_at"] == u[m]["opened_at"] for m in same),
        "same_option_same_entry_price": sum(math.isclose(s[m]["avg_entry_price"],
              u[m]["avg_entry_price"], abs_tol=1e-12, rel_tol=0) for m in same),
    }


def run_chunk(mode, names, snapshot, output, group):
    start, end = map(_parse_naive_utc, (group["start"], group["end"]))
    markets, trades, history, vectors, metadata = load_snapshot(snapshot, start, end)
    timeline = build_timeline(trades, build_resolution_events(markets), start, end)
    estimator = TraderSkillEstimator(markets, history,
        similarity_config=SimilarityConfig(True, 0.0), similarity_mode=mode,
        precomputed_market_vectors=vectors)
    base = BacktestConfig(**group["experiments"][REFERENCE_NAME]["config"])
    summaries = {}
    for name in names:
        config = replace(base, **CASES[name])
        folder = Path(output) / mode / name
        folder.mkdir(parents=True, exist_ok=False)
        _save_json(folder / "config.json", asdict(config))
        begun = time.monotonic()
        print(f"Running {mode}/{name}", flush=True)
        result = EventDrivenBacktester(markets, timeline, estimator, config).run()
        summary = _summarize_result(name, result, config.initial_balance)
        summary.update(similarity_mode=mode, snapshot_sha256=metadata["archive_sha256"])
        # Matched settled-position comparisons must not silently omit open risk.
        if summary["open_positions"]:
            raise AssertionError("Open positions require a different comparison scope")
        if name == "original":
            reference = json.loads((ROOT / "artifacts/iclr2027-reproduction" / mode /
                "full_window" / f"{REFERENCE_NAME}_summary.json").read_text())
            checks = verify_metrics(summary, {k: reference[k] for k in
                ("total_return", "max_drawdown", "closed_positions", "net_realized_pnl")})
            _save_json(folder / "baseline_verification.json", checks)
            assert all(x["passed"] for x in checks), checks
        positions = [asdict(x) for x in result["closed_positions"]]
        compare_positions(positions, positions)
        assert math.isclose(sum(x["pnl"] for x in positions) - summary["fees"],
                            summary["net_realized_pnl"], abs_tol=1e-7)
        _save_json(folder / "positions.json", positions)
        _save_json(folder / "fills.json", [asdict(x) for x in result["fills"]])
        _write_experiment_artifacts(name, result, summary, config.initial_balance, folder)
        _save_json(folder / "summary.json", summary)
        summaries[name] = summary
        print(json.dumps({"mode": mode, "case": name,
            "positions": summary["closed_positions"], "return": summary["total_return"],
            "max_drawdown": summary["max_drawdown"], "seconds": round(time.monotonic()-begun, 1)}),
            flush=True)
        del result
        gc.collect()
    return mode, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
        default=ROOT / "artifacts/iclr2027-reproduction/data")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "artifacts/global-score-research/relaxation_2026-09-16")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Never write into the manuscript, frozen inputs, or previous replay roots.
    research_root = (ROOT / "artifacts/global-score-research").resolve()
    if not output.is_relative_to(research_root) or output == research_root:
        parser.error("Use a fresh subdirectory of artifacts/global-score-research")
    output.mkdir(parents=True, exist_ok=False)
    group = json.loads((ROOT / "config/paper_experiments.json").read_text())["groups"]["full_window"]
    code_paths = [Path(__file__), ROOT / "src/backtest/engine.py",
                  ROOT / "src/alpha/trader_skill.py", ROOT / "src/alpha/signal.py",
                  ROOT / "config/paper_experiments.json"]
    _save_json(output / "study_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "start": group["start"], "end": group["end"],
        "cases_fixed_before_run": CASES,
        "reference_config": group["experiments"][REFERENCE_NAME]["config"],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in code_paths},
        "scope": "Exploratory research only; excluded from frozen ICLR version",
    })
    names = list(CASES)
    chunks = [names[:3], names[3:]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_chunk, mode, chunk, args.snapshot, output, group)
                   for mode in ("semantic", "uniform") for chunk in chunks]
        for future in as_completed(futures):
            future.result()
    comparisons = []
    for name in CASES:
        items, summaries = {}, {}
        for mode in ("semantic", "uniform"):
            folder = output / mode / name
            items[mode] = json.loads((folder / "positions.json").read_text())
            summaries[mode] = json.loads((folder / "summary.json").read_text())
        row = {"case": name, **compare_positions(items["semantic"], items["uniform"])}
        row["semantic"] = summaries["semantic"]
        row["uniform"] = summaries["uniform"]
        comparisons.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in
            ("semantic", "uniform", "semantic_only_market_ids", "uniform_only_market_ids")}), flush=True)
    _save_json(output / "comparisons.json", comparisons)
    print(f"Completed research-only paired grid: {output}", flush=True)


if __name__ == "__main__":
    main()
