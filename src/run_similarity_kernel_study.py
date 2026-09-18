"""Research-only locality kernels; preserve the frozen strategy and manuscript.

The grid is specified before this run, but is exploratory on an already examined
historical window, not a prospective or preregistered performance evaluation.
Only historical similarity weights change. Portfolio clustering keeps raw cosine.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_resolution_events, build_timeline
from data.replay_snapshot import load_snapshot
from features.embeddings import SimilarityConfig
from run_frozen_replay import verify_metrics
from run_global_score_relaxation import compare_positions
from run_research import _save_json, _summarize_result
from run_semantic_risk_study import _parse_naive_utc


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/global-score-research/relaxation_2026-09-16"


@dataclass(frozen=True)
class Kernel:
    threshold: float = 0.0
    power: float = 1.0
    normalize: bool = True

    def __post_init__(self):
        if not 0 <= self.threshold < 1 or not math.isfinite(self.power) or self.power <= 0:
            raise ValueError("Require 0 <= threshold < 1 and finite power > 0")

    def apply(self, cosine):
        cosine = np.asarray(cosine)
        if not np.isfinite(cosine).all():
            raise ValueError("Non-finite similarity")
        if self.threshold == 0 and self.power == 1:
            # Preserve baseline float32 rounding exactly for replay verification.
            return np.maximum(cosine, 0)
        weight = np.maximum(np.clip(cosine, -1, 1) - self.threshold, 0)
        if self.normalize:
            weight = weight / (1 - self.threshold)
        return weight ** self.power


# None is the no-embedding, globally weighted reference.
KERNELS = {
    "global": None,
    "cosine": Kernel(),
    "cosine_squared": Kernel(power=2),
    "shift_04": Kernel(threshold=.4),
    "shift_05": Kernel(threshold=.5),
    "shift_06": Kernel(threshold=.6),
    "shift_05_squared": Kernel(threshold=.5, power=2),
    "subtract_05_unscaled": Kernel(threshold=.5, normalize=False),
}


class KernelSkillEstimator(TraderSkillEstimator):
    """Change history weights without editing the production estimator."""

    def __init__(self, *args, kernel: Kernel, **kwargs):
        if kwargs.get("similarity_mode", "semantic") != "semantic":
            raise ValueError("Kernels apply only to semantic estimation")
        self.kernel = kernel
        super().__init__(*args, **kwargs)

    def _pairwise_similarity(self, query_vec, history_vecs):
        return self.kernel.apply(super()._pairwise_similarity(query_vec, history_vecs))

    def market_similarity(self, left_market_id, right_market_id):
        # Kernel locality is not a change to the portfolio risk-cluster metric.
        if left_market_id == right_market_id:
            return 1.0
        left = self.market_index.get(left_market_id)
        right = self.market_index.get(right_market_id)
        if left is None or right is None:
            return 0.0
        return float(TraderSkillEstimator._pairwise_similarity(
            self, self.market_vectors[left], self.market_vectors[[right]])[0])


def position_group(rows):
    return dict(count=len(rows), wins=sum(p["pnl"] > 0 for p in rows),
                losses=sum(p["pnl"] < 0 for p in rows),
                realized_pnl=sum(p["pnl"] for p in rows),
                notional=sum(p["notional"] for p in rows))


def run_chunk(names, snapshot, output, spec):
    start, end = map(_parse_naive_utc, (spec["start"], spec["end"]))
    markets, trades, history, vectors, metadata = load_snapshot(snapshot, start, end)
    timeline = build_timeline(trades, build_resolution_events(markets), start, end)
    config = BacktestConfig(**spec["experiments"]["tiered_position_cap_15pct"]["config"])
    for name in names:
        folder = Path(output) / name
        folder.mkdir(parents=True, exist_ok=False)
        kernel = KERNELS[name]
        options = dict(similarity_config=SimilarityConfig(True, 0.0),
                       precomputed_market_vectors=vectors)
        estimator = (TraderSkillEstimator(markets, history, similarity_mode="uniform", **options)
                     if kernel is None else KernelSkillEstimator(markets, history, kernel=kernel, **options))
        _save_json(folder / "config.json", asdict(config))
        begun = time.monotonic()
        print(f"Running similarity kernel: {name}", flush=True)
        result = EventDrivenBacktester(markets, timeline, estimator, config).run()
        summary = _summarize_result(name, result, config.initial_balance)
        summary.update(kernel=asdict(kernel) if kernel is not None else None,
                       snapshot_sha256=metadata["archive_sha256"])
        assert not summary["open_positions"], "Comparison requires all positions settled"
        positions = [asdict(p) for p in result["closed_positions"]]
        fills = [asdict(p) for p in result["fills"]]
        assert math.isclose(sum(p["pnl"] for p in positions) - summary["fees"],
                            summary["net_realized_pnl"], abs_tol=1e-7)
        if name in ("global", "cosine"):
            ref = REFERENCE / ("uniform" if name == "global" else "semantic") / "original"
            checks = verify_metrics(summary, json.loads((ref / "summary.json").read_text()))
            assert all(x["passed"] for x in checks), checks
            for key, rows in (("positions", positions), ("fills", fills)):
                assert json.loads(json.dumps(rows, default=str)) == json.loads((ref / f"{key}.json").read_text())
            _save_json(folder / "baseline_verification.json", checks)
        _save_json(folder / "summary.json", summary)
        _save_json(folder / "positions.json", positions)
        _save_json(folder / "fills.json", fills)
        _save_json(folder / "equity_curve.json", result["equity_curve"])
        print(json.dumps(dict(kernel=name, positions=summary["closed_positions"],
            total_return=summary["total_return"], max_drawdown=summary["max_drawdown"],
            seconds=round(time.monotonic() - begun, 1))), flush=True)
        del result, estimator
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/iclr2027-reproduction/data")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/global-score-research/kernels_2026-09-16")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    research = (ROOT / "artifacts/global-score-research").resolve()
    if output == research or not output.is_relative_to(research):
        parser.error("Use a fresh research artifact subdirectory")
    if not 1 <= args.workers <= 4:
        parser.error("Use 1 to 4 workers")
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((ROOT / "config/paper_experiments.json").read_text())["groups"]["full_window"]
    paths = [Path(__file__), ROOT / "src/alpha/trader_skill.py", ROOT / "src/alpha/signal.py",
             ROOT / "src/backtest/engine.py", ROOT / "config/paper_experiments.json"]
    _save_json(output / "manifest.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        scope="Exploratory historical kernel grid, outside frozen ICLR version",
        start=spec["start"], end=spec["end"],
        kernels={key: asdict(k) if k else None for key, k in KERNELS.items()},
        reference_config=spec["experiments"]["tiered_position_cap_15pct"]["config"],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        interpretation=["All non-kernel parameters held fixed; no volume/capacity validation added.",
                        "Kernel scale changes weighted-history notional and its eligibility gate.",
                        "Normalized and unscaled shift_05 isolate this constant-scale effect.",
                        "mean_expert_similarity now means mean transformed weight, not raw cosine.",
                        "Selection/PnL comparisons include portfolio path and sizing effects."]))
    names = list(KERNELS)
    chunks = [names[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_chunk, chunk, args.snapshot, output, spec) for chunk in chunks]
        for future in as_completed(futures):
            future.result()
    positions = {name: json.loads((output / name / "positions.json").read_text()) for name in names}
    summaries = {name: json.loads((output / name / "summary.json").read_text()) for name in names}
    comparisons = []
    for name in names:
        for reference in ("global", "cosine"):
            own = positions[name]
            ref = positions[reference]
            ref_ids = {p["market_id"] for p in ref}
            own_ids = {p["market_id"] for p in own}
            comparisons.append(dict(kernel=name, reference=reference,
                **compare_positions(own, ref),
                own_only=position_group([p for p in own if p["market_id"] not in ref_ids]),
                reference_only=position_group([p for p in ref if p["market_id"] not in own_ids])))
    _save_json(output / "comparisons.json", comparisons)
    _save_json(output / "summaries.json", summaries)
    print(f"Finished exploratory kernel grid: {output}", flush=True)


if __name__ == "__main__":
    main()
