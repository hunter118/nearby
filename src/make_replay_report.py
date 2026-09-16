"""Assemble numerical tables and vector figures from completed frozen replays.

Uses the original plotting functions with output roots explicitly redirected;
never modifies manuscript files or the original research artifacts.
"""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

import make_execution_study_artifacts as execution
import make_paper_figures as tails
import make_semantic_risk_artifacts as risk
from run_frozen_replay import verify_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/reproduced"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    parser.add_argument("--specifications", type=Path, default=Path("config/paper_experiments.json"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    specification = json.loads(args.specifications.read_text())
    frames, checks = {}, []
    for group, definition in specification["groups"].items():
        folder = (args.input_dir / group).resolve()
        rows = []
        for name, item in definition["experiments"].items():
            path = folder / f"{name}_summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Run --all-experiments first; missing {path}")
            row = json.loads(path.read_text())
            checks.extend(dict(group=group, experiment=name, **check)
                          for check in verify_metrics(row, item["expected"]))
            row["run_directory"] = str(folder)
            rows.append(row)
        frames[group] = pd.DataFrame(rows)
        frames[group].drop(columns="run_directory").to_csv(out / f"{group}_metrics.csv", index=False)
    (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    failures = [check for check in checks if not check["passed"]]
    if failures:
        raise AssertionError(f"Numerical mismatches: {failures}")

    full = (args.input_dir / "full_window").resolve()
    complete = (args.input_dir / "complete_tape").resolve()
    # Plot adapter aliases the renamed recent single-print diagnostic only inside
    # this report's scratch directory; the long-window baseline is unchanged.
    plot_recent = out / "plot_inputs"
    plot_recent.mkdir(exist_ok=True)
    for source in complete.glob("*.csv"):
        name = source.name.replace("primary_single_print_", "baseline_")
        shutil.copyfile(source, plot_recent / name)
    recent = frames["complete_tape"].copy()
    recent["experiment"] = recent["experiment"].replace({"primary_single_print": "baseline"})
    recent["run_directory"] = str(plot_recent)
    execution.ROOT = Path.cwd()
    execution.SEMANTIC_ARTIFACT_ROOT = full
    execution.FIGURES = execution.SUPPLEMENT = out
    execution._plot_main_long_recent_equity(recent)
    execution._plot_capacity_sensitivity(recent)
    execution._write_fixed_path_cost_stress(recent)

    risk.ROOT = risk.HYPERPARAMETER_ROOT = risk.ROBUSTNESS_ROOT = full
    risk.make_appendix_equity_figure()
    for suffix in ("pdf", "csv"):
        shutil.copyfile(full / f"appendix_equity_specifications.{suffix}",
                        out / f"appendix_equity_specifications.{suffix}")

    labels = {"baseline": "No risk overlay", "consensus_loose": "Multi-wallet consensus",
              "consensus_history_1p5": "+ history breadth", "semantic_event_cap_15pct": "+ event cap",
              "tiered_position_cap_25pct": "+ general cap 25%",
              "tiered_position_cap_20pct": "+ general cap 20%",
              "tiered_position_cap_15pct": "+ general cap 15%"}
    data = frames["full_window"].set_index("experiment").loc[list(labels)]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(-100 * data.max_drawdown, 100 * data.total_return,
            color="#8b97a3", linewidth=1.2, alpha=.85, zorder=1)
    ax.scatter(-100 * data.max_drawdown, 100 * data.total_return,
               s=68, color="#2463a6", edgecolor="white", linewidth=.7, zorder=2)
    for (name, row), dy in zip(data.iterrows(), (-1, 7, -13, 5, 5, -11, 3)):
        ax.annotate(labels[name], (-100 * row.max_drawdown, 100 * row.total_return),
                    xytext=(7, dy), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Maximum drawdown magnitude (%)", ylabel="Total return (%)",
           title="Long-window risk-return trade-off")
    ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(out / "risk_return_tradeoff.pdf")
    plt.close(fig)
    shutil.copyfile(args.input_dir / "tail_diagnostic/no_overlay_tail_closed_positions.csv",
                    out / "main_closed_positions.csv")
    # Original helper copies its figure to a second directory; use a distinct one.
    tails.ARTIFACT_DIR = out
    tails.PAPER_FIGURE_DIR = out / "tail_figure"
    tails.make_trade_pnl_tail()
    print(f"Verified {len(checks)} metrics across {sum(len(f) for f in frames.values())} replays.")
    print(f"Empirical tables, fixed-path costs and six vector figures: {out}")


if __name__ == "__main__":
    main()
