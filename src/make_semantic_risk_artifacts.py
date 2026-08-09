from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("artifacts/semantic-risk-2026-08-08")
PAPER = Path("reports/polymarket_paper")
HYPERPARAMETER_ROOT = ROOT / "hyperparameter"
ROBUSTNESS_ROOT = ROOT / "robustness"


LONG_NAMES = [
    "baseline",
    "consensus_loose",
    "consensus_history_1p5",
    "tiered_position_cap_25pct",
    "tiered_position_cap_20pct",
    "tiered_position_cap_15pct",
]


def load_summary(path: Path, window: str) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    row["window"] = window
    return row


def selected_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "window",
        "experiment",
        "fills",
        "closed_positions",
        "win_rate",
        "loss_count",
        "total_return",
        "max_drawdown",
        "worst_trade_pnl",
        "expected_shortfall_5pct_pnl",
        "profit_factor",
        "fees",
    ]
    return frame[columns]


def load_equity_curve(path: Path, specification: str, panel: str) -> pd.DataFrame:
    curve = pd.read_csv(path)
    curve["ts"] = pd.to_datetime(curve["ts"], format="mixed")
    curve["specification"] = specification
    curve["panel"] = panel
    return curve


def make_appendix_equity_figure() -> pd.DataFrame:
    """Plot the time path of equity across the main diagnostic specifications."""

    panels = [
        (
            "Risk-control sequence",
            [
                (ROOT / "baseline_equity_daily.csv", "No overlay", "#b13c3c", "-", 1.25),
                (ROOT / "consensus_loose_equity_daily.csv", "+ consensus", "#e07a5f", "-", 1.20),
                (ROOT / "consensus_history_1p5_equity_daily.csv", "+ history breadth", "#d9a441", "-", 1.20),
                (ROOT / "semantic_event_cap_15pct_equity_daily.csv", "+ event cap", "#4c956c", "-", 1.25),
                (ROOT / "tiered_position_cap_15pct_equity_daily.csv", "Primary 15%", "#1f4e79", "-", 1.85),
            ],
        ),
        (
            "General position cap",
            [
                (HYPERPARAMETER_ROOT / "hp_position_cap_10pct_equity_daily.csv", "10% cap", "#2a9d8f", "-", 1.25),
                (ROOT / "tiered_position_cap_15pct_equity_daily.csv", "15% primary", "#1f4e79", "-", 1.85),
                (HYPERPARAMETER_ROOT / "hp_position_cap_20pct_equity_daily.csv", "20% cap", "#e9a93f", "-", 1.25),
                (ROOT / "tiered_position_cap_25pct_equity_daily.csv", "25% cap", "#d1495b", "-", 1.25),
            ],
        ),
        (
            "Expert-breadth and concentration checks",
            [
                (HYPERPARAMETER_ROOT / "hp_max_concentration_0p65_equity_daily.csv", "Wallet share 65%", "#2a9d8f", "-", 1.20),
                (ROOT / "tiered_position_cap_15pct_equity_daily.csv", "Primary: 75%, history 1.5", "#1f4e79", "-", 1.85),
                (HYPERPARAMETER_ROOT / "hp_max_concentration_0p85_equity_daily.csv", "Wallet share 85%", "#d1495b", "-", 1.20),
                (HYPERPARAMETER_ROOT / "hp_mean_history_1p0_equity_daily.csv", "History 1.0", "#8c6bb1", "--", 1.20),
                (HYPERPARAMETER_ROOT / "hp_mean_history_2p0_equity_daily.csv", "History 2.0", "#e9a93f", "--", 1.20),
            ],
        ),
        (
            "Execution stresses",
            [
                (ROOT / "tiered_position_cap_15pct_equity_daily.csv", "Primary", "#1f4e79", "-", 1.85),
                (ROBUSTNESS_ROOT / "tiered_15pct_cost_stress_equity_daily.csv", "50 bp fee + 50 bp slippage", "#2a9d8f", "--", 1.35),
                (ROBUSTNESS_ROOT / "tiered_15pct_delay_30m_equity_daily.csv", "30-minute delay", "#d1495b", "--", 1.35),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex=True)
    exported_curves: list[pd.DataFrame] = []
    panel_letters = ["a", "b", "c", "d"]
    for axis, letter, (title, specifications) in zip(
        axes.flat, panel_letters, panels, strict=True
    ):
        for path, label, color, linestyle, linewidth in specifications:
            curve = load_equity_curve(path, label, letter)
            exported_curves.append(curve)
            axis.plot(
                curve["ts"],
                curve["total_equity"] / 1_000.0,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.96,
            )
        axis.axhline(10.0, color="#666666", linewidth=0.7, linestyle=":")
        axis.set_title(f"({letter}) {title}", loc="left", fontsize=12)
        axis.grid(alpha=0.20)
        axis.legend(fontsize=8.8, frameon=False, loc="upper left")
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(axis="both", labelsize=9.5)
    axes[0, 0].set_ylabel("Portfolio equity (thousand USDC)", fontsize=10.5)
    axes[1, 0].set_ylabel("Portfolio equity (thousand USDC)", fontsize=10.5)
    axes[1, 0].set_xlabel("Date", fontsize=10.5)
    axes[1, 1].set_xlabel("Date", fontsize=10.5)
    fig.suptitle(
        "Equity paths across risk, parameter, and execution specifications",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ROOT / "appendix_equity_specifications.png", dpi=220)
    plt.close(fig)

    exported = pd.concat(exported_curves, ignore_index=True)
    exported.to_csv(ROOT / "appendix_equity_specifications.csv", index=False)
    return exported


def main() -> None:
    rows = [load_summary(ROOT / f"{name}_summary.json", "long_development") for name in LONG_NAMES]
    rows.extend(
        pd.read_csv(ROOT / "holdout" / "experiment_results.csv")
        .assign(window="complete_incremental")
        .to_dict("records")
    )
    rows.extend(
        pd.read_csv(ROOT / "robustness" / "experiment_results.csv")
        .assign(window="long_robustness")
        .to_dict("records")
    )
    results = selected_columns(pd.DataFrame(rows))
    results.to_csv(ROOT / "consolidated_results.csv", index=False)
    (ROOT / "consolidated_results.json").write_text(
        json.dumps(results.to_dict("records"), indent=2),
        encoding="utf-8",
    )

    label_map = {
        "baseline": "No risk overlay",
        "consensus_loose": "Independent consensus",
        "consensus_history_1p5": "+ history breadth",
        "tiered_position_cap_25pct": "Tiered cap 25%",
        "tiered_position_cap_20pct": "Tiered cap 20%",
        "tiered_position_cap_15pct": "Tiered cap 15%",
    }
    long = results[results.window == "long_development"].copy()
    long["label"] = long.experiment.map(label_map)
    long["return_pct"] = 100.0 * long.total_return
    long["drawdown_pct"] = -100.0 * long.max_drawdown

    plt.figure(figsize=(8.4, 5.2))
    plt.scatter(long.drawdown_pct, long.return_pct, s=60, color="#2463a6")
    for row in long.itertuples():
        plt.annotate(row.label, (row.drawdown_pct, row.return_pct), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Maximum drawdown magnitude (%)")
    plt.ylabel("Total return (%)")
    plt.title("Long-window risk-return trade-off")
    plt.grid(alpha=0.22)
    plt.tight_layout()
    plt.savefig(ROOT / "risk_return_tradeoff.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for name, label, color in [
        ("baseline", "No risk overlay", "#b13c3c"),
        ("tiered_position_cap_15pct", "Primary 15%", "#2463a6"),
    ]:
        curve = pd.read_csv(ROOT / f"{name}_equity_daily.csv")
        curve["ts"] = pd.to_datetime(curve.ts, format="mixed")
        axes[0].plot(curve.ts, curve.total_equity, label=label, linewidth=1.7, color=color)
    axes[0].set_title("Full retrospective sample")
    axes[0].set_ylabel("Portfolio equity (USDC)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    for name, label, color in [
        ("baseline", "No risk overlay", "#b13c3c"),
        ("tiered_position_cap_15pct", "Primary 15%", "#2463a6"),
    ]:
        curve = pd.read_csv(ROOT / "holdout" / f"{name}_equity_daily.csv")
        curve["ts"] = pd.to_datetime(curve.ts, format="mixed")
        axes[1].plot(curve.ts, curve.total_equity, label=label, linewidth=1.7, color=color)
    axes[1].set_title("Recent complete-data segment")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "risk_control_equity.png", dpi=200)
    plt.close(fig)

    make_appendix_equity_figure()

    PAPER.joinpath("figures").mkdir(parents=True, exist_ok=True)
    PAPER.joinpath("supplement").mkdir(parents=True, exist_ok=True)
    for name in [
        "risk_return_tradeoff.png",
        "risk_control_equity.png",
        "appendix_equity_specifications.png",
    ]:
        shutil.copy2(ROOT / name, PAPER / "figures" / name)
    for name in ["consolidated_results.csv", "consolidated_results.json", "study_manifest.json"]:
        shutil.copy2(ROOT / name, PAPER / "supplement" / f"semantic_risk_{name}")
    shutil.copy2(
        ROOT / "holdout" / "experiment_results.csv",
        PAPER / "supplement" / "semantic_risk_complete_incremental.csv",
    )
    shutil.copy2(
        ROOT / "robustness" / "experiment_results.csv",
        PAPER / "supplement" / "semantic_risk_robustness.csv",
    )
    shutil.copy2(
        ROOT / "appendix_equity_specifications.csv",
        PAPER / "supplement" / "semantic_risk_appendix_equity_specifications.csv",
    )


if __name__ == "__main__":
    main()
