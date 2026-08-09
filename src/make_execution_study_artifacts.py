"""Consolidate the recent complete-tape execution study and make paper figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "execution-2026-08-09"
SEMANTIC_ARTIFACT_ROOT = ROOT / "artifacts" / "semantic-risk-2026-08-08"
PAPER_ROOT = ROOT / "reports" / "polymarket_paper"
SUPPLEMENT = PAPER_ROOT / "supplement"
FIGURES = PAPER_ROOT / "figures"

RUN_DIRS = [
    ARTIFACT_ROOT / "recent-pilot",
    ARTIFACT_ROOT / "recent-reserved-grid",
    ARTIFACT_ROOT / "recent-reserved-extended-ttl",
    ARTIFACT_ROOT / "recent-reserved-capital",
    ARTIFACT_ROOT / "recent-reserved-cost",
]

PRIMARY = "execution_partial_target_token_25pct_24h"


def _read_results() -> pd.DataFrame:
    frames = []
    for order, directory in enumerate(RUN_DIRS):
        path = directory / "experiment_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["run_directory"] = str(directory.relative_to(ROOT))
        frame["run_order"] = order
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No execution experiment result files were found.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = (
        combined.sort_values("run_order")
        .drop_duplicates("experiment", keep="last")
        .sort_values("experiment")
        .reset_index(drop=True)
    )
    if "net_pnl_per_execution_filled_notional" not in combined:
        combined["net_pnl_per_execution_filled_notional"] = pd.NA
    missing_efficiency = combined["net_pnl_per_execution_filled_notional"].isna()
    combined.loc[missing_efficiency, "net_pnl_per_execution_filled_notional"] = (
        combined.loc[missing_efficiency, "net_realized_pnl"]
        / combined.loc[missing_efficiency, "execution_filled_notional"]
    )
    return combined.drop(columns="run_order")


def _row(results: pd.DataFrame, experiment: str) -> pd.Series:
    rows = results.loc[results["experiment"] == experiment]
    if len(rows) != 1:
        raise KeyError(f"Expected one result for {experiment}, found {len(rows)}")
    return rows.iloc[0]


def _equity_path(results: pd.DataFrame, experiment: str) -> Path:
    row = _row(results, experiment)
    return ROOT / row["run_directory"] / f"{experiment}_equity_daily.csv"


def _plot_capacity_sensitivity(results: pd.DataFrame) -> None:
    participation_specs = {
        10: "execution_partial_target_token_10pct_24h",
        25: "execution_partial_target_token_25pct_24h",
        50: "execution_partial_target_token_50pct_24h",
    }
    ttl_specs = {
        0.5: "execution_partial_target_token_25pct_30m",
        2: "execution_partial_target_token_25pct_2h",
        6: "execution_partial_target_token_25pct_6h",
        24: "execution_partial_target_token_25pct_24h",
    }
    capital_specs = {
        10: "execution_partial_target_token_25pct_24h_capital_10000",
        25: "execution_partial_target_token_25pct_24h_capital_25000",
        50: "execution_partial_target_token_25pct_24h_capital_50000",
        100: "execution_partial_target_token_25pct_24h_capital_100000",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))
    for ax, specs, xlabel in [
        (axes[0], participation_specs, "Participation ceiling per print (%)"),
        (axes[1], ttl_specs, "Parent-order lifetime (hours)"),
        (axes[2], capital_specs, "Initial capital (thousand USDC)"),
    ]:
        x = list(specs)
        rows = [_row(results, name) for name in specs.values()]
        returns = [100.0 * float(row["total_return"]) for row in rows]
        fill_ratios = [
            100.0 * float(row["execution_requested_fill_ratio"]) for row in rows
        ]
        ax.plot(x, returns, marker="o", linewidth=2.0, label="Portfolio return")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Portfolio return (%)")
        ax.grid(alpha=0.22)
        twin = ax.twinx()
        twin.plot(
            x,
            fill_ratios,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color="#b04a35",
            label="Requested-notional fill ratio",
        )
        twin.set_ylabel("Requested-notional fill ratio (%)")
        twin.set_ylim(0, 100)
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="lower right", fontsize=8)
    axes[0].set_title("Participation-rate sensitivity")
    axes[1].set_title("Order-lifetime sensitivity")
    axes[2].set_title("Capacity scaling")
    fig.tight_layout()
    fig.savefig(FIGURES / "execution_capacity_sensitivity.png", dpi=220)
    plt.close(fig)


def _plot_equity_comparison(results: pd.DataFrame) -> None:
    specifications = [
        (
            "Single-print fill (diagnostic upper bound)",
            "baseline",
            {"color": "#777777", "linestyle": "--", "linewidth": 1.5},
        ),
        (
            "25% POV, 6h",
            "execution_partial_target_token_25pct_6h",
            {"color": "#d9902f", "linestyle": "-", "linewidth": 1.5},
        ),
        (
            "10% POV, 24h",
            "execution_partial_target_token_10pct_24h",
            {"color": "#4c956c", "linestyle": "-", "linewidth": 1.5},
        ),
        (
            "25% POV, 24h (primary)",
            PRIMARY,
            {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.3},
        ),
        (
            "25% POV, 24h, price ≤ 0.998",
            "execution_partial_target_token_25pct_24h_entry_0998",
            {"color": "#8c6bb1", "linestyle": ":", "linewidth": 1.8},
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    exported_curves = []
    for label, experiment, style in specifications:
        curve = pd.read_csv(_equity_path(results, experiment))
        curve["ts"] = pd.to_datetime(curve["ts"], utc=True)
        curve["specification"] = label
        curve["experiment"] = experiment
        exported_curves.append(curve)
        ax.plot(curve["ts"], curve["total_equity"], label=label, **style)
    ax.axhline(10_000.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio equity (USDC)")
    ax.set_title("Recent complete-tape equity under capacity-aware execution")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "execution_equity_comparison.png", dpi=220)
    plt.close(fig)
    pd.concat(exported_curves, ignore_index=True).to_csv(
        SUPPLEMENT / "execution_equity_comparison.csv", index=False
    )


def _plot_main_long_recent_equity(results: pd.DataFrame) -> None:
    """Export distinct long-window and recent-execution main figures."""

    exported_curves: list[pd.DataFrame] = []
    long_fig, long_ax = plt.subplots(figsize=(9.2, 5.0))

    long_specs = [
        (
            "No consensus/exposure overlay",
            "baseline",
            SEMANTIC_ARTIFACT_ROOT / "baseline_equity_daily.csv",
            {"color": "#b13c3c", "linestyle": "-", "linewidth": 1.5},
        ),
        (
            "Primary signal-and-risk replay",
            "tiered_position_cap_15pct",
            SEMANTIC_ARTIFACT_ROOT / "tiered_position_cap_15pct_equity_daily.csv",
            {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.2},
        ),
    ]
    for label, experiment, path, style in long_specs:
        curve = pd.read_csv(path)
        curve["ts"] = pd.to_datetime(curve["ts"], utc=True)
        curve["panel"] = "long"
        curve["specification"] = label
        curve["experiment"] = experiment
        curve["execution_scope"] = "quantity-unconstrained signal-and-risk replay"
        exported_curves.append(curve)
        long_ax.plot(
            curve["ts"],
            curve["total_equity"] / 1_000.0,
            label=label,
            **style,
        )

    long_ax.set_title("Long-window signal and risk replay", loc="left")
    long_ax.set_ylabel("Portfolio equity (thousand USDC)")
    long_ax.set_xlabel("Date")
    long_ax.grid(alpha=0.20)
    long_ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    long_ax.tick_params(axis="x", rotation=20, labelsize=8)
    long_fig.tight_layout()
    long_fig.savefig(FIGURES / "main_long_equity.png", dpi=220)
    plt.close(long_fig)

    recent_specs = [
        (
            "Single-print diagnostic",
            "baseline",
            {"color": "#777777", "linestyle": "--", "linewidth": 1.4},
        ),
        (
            "10% POV, 24h",
            "execution_partial_target_token_10pct_24h",
            {"color": "#4c956c", "linestyle": "-", "linewidth": 1.5},
        ),
        (
            "25% POV, 24h (primary)",
            PRIMARY,
            {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.2},
        ),
        (
            "25% POV, 24h, price ≤ 0.998",
            "execution_partial_target_token_25pct_24h_entry_0998",
            {"color": "#8c6bb1", "linestyle": ":", "linewidth": 1.8},
        ),
    ]
    recent_fig, recent_ax = plt.subplots(figsize=(9.2, 5.0))
    for label, experiment, style in recent_specs:
        curve = pd.read_csv(_equity_path(results, experiment))
        curve["ts"] = pd.to_datetime(curve["ts"], utc=True)
        curve["panel"] = "recent"
        curve["specification"] = label
        curve["experiment"] = experiment
        curve["execution_scope"] = "recent complete-tape execution"
        exported_curves.append(curve)
        recent_ax.plot(
            curve["ts"],
            curve["total_equity"] / 1_000.0,
            label=label,
            **style,
        )

    recent_ax.set_title("Recent complete-tape execution", loc="left")
    recent_ax.set_ylabel("Portfolio equity (thousand USDC)")
    recent_ax.set_xlabel("Date")
    recent_ax.grid(alpha=0.20)
    recent_ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    recent_ax.tick_params(axis="x", rotation=20, labelsize=8)
    recent_fig.tight_layout()
    recent_fig.savefig(FIGURES / "main_recent_execution_equity.png", dpi=220)
    plt.close(recent_fig)

    pd.concat(exported_curves, ignore_index=True).to_csv(
        SUPPLEMENT / "main_long_recent_equity.csv", index=False
    )


def _write_fixed_path_cost_stress(results: pd.DataFrame) -> None:
    """Reprice frozen primary fills so costs cannot change capital allocation."""
    source_dir = ROOT / _row(results, PRIMARY)["run_directory"]
    fills = pd.read_csv(source_dir / f"{PRIMARY}_fills.csv")
    closed = pd.read_csv(source_dir / f"{PRIMARY}_closed_positions.csv")
    won = dict(zip(closed["market_id"], closed["payout"] > 0.0))
    rows = []
    for total_slippage_bps, fee_bps in [(10.0, 0.0), (30.0, 20.0), (50.0, 50.0)]:
        extra_slippage = max(0.0, total_slippage_bps - 10.0) / 10000.0
        gross_pnl = 0.0
        total_fees = 0.0
        for fill in fills.itertuples(index=False):
            notional = float(fill.notional)
            stressed_price = min(1.0, float(fill.fill_price) * (1.0 + extra_slippage))
            payout = notional / stressed_price if won.get(fill.market_id, False) else 0.0
            gross_pnl += payout - notional
            total_fees += notional * fee_bps / 10000.0
        net_pnl = gross_pnl - total_fees
        rows.append(
            {
                "total_slippage_bps": total_slippage_bps,
                "fee_bps": fee_bps,
                "frozen_child_fills": len(fills),
                "gross_pnl": gross_pnl,
                "fees": total_fees,
                "net_pnl": net_pnl,
                "portfolio_return": net_pnl / 10_000.0,
            }
        )
    pd.DataFrame(rows).to_csv(
        SUPPLEMENT / "execution_primary_fixed_path_cost_stress.csv", index=False
    )


def main() -> None:
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = _read_results()
    results.to_csv(SUPPLEMENT / "execution_recent_complete_tape_results.csv", index=False)

    capacity_config_path = (
        ARTIFACT_ROOT / "recent-reserved-capital" / "experiment_configs.json"
    )
    if capacity_config_path.exists():
        capacity_configs = json.loads(capacity_config_path.read_text(encoding="utf-8"))
        primary_config = capacity_configs[
            "execution_partial_target_token_25pct_24h_capital_10000"
        ]
        (SUPPLEMENT / "execution_primary_config.json").write_text(
            json.dumps(primary_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    primary_source = _row(results, PRIMARY)["run_directory"]
    primary_dir = ROOT / primary_source
    for suffix in [
        "fills",
        "closed_positions",
        "equity_daily",
        "annual_results",
        "quarterly_results",
    ]:
        source = primary_dir / f"{PRIMARY}_{suffix}.csv"
        if source.exists():
            pd.read_csv(source).to_csv(
                SUPPLEMENT / f"execution_primary_{suffix}.csv", index=False
            )
    monthly = ROOT / primary_source / f"{PRIMARY}_monthly_results.csv"
    if monthly.exists():
        pd.read_csv(monthly).to_csv(
            SUPPLEMENT / "execution_primary_monthly_results.csv", index=False
        )

    _plot_capacity_sensitivity(results)
    _plot_equity_comparison(results)
    _plot_main_long_recent_equity(results)
    _write_fixed_path_cost_stress(results)


if __name__ == "__main__":
    main()
