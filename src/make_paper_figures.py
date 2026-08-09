from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARTIFACT_DIR = Path("artifacts/research_2026-08-08")


def make_return_comparison() -> None:
    results = pd.read_csv(ARTIFACT_DIR / "experiment_results.csv")
    selected = results[
        results["experiment"].isin(
            [
                "semantic_main",
                "global_skill_no_semantics",
                "favorite_price_only",
                "fixed_fraction_2pct",
                "costs_capacity_risk_caps",
                "rolling_24h_flow",
            ]
        )
    ].copy()
    labels = {
        "semantic_main": "Semantic\n(target sizing)",
        "global_skill_no_semantics": "Global skill\n(no semantics)",
        "favorite_price_only": "Price-only\nfavorite",
        "fixed_fraction_2pct": "Semantic\n(fixed 2%)",
        "costs_capacity_risk_caps": "Semantic\n(cost/capacity)",
        "rolling_24h_flow": "Semantic\n(24h flow)",
    }
    selected["label"] = selected["experiment"].map(labels)
    selected["return_pct"] = 100.0 * selected["total_return"]
    colors = ["#2f7d32" if value >= 0 else "#a23b3b" for value in selected["return_pct"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    bars = ax.bar(selected["label"], selected["return_pct"], color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Holdout total return (%)")
    ax.set_title("Out-of-sample performance is sensitive to sizing and constraints")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(selected["return_pct"].min() - 1.4, selected["return_pct"].max() + 1.0)
    for bar, value in zip(bars, selected["return_pct"], strict=True):
        offset = 0.22 if value >= 0 else -0.35
        vertical = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}%", ha="center", va=vertical)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "experiment_returns.png", dpi=220)
    plt.close(fig)


def make_trade_pnl_tail() -> None:
    closed = pd.read_csv(ARTIFACT_DIR / "main_closed_positions.csv").sort_values("pnl")
    closed = closed.reset_index(drop=True)
    colors = ["#a23b3b" if value < 0 else "#2f7d32" for value in closed["pnl"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.bar(closed.index + 1, closed["pnl"], width=0.85, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Closed positions, sorted by PnL")
    ax.set_ylabel("Position PnL (USDC)")
    ax.set_title("One full-loss position dominates 88 profitable positions")
    ax.grid(axis="y", alpha=0.2)
    worst = closed.iloc[0]
    ax.annotate(
        f"Largest loss: {worst['pnl']:.0f} USDC",
        xy=(1, worst["pnl"]),
        xytext=(13, worst["pnl"] * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "main_trade_pnl_tail.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_return_comparison()
    make_trade_pnl_tail()
