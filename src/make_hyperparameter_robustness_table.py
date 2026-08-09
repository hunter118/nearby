"""Build the paper's one-at-a-time hyperparameter sensitivity table.

The center row in each family is the same primary specification.  Repeating it
keeps every three-point comparison self-contained while making clear that this
is a local sensitivity analysis, not a Cartesian hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ARTIFACT_DIR = Path("artifacts/semantic-risk-2026-08-08")
DEFAULT_REPORT_PATH = Path(
    "reports/polymarket_paper/supplement/semantic_risk_hyperparameter_robustness.csv"
)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed experiment summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the local semantic-risk hyperparameter sensitivity runs."
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    hyperparameter_dir = artifact_dir / "hyperparameter"
    report_path = Path(args.report_path)

    primary_path = artifact_dir / "tiered_position_cap_15pct_summary.json"
    grid = (
        (
            "Maximum directional expert share",
            "max_directional_trader_weight",
            0.65,
            hyperparameter_dir / "hp_max_concentration_0p65_summary.json",
            False,
        ),
        (
            "Maximum directional expert share",
            "max_directional_trader_weight",
            0.75,
            primary_path,
            True,
        ),
        (
            "Maximum directional expert share",
            "max_directional_trader_weight",
            0.85,
            hyperparameter_dir / "hp_max_concentration_0p85_summary.json",
            False,
        ),
        (
            "Mean effective related-market history",
            "min_signal_mean_expert_history_markets",
            1.0,
            hyperparameter_dir / "hp_mean_history_1p0_summary.json",
            False,
        ),
        (
            "Mean effective related-market history",
            "min_signal_mean_expert_history_markets",
            1.5,
            primary_path,
            True,
        ),
        (
            "Mean effective related-market history",
            "min_signal_mean_expert_history_markets",
            2.0,
            hyperparameter_dir / "hp_mean_history_2p0_summary.json",
            False,
        ),
        (
            "General single-position equity cap",
            "max_position_exposure_fraction",
            0.10,
            hyperparameter_dir / "hp_position_cap_10pct_summary.json",
            False,
        ),
        (
            "General single-position equity cap",
            "max_position_exposure_fraction",
            0.15,
            primary_path,
            True,
        ),
        (
            "General single-position equity cap",
            "max_position_exposure_fraction",
            0.20,
            hyperparameter_dir / "hp_position_cap_20pct_summary.json",
            False,
        ),
    )

    rows: list[dict[str, Any]] = []
    for family, parameter, value, path, is_primary in grid:
        summary = _read_summary(path)
        rows.append(
            {
                "family": family,
                "parameter": parameter,
                "value": value,
                "primary_specification": is_primary,
                "source_experiment": summary["experiment"],
                "closed_positions": int(summary["closed_positions"]),
                "loss_count": int(summary["loss_count"]),
                "win_rate_pct": 100.0 * float(summary["win_rate"]),
                "total_return_pct": 100.0 * float(summary["total_return"]),
                "max_drawdown_pct": 100.0 * float(summary["max_drawdown"]),
                "worst_trade_pnl_usdc": float(summary["worst_trade_pnl"]),
                "expected_shortfall_5pct_pnl_usdc": float(
                    summary["expected_shortfall_5pct_pnl"]
                ),
                "profit_factor": float(summary["profit_factor"]),
            }
        )

    table = pd.DataFrame(rows)
    artifact_path = hyperparameter_dir / "hyperparameter_robustness.csv"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifact_path, index=False)
    table.to_csv(report_path, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {artifact_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
