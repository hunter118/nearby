"""Check that every performance table and the main equity figure match sources."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports" / "polymarket_paper"
SUPPLEMENT = PAPER / "supplement"
ARTIFACTS = ROOT / "artifacts" / "semantic-risk-2026-08-08"


def main() -> None:
    checks: list[dict[str, object]] = []

    def numeric(
        section: str,
        metric: str,
        source: str,
        actual: float,
        reported: float,
        tolerance: float,
    ) -> None:
        difference = abs(float(actual) - float(reported))
        passed = difference <= tolerance
        checks.append(
            {
                "section": section,
                "metric": metric,
                "source": source,
                "actual": float(actual),
                "reported": float(reported),
                "tolerance": tolerance,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"{section} / {metric}: source={actual}, paper={reported}"
            )

    def exact(
        section: str, metric: str, source: str, actual: object, reported: object
    ) -> None:
        passed = actual == reported
        checks.append(
            {
                "section": section,
                "metric": metric,
                "source": source,
                "actual": actual,
                "reported": reported,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"{section} / {metric}: source={actual!r}, paper={reported!r}"
            )

    data_manifest_path = SUPPLEMENT / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    study_manifest_path = SUPPLEMENT / "semantic_risk_study_manifest.json"
    study_manifest = json.loads(study_manifest_path.read_text(encoding="utf-8"))
    for metric, reported in {
        "cohort_markets": 2993,
        "analysis_markets": 2989,
        "combined_trades": 8821935,
        "eligible_holdout_trades": 1690218,
        "resolved_markets": 2781,
        "trader_market_settlements": 1968848,
        "unknown_outcome_rows_dropped": 0,
        "resolution_time_adjustments": 314,
    }.items():
        exact("data snapshot", metric, str(data_manifest_path), data_manifest[metric], reported)
    exact(
        "data snapshot",
        "excluded_dense_markets",
        str(data_manifest_path),
        len(data_manifest["analysis_excluded_truncated_markets"]),
        4,
    )
    exact(
        "data snapshot",
        "eligible_full_sample_trades",
        str(study_manifest_path),
        study_manifest["eligible_trades"],
        2935511,
    )
    exact(
        "data snapshot",
        "capped_pre_period_histories",
        str(study_manifest_path),
        study_manifest["legacy_markets_with_exactly_1000_cached_rows"],
        2901,
    )

    consolidated_path = SUPPLEMENT / "semantic_risk_consolidated_results.csv"
    consolidated = pd.read_csv(consolidated_path)
    long_reported = {
        "baseline": (637, 679.52, -33.95, -18485, -1278, 2.66),
        "consensus_loose": (342, 734.00, -33.95, -8775, -910, 5.48),
        "consensus_history_1p5": (312, 715.49, -33.95, -8021, -601, 8.44),
        "tiered_position_cap_25pct": (312, 355.69, -26.08, -2708, -260, 9.54),
        "tiered_position_cap_20pct": (312, 331.88, -25.78, -2651, -255, 9.12),
        "tiered_position_cap_15pct": (312, 294.32, -21.08, -2566, -253, 8.28),
    }
    for experiment, reported in long_reported.items():
        row = consolidated.loc[
            (consolidated["window"] == "long_development")
            & (consolidated["experiment"] == experiment)
        ].iloc[0]
        section = f"risk sequence: {experiment}"
        exact(section, "positions", str(consolidated_path), int(row.closed_positions), reported[0])
        numeric(section, "return_pct", str(consolidated_path), 100 * row.total_return, reported[1], 0.005)
        numeric(section, "max_drawdown_pct", str(consolidated_path), 100 * row.max_drawdown, reported[2], 0.005)
        numeric(section, "worst_pnl", str(consolidated_path), row.worst_trade_pnl, reported[3], 0.51)
        numeric(section, "expected_shortfall_5pct", str(consolidated_path), row.expected_shortfall_5pct_pnl, reported[4], 0.51)
        numeric(section, "profit_factor", str(consolidated_path), row.profit_factor, reported[5], 0.005)

    event_cap_path = ARTIFACTS / "semantic_event_cap_15pct_summary.json"
    event_cap = json.loads(event_cap_path.read_text(encoding="utf-8"))
    for metric, actual, reported, tolerance in [
        ("positions", event_cap["closed_positions"], 312, 0),
        ("return_pct", 100 * event_cap["total_return"], 378.32, 0.005),
        ("max_drawdown_pct", 100 * event_cap["max_drawdown"], -28.36, 0.005),
        ("worst_pnl", event_cap["worst_trade_pnl"], -2755, 0.51),
        ("expected_shortfall_5pct", event_cap["expected_shortfall_5pct_pnl"], -265, 0.51),
        ("profit_factor", event_cap["profit_factor"], 9.93, 0.005),
    ]:
        numeric("risk sequence: semantic_event_cap_15pct", metric, str(event_cap_path), actual, reported, tolerance)

    primary_summary_path = ARTIFACTS / "tiered_position_cap_15pct_summary.json"
    primary_summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    for metric, actual, reported, tolerance in [
        ("wins", primary_summary["closed_positions"] - primary_summary["loss_count"], 310, 0),
        ("end_equity", primary_summary["total_equity"], 39432, 0.51),
        ("mean_position_return_pct", 100 * primary_summary["mean_trade_return"], 3.65, 0.005),
        ("bootstrap_low_pct", 100 * primary_summary["mean_trade_return_ci_low"], 2.36, 0.005),
        ("bootstrap_high_pct", 100 * primary_summary["mean_trade_return_ci_high"], 4.78, 0.005),
        ("median_position_return_pct", 100 * primary_summary["median_trade_return"], 0.20, 0.005),
    ]:
        numeric("full-sample prose", metric, str(primary_summary_path), actual, reported, tolerance)

    directional_path = ARTIFACTS / "hyperparameter" / "hp_directional_traders_3_summary.json"
    directional = json.loads(directional_path.read_text(encoding="utf-8"))
    numeric("support-breadth prose", "return_pct", str(directional_path), 100 * directional["total_return"], 268.32, 0.005)
    numeric("support-breadth prose", "max_drawdown_pct", str(directional_path), 100 * directional["max_drawdown"], -15.00, 0.005)

    hyper_path = SUPPLEMENT / "semantic_risk_hyperparameter_robustness.csv"
    hyper = pd.read_csv(hyper_path)
    hyper_reported = {
        "hp_max_concentration_0p65": (278, 2, 198.15, -15.00, -2355, 6.16),
        "tiered_position_cap_15pct": (312, 2, 294.32, -21.08, -2566, 8.28),
        "hp_max_concentration_0p85": (347, 3, 281.95, -20.17, -6460, 3.45),
        "hp_mean_history_1p0": (342, 3, 291.25, -23.60, -3066, 5.01),
        "hp_mean_history_2p0": (296, 2, 249.85, -21.34, -2406, 7.47),
        "hp_position_cap_10pct": (312, 2, 163.46, -16.33, -1457, 6.92),
        "hp_position_cap_20pct": (312, 2, 331.88, -25.78, -2651, 9.12),
    }
    for experiment, reported in hyper_reported.items():
        rows = hyper.loc[hyper["source_experiment"] == experiment]
        row = rows.iloc[0]
        section = f"hyperparameter: {experiment}"
        exact(section, "positions", str(hyper_path), int(row.closed_positions), reported[0])
        exact(section, "losses", str(hyper_path), int(row.loss_count), reported[1])
        numeric(section, "return_pct", str(hyper_path), row.total_return_pct, reported[2], 0.005)
        numeric(section, "max_drawdown_pct", str(hyper_path), row.max_drawdown_pct, reported[3], 0.005)
        numeric(section, "worst_pnl", str(hyper_path), row.worst_trade_pnl_usdc, reported[4], 0.51)
        numeric(section, "profit_factor", str(hyper_path), row.profit_factor, reported[5], 0.005)

    execution_path = SUPPLEMENT / "execution_recent_complete_tape_results.csv"
    execution = pd.read_csv(execution_path).set_index("experiment")
    execution_reported = {
        "baseline": (48, 8.21, -5.16, 100.0, 48, None),
        "execution_partial_target_token_10pct_24h": (48, 3.12, -3.83, 79.4, 20, 9.29),
        "execution_partial_target_token_25pct_24h": (48, 4.30, -3.58, 87.1, 30, 5.31),
        "execution_partial_target_token_50pct_24h": (48, 4.30, -3.49, 87.5, 40, 3.18),
        "execution_partial_target_token_25pct_30m": (44, 0.80, -4.35, 29.8, 6, 0.58),
        "execution_partial_target_token_25pct_2h": (47, 1.13, -3.92, 58.6, 22, 1.92),
        "execution_partial_target_token_25pct_6h": (47, 2.35, -4.78, 74.0, 26, 5.29),
        "execution_partial_target_token_25pct_24h_entry_0999": (48, 4.31, -3.59, 87.1, 30, 6.82),
        "execution_partial_target_token_25pct_24h_entry_0998": (37, 4.34, -3.68, 73.0, 24, 6.72),
        "execution_partial_target_token_25pct_24h_recheck": (44, 2.49, -3.43, 67.4, 33, 3.87),
        "execution_partial_target_sell_25pct_6h": (45, 1.21, -6.07, 55.2, 20, 5.71),
    }
    for experiment, reported in execution_reported.items():
        row = execution.loc[experiment]
        section = f"execution grid: {experiment}"
        exact(section, "positions", str(execution_path), int(row.closed_positions), reported[0])
        numeric(section, "return_pct", str(execution_path), 100 * row.total_return, reported[1], 0.005)
        numeric(section, "max_drawdown_pct", str(execution_path), 100 * row.max_drawdown, reported[2], 0.005)
        if experiment == "baseline":
            fill_ratio, full_orders = 100.0, 48
        else:
            fill_ratio = 100 * row.execution_requested_fill_ratio
            full_orders = int(row.execution_fully_filled_orders)
        numeric(section, "fill_ratio_pct", str(execution_path), fill_ratio, reported[3], 0.051)
        exact(section, "full_orders", str(execution_path), full_orders, reported[4])
        if reported[5] is not None:
            numeric(section, "p90_hours", str(execution_path), row.execution_p90_completion_seconds / 3600, reported[5], 0.0051)

    baseline = execution.loc["baseline"]
    primary_execution = execution.loc["execution_partial_target_token_25pct_24h"]
    for metric, actual, reported, tolerance in [
        ("baseline_median_print_participation_pct", 100 * baseline.execution_median_print_participation, 245, 0.51),
        ("baseline_p90_print_participation_pct", 100 * baseline.execution_p90_print_participation, 2746, 0.51),
        ("primary_child_fills", primary_execution.fills, 808, 0),
        ("primary_median_child_fills", primary_execution.execution_median_child_fills, 8, 0),
        ("primary_pnl", primary_execution.net_realized_pnl, 430, 0.51),
    ]:
        numeric("recent execution prose", metric, str(execution_path), actual, reported, tolerance)

    capacity_reported = {
        "execution_partial_target_token_25pct_24h_capital_10000": (430, 4.30, 87.1, 30, 5.31),
        "execution_partial_target_token_25pct_24h_capital_25000": (780, 3.12, 80.4, 31, 9.29),
        "execution_partial_target_token_25pct_24h_capital_50000": (941, 1.88, 70.1, 27, 18.10),
        "execution_partial_target_token_25pct_24h_capital_100000": (1188, 1.19, 61.5, 24, 23.42),
    }
    for experiment, reported in capacity_reported.items():
        row = execution.loc[experiment]
        section = f"capacity curve: {experiment}"
        numeric(section, "pnl", str(execution_path), row.net_realized_pnl, reported[0], 0.51)
        numeric(section, "return_pct", str(execution_path), 100 * row.total_return, reported[1], 0.005)
        numeric(section, "fill_ratio_pct", str(execution_path), 100 * row.execution_requested_fill_ratio, reported[2], 0.051)
        exact(section, "full_orders", str(execution_path), int(row.execution_fully_filled_orders), reported[3])
        numeric(section, "p90_hours", str(execution_path), row.execution_p90_completion_seconds / 3600, reported[4], 0.0051)

    cost_path = SUPPLEMENT / "execution_primary_fixed_path_cost_stress.csv"
    cost = pd.read_csv(cost_path)
    for slip, fee, gross, fees, paper_return in [
        (10, 0, 430.0, 0.0, 4.30),
        (30, 20, 411.0, 22.3, 3.89),
        (50, 50, 398.1, 55.8, 3.42),
    ]:
        row = cost.loc[
            (cost["total_slippage_bps"] == slip) & (cost["fee_bps"] == fee)
        ].iloc[0]
        section = f"cost stress: {slip}/{fee} bps"
        numeric(section, "gross_pnl", str(cost_path), row.gross_pnl, gross, 0.051)
        numeric(section, "fees", str(cost_path), row.fees, fees, 0.051)
        numeric(section, "return_pct", str(cost_path), 100 * row.portfolio_return, paper_return, 0.005)

    annual_path = ARTIFACTS / "tiered_position_cap_15pct_annual_results.csv"
    annual = pd.read_csv(annual_path).set_index("period")
    for year, reported in {
        2024: (59, 59, 15.22, 10000, 11522, 0.00013),
        2025: (165, 163, 185.69, 11522, 32917, -2566),
        2026: (88, 88, 19.79, 32917, 39432, 0.00017),
    }.items():
        row = annual.loc[year]
        section = f"annual table: {year}"
        exact(section, "positions", str(annual_path), int(row.closed_positions), reported[0])
        exact(section, "wins", str(annual_path), int(row.wins), reported[1])
        numeric(section, "return_pct", str(annual_path), 100 * row.portfolio_return, reported[2], 0.005)
        numeric(section, "start_equity", str(annual_path), row.start_equity, reported[3], 0.51)
        numeric(section, "end_equity", str(annual_path), row.end_equity, reported[4], 0.51)
        tolerance = 0.51 if abs(reported[5]) > 1 else 0.0000051
        numeric(section, "worst_pnl", str(annual_path), row.worst_trade_pnl, reported[5], tolerance)

    ablation_path = SUPPLEMENT / "experiment_results.csv"
    ablations = pd.read_csv(ablation_path).set_index("experiment")
    for experiment, reported in {
        "semantic_main": (89, 98.88, -8.47, -17.80, 0.94),
        "global_skill_no_semantics": (89, 98.88, -8.70, -18.03, 0.94),
        "randomized_semantic_assignment": (89, 98.88, -8.48, -17.97, 0.96),
        "favorite_price_only": (132, 96.97, 2.40, -2.34, -0.05),
        "volume_weighted_consensus": (91, 97.80, -9.82, -16.17, -0.11),
        "fixed_fraction_2pct": (89, 98.88, 0.91, -2.07, 0.94),
    }.items():
        row = ablations.loc[experiment]
        section = f"signal ablation: {experiment}"
        exact(section, "positions", str(ablation_path), int(row.closed_positions), reported[0])
        numeric(section, "positive_pct", str(ablation_path), 100 * row.win_rate, reported[1], 0.005)
        numeric(section, "return_pct", str(ablation_path), 100 * row.total_return, reported[2], 0.005)
        numeric(section, "max_drawdown_pct", str(ablation_path), 100 * row.max_drawdown, reported[3], 0.005)
        numeric(section, "mean_position_return_pct", str(ablation_path), 100 * row.mean_trade_return, reported[4], 0.005)

    figure_path = SUPPLEMENT / "main_long_recent_equity.csv"
    figure = pd.read_csv(figure_path)
    long_figure = figure.loc[figure["panel"] == "long"]
    for experiment, rows in long_figure.groupby("experiment"):
        terminal_equity = rows.sort_values("ts").iloc[-1].total_equity
        source_return = consolidated.loc[
            (consolidated["window"] == "long_development")
            & (consolidated["experiment"] == experiment),
            "total_return",
        ].iloc[0]
        numeric(
            f"main equity figure, long panel: {experiment}",
            "terminal_equity",
            str(figure_path),
            terminal_equity,
            10000 * (1 + source_return),
            1e-6,
        )

    recent_figure = figure.loc[figure["panel"] == "recent"]
    for experiment, rows in recent_figure.groupby("experiment"):
        terminal_equity = rows.sort_values("ts").iloc[-1].total_equity
        numeric(
            f"main equity figure, recent panel: {experiment}",
            "terminal_equity",
            str(figure_path),
            terminal_equity,
            execution.loc[experiment].total_equity,
            1e-6,
        )

    tex_path = PAPER / "main.tex"
    tex = tex_path.read_text(encoding="utf-8")
    exact(
        "paper wiring",
        "long-window equity is a separate main-text figure",
        str(tex_path),
        "\\includegraphics[width=0.88\\linewidth]{main_long_equity.png}" in tex,
        True,
    )
    exact(
        "paper wiring",
        "recent execution equity is a separate main-text figure",
        str(tex_path),
        "\\includegraphics[width=0.88\\linewidth]{main_recent_execution_equity.png}" in tex,
        True,
    )
    exact(
        "paper wiring",
        "obsolete duplicate risk figure removed",
        str(tex_path),
        "risk_control_equity.png" not in tex,
        True,
    )
    exact(
        "paper wiring",
        "long panel has two distinct specifications",
        str(figure_path),
        long_figure["experiment"].nunique(),
        2,
    )
    exact(
        "paper wiring",
        "recent panel has four distinct execution specifications",
        str(figure_path),
        recent_figure["experiment"].nunique(),
        4,
    )

    output = {
        "status": "passed",
        "checks": len(checks),
        "scope": (
            "All performance tables, headline specifications, and terminal "
            "equity values in both panels of the main figure."
        ),
        "details": checks,
    }
    output_path = SUPPLEMENT / "paper_result_consistency_audit.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)} checks; wrote {output_path}")


if __name__ == "__main__":
    main()
