"""Summarize fixed-universe replays without changing either paper version."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

from expand_market_universe import DEFAULT, ROOT, read, dump, sha

CASES = ["markets3000_history3000", "markets3000_history10000", "markets10000_history10000"]
LABELS = ["3k trading / 3k history", "3k trading / 10k history", "10k trading / 10k history"]


def main(output=DEFAULT):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from plot_style import configure_paper_plots

    configure_paper_plots()
    manifest = read(output / "study_manifest.json")
    assert sha(ROOT / "config/paper_experiments.json") == manifest["reference_config_sha256"]
    records = [read(output / "normalized" / m["market_id"] / "manifest.json")
               for m in read(output / "cohort.json")]
    metadata = read(output / "metadata.json.gz")
    ranks = {r["market_id"]: r["rank"] for r in records}
    def events(mids):
        return {str(e["id"]) for mid in mids for e in metadata[mid].get("events", []) if e.get("id")}
    cohort_counts = {}
    for n in (3000, 10000):
        selected = records[:n]
        cohort_counts[str(n)] = dict(
            candidates=len(selected), markets_with_window_trades=sum(r["source_rows"] > 0 for r in selected),
            distinct_gamma_events=len(events(r["market_id"] for r in selected)),
            quarantined_price_rows=sum(r.get("quarantined_price_rows",0) for r in selected),
            markets_with_entry_horizon_trades=sum(r["eligible_trades"] > 0 for r in selected),
            source_rows=sum(r["source_rows"] for r in selected),
            eligible_rows=sum(r["eligible_trades"] for r in selected),
            wallet_market_settlements=sum(r["settlements"] for r in selected),
            pre_window_created=sum(datetime.fromisoformat(r["market"]["created_at"]) < datetime(2023,9,18)
                                   for r in selected))
    results, support, sources, held, open_risk = {}, {}, {}, {}, {}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    colors = ["#58677b", "#b65a22", "#126e80"]
    for case, label, color in zip(CASES, LABELS, colors):
        folder = output / "results" / case
        summary = read(folder / "complete.json")
        results[case] = summary
        fills = read(folder / "fills.json")
        positions = read(folder / "positions.json")
        closed_mids={p["market_id"] for p in positions}
        remaining=[f for f in fills if f["market_id"] not in closed_mids]
        open_risk[case]=[dict(market_id=f["market_id"],question=metadata[f["market_id"]].get("question"),
            direction=f["direction"],notional=f["notional"],quantity=f["quantity"],opened_at=f["filled_at"],
            metadata_closed=metadata[f["market_id"]].get("closed"),
            uma_resolution_status=metadata[f["market_id"]].get("umaResolutionStatus"),
            outcome_prices=metadata[f["market_id"]].get("outcomePrices"),
            warning="Unmapped settlement: closed metadata but engine has no binary outcome" if metadata[f["market_id"]].get("closed") else "Open position") for f in remaining]
        if any(r["metadata_closed"] for r in open_risk[case]):
            label += " (provisional)"
        assert math.isclose(sum(p["pnl"] for p in positions)-summary["fees"],
                            summary["net_realized_pnl"],rel_tol=1e-10,abs_tol=1e-7)
        assert math.isclose(10000+summary["net_realized_pnl"]+summary["open_market_value"]-summary["open_notional"],
                            summary["total_equity"],rel_tol=1e-9,abs_tol=1e-6)
        held[case] = {p["market_id"]: p["direction"] for p in positions}
        assert len(held[case]) == len(positions), "Aggregate repeat positions explicitly"
        support[case] = dict(
            mean_expert_history_markets=float(np.mean([f["mean_expert_history_markets"] for f in fills])) if fills else None,
            mean_directional_traders=float(np.mean([f["directional_trader_count"] for f in fills])) if fills else None,
            unique_traded_markets=len({f["market_id"] for f in fills}),
            distinct_traded_gamma_events=len(events(p["market_id"] for p in positions)),
            positions_in_original_top3000=sum(ranks[p["market_id"]]<=3000 for p in positions))
        curve = read(folder / "equity.json")
        times = [datetime.fromisoformat(r["ts"]) for r in curve]
        equity = np.array([r["total_equity"] for r in curve])
        axes[0].plot(times, equity, label=label, color=color, linewidth=1.6)
        axes[1].plot(times, 100*(equity/np.maximum.accumulate(equity)-1), color=color, linewidth=1.2)
        sources[case] = {f: sha(folder / f) for f in ["complete.json", "config.json", "equity.json", "fills.json"]}
    axes[0].set_ylabel("Portfolio value (USDC)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].legend(frameon=False, loc="best")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "equity_comparison.pdf")
    fig.savefig(output / "equity_comparison.svg")
    fig.savefig(output / "equity_comparison.png", dpi=160)
    if all(r["total_equity"] > 0 for case in CASES for r in read(output / "results" / case / "equity.json")):
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Portfolio value (USDC; log scale)")
        fig.savefig(output / "equity_comparison_log.svg")
    plt.close(fig)
    frozen_checks = {path: sha(ROOT / path) == value for path,value in manifest["frozen_sha256"].items()}
    assert all(frozen_checks.values()), "Frozen artifacts changed"
    reference = read(ROOT / "config/paper_experiments.json")["groups"]["full_window"]["experiments"]["tiered_position_cap_15pct"]["config"]
    config_checks = {case: read(output / "results" / case / "config.json") == reference for case in CASES}
    assert all(config_checks.values()), "Strategy parameters changed"
    source_audit=read(output / "source_audit.json")
    array_audit=read(output / "array_audit.json")
    embedding_audit=read(output / "embedding_audit.json")
    assert source_audit["status"]==array_audit["status"]=="passed"
    assert embedding_audit["exact_vector_match"]
    frozen_reference_checks=read(output / "frozen_reference/full_window/verification.json")
    assert frozen_reference_checks and all(c["passed"] for c in frozen_reference_checks)
    overlap = []
    for left,right in zip(CASES,CASES[1:]):
        a,b=held[left],held[right]
        shared=set(a)&set(b)
        overlap.append(dict(left=left,right=right,shared_markets=len(shared),
                            same_option=sum(a[mid]==b[mid] for mid in shared),
                            opposite_option=sum(a[mid]!=b[mid] for mid in shared),
                            left_only=len(set(a)-set(b)),right_only=len(set(b)-set(a))))
    dump(output / "comparison.json", dict(window=manifest, cohorts=cohort_counts, results=results,
                                          expert_support=support, position_overlap=overlap,
                                          overlap_scope="Closed positions only; open exposure is separately listed",
                                          open_risk=open_risk))
    dump(output / "validation.json", dict(frozen_unchanged=frozen_checks, configs_unchanged=config_checks,
                                          source_audit=source_audit, array_audit=array_audit,
                                          embedding_audit=embedding_audit,
                                          frozen_reference_checks=frozen_reference_checks,
                                          result_hashes=sources,
                                          open_risk=open_risk,
                                          unresolved_settlement_metadata=any(r["metadata_closed"] for rows in open_risk.values() for r in rows),
                                          code_sha256={name:sha(ROOT/"src"/name) for name in
                                              ["expand_market_universe.py","build_expanded_replay.py","report_expanded_universe.py"]}))
    lines = ["# 10,000-market expanded-history comparison", "",
             "Study window: 2023-09-18 00:00:00 UTC to 2026-08-08 06:42:30 UTC.", "",
             "Candidates are fixed by volume in the April 24 metadata archive. All three runs use the original ICLR primary strategy configuration, a fresh 10,000 USDC balance, and the same newly collected history window. No global-score variant is involved.", "",
             "| Trading universe | Expert-history universe | Closed / open positions | Return | Max drawdown | Final equity |",
             "|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        r=results[case]
        lines.append(f"| {r['trade_universe']:,} | {r['history_universe']:,} | {r['closed_positions']:,} / {r['open_positions']:,} | {r['total_return']:.2%} | {r['max_drawdown']:.2%} | {r['total_equity']:,.2f} |")
    for case,items in open_risk.items():
        for item in items:
            lines += ["", f"**Provisional open-exposure warning ({case}):** `{item['question']}` was bought on {item['opened_at']} for {item['notional']:,.2f} USDC. Gamma reports closed={item['metadata_closed']}, UMA status={item['uma_resolution_status']}, and outcomePrices={item['outcome_prices']}. The binary-outcome normalizer does not map this response to a settlement, so the backtest still carries it open. Its aggregate marked value is {results[case]['open_market_value']:,.2f} USDC. The 10k result must not be treated as a fully settlement-validated final estimate until this source ambiguity is resolved and, if needed, the path rerun. It does not affect the two 3k-trading results, which have no open positions.", ""]
    lines += ["", "The first-to-second comparison isolates additional observed expert histories while holding the trade universe fixed. The second-to-third comparison adds trading opportunities. These are separate, path-dependent portfolio replays, not additive PnL components.", "",
              "## Data coverage", "", "| Candidates | Markets with trades | Source trades | Eligible trade events | Wallet-market histories |",
              "|---:|---:|---:|---:|---:|"]
    for n in (3000,10000):
        r=cohort_counts[str(n)]
        lines.append(f"| {n:,} | {r['markets_with_window_trades']:,} | {r['source_rows']:,} | {r['eligible_rows']:,} | {r['wallet_market_settlements']:,} |")
    lines += ["", f"The candidate universes span {cohort_counts['3000']['distinct_gamma_events']:,} and {cohort_counts['10000']['distinct_gamma_events']:,} distinct Gamma events. Event counts measure grouping breadth, not statistical independence.", ""]
    diagnosis_path=output/"history_diagnosis.json"
    if diagnosis_path.exists():
        d=read(diagnosis_path)["summary"]
        no_tape=sum(r["earliest_original_eligible_trade"] is None for r in d["loss_records"])
        lines += ["## Why the matched control differs from the frozen result", "",
                  f"The new 3,000-market control holds {d['expanded_history_positions']} positions, sharing {d['shared_markets']} markets with the original {d['original_positions']}. None of its held markets is outside the original 2,989-market analysis sample. Of its new signals, {d['new_signals_before_original_tape']} precede the original market's first eligible observation or belong to a market with no eligible original observations.", "",
                  f"All {d['losses']} losing positions occur in those missing early portions: {no_tape} have no eligible original tape at all, and {d['losses']-no_tape} occur before the first original eligible trade. This does not mean only losses were missing; profitable early positions were also absent. It demonstrates material sensitivity to historical coverage. It is an observed support diagnostic, not a complete causal decomposition of the effects of histories, signal events, and the common start date.", "",
                  "For example, the Bitcoin-below-85,000-by-February-2025 position enters on February 1, 2025, whereas the original eligible tape in that market begins February 26. The original capped tape could not expose the strategy to that entry. Position payouts and realized PnLs in the new matched control were independently checked against the outcome metadata.", ""]
    lines += ["", "## Interpretation boundaries", "",
              "The old 335.66% result uses capped early histories and a different start date; it is not the matched full-history 3,000-market control reported above. The current public API imposes an approximately three-year market-query floor. Exhausted-window collection is not proof of on-chain or lifetime wallet completeness. Markets opened before the common start remain left-truncated. Markets with zero retrieved rows remain in the candidate count, not the active sample count.", "",
              "The April volume-selected cohort is retrospective before selection. Results retain the paper's quantity-unconstrained print execution; this study does not establish executable capacity. No frozen paper, PDF, ZIP, or public Git ref was modified.", "",
              "![Matched expanded-universe equity and drawdown](equity_comparison.png)", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
