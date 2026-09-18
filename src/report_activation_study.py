"""Retain and compare every activation-grid result without touching the paper."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
from pathlib import Path

import numpy as np

from expand_market_universe import read, dump, sha, ROOT
from market_activation import GATES
from prepare_activation_study import OUTPUT, SOURCE
from plot_style import configure_paper_plots


def input_audit(count):
    cohort = read(SOURCE / 'cohort.json')[:count]
    counters, gate_counts, half = Counter(), Counter(), []
    mismatches, checked = [], 0
    maximum_score_difference = 0.
    for item in cohort:
        mid = item['market_id']
        folder = OUTPUT / 'normalized' / mid
        m = read(folder / 'manifest.json')
        counters.update(m['counters'])
        for g, activation in m['activations'].items():
            assert activation is None or activation['time'] <= 1786171350
            gate_counts[g] += activation is not None
        if m['payout_yes'] == .5:
            half.append(dict(rank=m['rank'], market_id=mid, question=m['market']['question'],
                             resolved_at=m['market']['resolved_at'], evidence=m['payout_evidence']))
        # Identical binary markets with neither corrected raw field must retain
        # identical events and wallet labels; otherwise stop the report.
        if m['market']['resolution'] and not m['counters'].get('zero_price_corrected') and not m['counters'].get('outcome_index_corrected'):
            with np.load(folder / 'data.npz', allow_pickle=False) as a, np.load(SOURCE / 'normalized' / mid / 'data.npz', allow_pickle=False) as b:
                for key in ('trade_time', 'trade_side', 'trade_price', 'trade_size'):
                    if not np.array_equal(a[key], b[key]):
                        mismatches.append((mid, key))
                if not np.array_equal(a['wallets'][a['trade_wallet']], b['wallets'][b['trade_wallet']]):
                    mismatches.append((mid, 'trade_wallet'))
                wa, wb = a['wallets'][a['baseline_history_wallet']], b['wallets'][b['history_wallet']]
                ia, ib = np.argsort(wa), np.argsort(wb)
                if not np.array_equal(wa[ia], wb[ib]):
                    mismatches.append((mid, 'history_wallet'))
                elif len(ia):
                    delta = float(np.max(np.abs(a['baseline_history_score'][ia]-b['history_score'][ib])))
                    maximum_score_difference = max(maximum_score_difference, delta)
                    if delta > 1e-9 or not np.allclose(a['baseline_history_notional'][ia], b['history_notional'][ib], rtol=1e-12, atol=1e-6):
                        mismatches.append((mid, 'history_values'))
            checked += 1
    result = dict(count=count, counters=dict(counters), activated_markets=dict(gate_counts),
                  fractional_markets=half, unchanged_binary_markets_checked=checked,
                  max_score_difference=maximum_score_difference, unexpected_mismatches=mismatches)
    dump(OUTPUT / f'input_audit_{count}.json', result)
    assert not mismatches, mismatches[:10]
    return result


def cumulative_comparison(count):
    """Compare cumulative and rolling volume with all other gates fixed."""
    from matplotlib import pyplot as plt
    folder=OUTPUT/'results'/str(count)
    paths=[('24h $250k',folder/'both__p90_v250k')]
    paths += [(f'Cumulative ${v/1000:g}m' if v>=1000 else 'Cumulative $250k',
               OUTPUT/f'cumulative-volume-check/results/{count}/both__p90_cum{v}k')
              for v in (250,1000,5000)]
    rows=[]
    reference={p['market_id']:p for p in read(paths[0][1]/'positions.json')}
    for label,path in paths:
        if not (path/'complete.json').exists():
            continue
        r=read(path/'complete.json')
        positions={p['market_id']:p for p in read(path/'positions.json')}
        common=positions.keys() & reference.keys()
        same={m for m in common if positions[m]['direction']==reference[m]['direction']}
        rows.append(dict(label=label,path=str(path),summary=r,
            overlap=dict(same_option=len(same),opposite_option=len(common-same),
                         reference_only=len(reference.keys()-positions.keys()),
                         variant_only=len(positions.keys()-reference.keys())),
            losing_positions=sorted((p for p in positions.values() if p['pnl']<0),key=lambda p:p['pnl'])))
    dump(OUTPUT/f'cumulative_comparison_{count}.json',rows)
    lines=['## Cumulative versus trailing-24h volume','',
        'Only the dollar-volume aggregation changes. Both scopes, the trailing-24h '
        'quantity-weighted price threshold p=0.90, 100 recent prints, warmup, and '
        'the original portfolio rules are identical. Cumulative means the sum of '
        'actual-token dollar trades observed since the available source floor '
        '(2023-09-18, or a later first print) through the current completed hour. '
        'It is not final Gamma volume, and any earlier unavailable lifetime trades '
        'are not counted. These extra thresholds were chosen after the rolling '
        'results, at the user\'s request; all outcomes are retained.','',
        '| Volume requirement | Return | Drawdown | Closed / losses | Activated markets |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        r=row['summary']
        lines.append(f'| {row["label"]} | {r["total_return"]:.2%} | {-r["max_drawdown"]:.2%} | '
                     f'{r["closed_positions"]} / {r["loss_count"]} | {r["activated_markets"]} |')
    if len(rows)<4:
        lines += ['',f'Incomplete: {len(rows)-1}/3 cumulative cases finished.']
    lines += ['', 'The same $250k cumulative threshold is weaker than $250k in 24 hours: '
        'it admits a superset of markets and can activate shared markets earlier. '
        'The price and recent-print tests still require recent activity. '
        'Passing the gate does not guarantee that expert-consensus and portfolio '
        'constraints will permit a trade.','']
    lines += ['### Traded markets relative to the rolling $250k reference','',
        '| Variant | Same market and option | Opposite option | New markets | Reference-only markets |',
        '|---|---:|---:|---:|---:|']
    for row in rows[1:]:
        o=row['overlap']
        lines.append(f'| {row["label"]} | {o["same_option"]} | {o["opposite_option"]} | '
                     f'{o["variant_only"]} | {o["reference_only"]} |')
    lines += ['', 'Different volume horizons change both entry timing and the historical '
        'wallet evidence. The larger universe also changes wallet evidence, so even '
        'shared markets need not have identical signals or orders. Position overlap '
        'is descriptive, not an additive attribution of portfolio profit.','']
    (OUTPUT/f'CUMULATIVE_{count}.md').write_text('\n'.join(lines))
    configure_paper_plots()
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for row,color in zip(rows,('#111111','#BB5522','#3377AA','#228855')):
        curve=read(Path(row['path'])/'equity.json')
        x=[datetime.fromisoformat(r['ts']) for r in curve]
        y=np.asarray([r['total_equity']/10000 for r in curve])
        dd=100*(y/np.maximum.accumulate(np.maximum(y,1))-1)
        axes[0].plot(x,y,label=row['label'],color=color,lw=1.7)
        axes[1].plot(x,dd,label=row['label'],color=color,lw=1.5)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='best')
        ax.tick_params(axis='x',rotation=25)
    axes[0].set(title='Portfolio equity',ylabel='Equity / initial capital')
    axes[0].axhline(1,color='#888888',lw=.7,ls=':')
    axes[1].set(title='Drawdown from running peak',ylabel='Drawdown (%)')
    fig.suptitle(f'{count:,} markets: change only the volume horizon',fontsize=13)
    fig.text(.5,.005,'Both consensus and wallet history cut; p=0.90 fixed. Exploratory, quantity-unconstrained replay.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.95))
    for ext in ('svg','png'):
        fig.savefig(OUTPUT/f'cumulative_equity_{count}.{ext}',dpi=170)
    return lines,fig


def report(count,cumulative_lines,cumulative_figure):
    folder = OUTPUT / 'results' / str(count)
    if not folder.exists():
        return
    results = {p.parent.name: read(p) for p in folder.glob('*/complete.json')}
    baseline = results.get('baseline')
    selected_gates = list(GATES) if count == 3000 else ['p90_v50k', 'p90_v250k']
    reference_positions = ({p['market_id']:p for p in read(folder / 'baseline/positions.json')}
                           if baseline else {})
    comparisons = {}
    for name in results:
        if name == 'baseline' or baseline is None:
            continue
        positions = {p['market_id']:p for p in read(folder / name / 'positions.json')}
        common = reference_positions.keys() & positions.keys()
        same = {m for m in common if reference_positions[m]['direction'] == positions[m]['direction']}
        losing = {m for m,p in reference_positions.items() if p['pnl'] < 0}
        # Non-overlap is descriptive. Compounding reallocates capital, so this
        # is not an isolated dollar "saving" caused by the gate.
        comparisons[name] = dict(shared_markets=len(common), same_option=len(same),
            opposite_option=len(common-same), baseline_losses_not_repeated_same_option=len(losing-same),
            baseline_losses_repeated_same_option=len(losing & same),
            baseline_only=len(reference_positions.keys()-positions.keys()),
            variant_only=len(positions.keys()-reference_positions.keys()),
            baseline_losing_market_ids_not_repeated=sorted(losing-same))
    dump(OUTPUT / f'comparison_{count}.json', dict(results=results, positions=comparisons))
    lines = [f'# Causal market activation: {count:,} markets', '',
        'Exploratory same-window comparisons, 2023-09-18 through 2026-08-08. '
        'The archived universe is retrospective. These are quantity-unconstrained '
        'trade-print replays, not a claim of executable live returns.', '',
        'The input corrections (fractional settlement, zero token prices, and '
        'conflicting API outcome indexes) apply to every row, including the no-gate baseline. '
        'The old +335.66% late-tail result is not the baseline for this experiment.', '',
        (f'No gate: return {baseline["total_return"]:.2%}; maximum drawdown {-baseline["max_drawdown"]:.2%}; '
        f'{baseline["closed_positions"]} closed positions, {baseline["loss_count"]} losing, '
        f'{baseline["open_positions"]} still open.' if baseline else
         'No-gate baseline is still running. Its result is not estimated or taken from a different input version.'), '',
        '## What the cutoff means', '',
        'After at least 24 hours of observed tape, examine each completed UTC hour. '
        'The preceding 24 hours must contain at least 100 prints and the specified '
        'dollar volume. The quantity-weighted YES-equivalent price must be at least p '
        'or at most 1-p. The market must be within the existing 40-day scheduled-close '
        'horizon. Activate once, at the end of that hour, and admit only subsequent prints. '
        'The trigger bar does not contribute to the new consensus.', '',
        'Flow-only changes current-market consensus but keeps full-history wallet labels. '
        'Both changes current consensus and the historical trading contributions used '
        'to rate wallets. Historical labels are still unavailable until settlement. '
        'Suffix scores measure incremental signed trades, not a reconstructed wallet account '
        'return or opening inventory. BGE similarity, expert filters, sizing, delay, price '
        'limits and caps otherwise remain unchanged.', '',
        '## All initial parameter combinations', '',
        '| Price support p | 24h volume | Flow-only return / drawdown | Flow-only closed / losses | Both return / drawdown | Both closed / losses |',
        '|---|---:|---:|---:|---:|---:|']
    for gate in selected_gates:
        spec = GATES[gate]
        cells = []
        for scope in ('flow', 'both'):
            r = results.get(f'{scope}__{gate}')
            cells.extend([f'{r["total_return"]:.2%} / {-r["max_drawdown"]:.2%}', f'{r["closed_positions"]} / {r["loss_count"]}'] if r else ['pending','pending'])
        lines.append(f'| {spec.min_conviction:.2f} | ${spec.min_recent_notional:,.0f} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'A selected best row is in-sample tuning, not independent validation. '
        'Adjacent rows report initial parameter sensitivity. A return improvement alone '
        'does not establish that early traders are guessing: the gate also changes entry '
        'prices, coverage, expert support and capital allocation.', '',
        '## Position comparison', '',
        '| Variant | Same market/option as baseline | Baseline losing options not repeated | New markets |',
        '|---|---:|---:|---:|']
    for name, c in comparisons.items():
        lines.append(f'| {name} | {c["same_option"]} | {c["baseline_losses_not_repeated_same_option"]} | {c["variant_only"]} |')
    if baseline is None:
        lines += ['', 'Position comparisons to no gate are pending its completed replay.']
    if count == 10000:
        lines += ['', 'The p=0.90 / $50,000 check was chosen before the first accepted '
            '3,000-market return. The p=0.90 / $250,000 pair was added after seeing '
            'the promising 3,000-market both-scope result. Neither check is a new '
            'calendar holdout; see EXPERIMENT_PROTOCOL.md for the selection sequence.']
    lines += ['', 'A loss not repeated can be a skipped position, a changed direction or a changed '
        'allocation path. It is not necessarily a loss avoided solely by a better expert rating.', '',
        '## Local volume sensitivity', '',
        'After identifying the p=0.90 / $250,000 both-scope candidate, repeat at '
        '$200,000 and $300,000, retaining the same data and other rules. This is '
        'post-selection sensitivity, not an independent test.', '',
        '| 24h volume | Both-scope return | Drawdown | Closed / losses |',
        '|---:|---:|---:|---:|']
    for v in (200,250,300):
        path=(folder/f'both__p90_v{v}k/complete.json' if v==250 else
              OUTPUT/f'local-volume-check/results/{count}/both__p90_v{v}k/complete.json')
        if path.exists():
            r=read(path)
            lines.append(f'| ${v*1000:,} | {r["total_return"]:.2%} | {-r["max_drawdown"]:.2%} | {r["closed_positions"]} / {r["loss_count"]} |')
        else:
            lines.append(f'| ${v*1000:,} | pending | pending | pending |')
    lines += ['',*cumulative_lines]
    if (OUTPUT/'etf_signal_audit.json').exists():
        lines += ['', '## Controlled example: Ethereum ETF approved by May 31?', '',
            'The 10,000-market flow-only p=0.90 / $250,000 replay bought NO on '
            '2024-05-18 and lost its $1,500 notional. Both-scope did not buy. A '
            'separate replay holds the target-market prefix and initial capital '
            'fixed and reproduces the original signal, price and notional exactly. '
            'Only the history labels change.', '',
            '| At the original signal time | Full wallet history | Post-activation wallet history |',
            '|---|---:|---:|',
            '| Qualified wallets | 8 | 1 |',
            '| Directional qualified wallets | 7, NO | 1, YES |',
            '| Effective directional count | 2.304 | 1.000 |',
            '| Largest directional weight | 57.80% | 100.00% |',
            '| Mean effective history support | 1.965 | 1.000 |',
            '| Order submitted | Yes | No |', '',
            'The post-activation signal fails the unchanged minimum count (2), '
            'effective count (1.25), maximum directional concentration (75%) and '
            'mean effective history (1.5) constraints. This isolates a history-based '
            'reason for skipping this particular loss; it does not prove that the '
            'excluded wallets were guessing, nor that every skipped loss has the same cause.', '',
            'This was also the flow-only strategy\'s first trade. A future mechanism '
            'analysis should distinguish market-specific history filtering from a '
            'general startup/warmup effect.']
    lines += ['',
        '## Reproduce', '',
        '```sh', 'PYTHONPATH=src .venv/bin/python src/prepare_activation_study.py --count 10000 --workers 4',
        f'PYTHONPATH=src .venv/bin/python src/run_activation_study.py --count {count} --workers 4',
        'PYTHONPATH=src .venv/bin/python src/run_activation_neighborhood.py --workers 4',
        'PYTHONPATH=src .venv/bin/python src/run_activation_cumulative.py --prepare-workers 4 --replay-workers 6',
        'PYTHONPATH=src .venv/bin/python src/audit_activation_example.py',
        'PYTHONPATH=src .venv/bin/python src/report_activation_study.py', '```', '',
        'Source public response pages, hashes and acquisition bounds are retained in the '
        'universe-expansion artifact directory. Each variant retains exact config, all '
        'fills/positions, open positions, daily equity and accounting checks. '
        'Neither frozen ICLR artifact nor any Git ref is changed.']
    (OUTPUT / f'RESULTS_{count}.md').write_text('\n'.join(lines)+'\n')
    configure_paper_plots()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    colors = dict(zip(GATES, ['#3377AA', '#BB5522', '#228855', '#AA3377', '#777722', '#7755AA']))
    for ax, scope, title in zip(axes, ('flow','both'), ('Cut current consensus only','Cut current consensus and wallet history')):
        specifications = [('baseline','#111111','No gate')] + [(f'{scope}__{g}',colors[g],
            f'p={GATES[g].min_conviction:.2f}, V={GATES[g].min_recent_notional/1000:.0f}k') for g in selected_gates]
        for name, color, label in specifications:
            if name not in results:
                continue
            curve = read(folder/name/'equity.json')
            x = [datetime.fromisoformat(r['ts']) for r in curve]
            y = [r['total_equity']/10000 for r in curve]
            ax.plot(x,y,label=label,color=color,lw=1.8 if name=='baseline' else 1.25)
        ax.set_title(title)
        ax.axhline(1,color='#888888',lw=.7,ls=':')
        ax.grid(alpha=.2)
        ax.legend(loc='best',ncol=2,fontsize=7.5,title='24-hour gate; V in dollars',title_fontsize=8)
        ax.tick_params(axis='x',rotation=25)
    axes[0].set_ylabel('Equity / initial capital')
    fig.suptitle(f'{count:,} markets: exploratory activation comparison', fontsize=13)
    footer='Same portfolio rules and corrected public tape; trade-print execution, not an executable-capacity estimate.'
    if baseline is None:
        footer='No-gate baseline still running and not plotted. Completed gate variants use identical portfolio rules.'
    fig.text(.5,.005,footer,
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.95))
    for ext in ('svg','png'):
        fig.savefig(OUTPUT/f'equity_{count}.{ext}',dpi=170)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUTPUT/f'equity_{count}.pdf') as pdf:
        pdf.savefig(cumulative_figure)
        pdf.savefig(fig)
    plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-count',type=int)
    args=parser.parse_args()
    if args.audit_count:
        print(input_audit(args.audit_count))
    for count in (3000,10000):
        lines,fig=cumulative_comparison(count)
        report(count,lines,fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    protocol=read(OUTPUT/'protocol.json')
    for k,f in [('frozen_pdf_sha256','main.pdf'),('frozen_zip_sha256','anonymous_code.zip')]:
        assert sha(ROOT/'reports/polymarket_iclr2027/frozen'/f)==protocol[k]


if __name__=='__main__':
    main()
