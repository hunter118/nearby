"""Matched causal activation replays, separate from the frozen paper results."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import gc
import json
import math
import time

import numpy as np

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig
from build_expanded_replay import SIDES, PRIMARY, utc
from data.build_dataset import build_markets, build_resolution_events
from expand_market_universe import ROOT, START, END, read, dump, sha
from features.embeddings import SimilarityConfig
from market_activation import ActivationBacktester, FractionalResolution, GATES
from models import TimelineEvent, TradeEvent, TraderMarketSettlement
from prepare_activation_study import SOURCE, OUTPUT, NORMALIZATION_VERSION
from run_research import _summarize_result
from run_semantic_risk_study import _write_experiment_artifacts


def load_tape(count):
    cohort = read(SOURCE / "cohort.json")[:count]
    markets, manifests, chunks, remaps = {}, [], [], []
    addresses, address_index = [], {}
    for item in cohort:
        mid = item['market_id']
        folder = OUTPUT / 'normalized' / mid
        manifest = read(folder / 'manifest.json')
        if sha(folder / 'data.npz') != manifest['array_sha256']:
            raise ValueError('Research array hash mismatch.')
        manifests.append(manifest)
        record = dict(manifest['market'])
        for k in ('created_at', 'close_time', 'resolved_at'):
            record[k] = datetime.fromisoformat(record[k]) if record[k] else None
        markets.update(build_markets([record]))
        with np.load(folder / 'data.npz', allow_pickle=False) as a:
            indices = []
            for address in a['wallets'].tolist():
                if address not in address_index:
                    address_index[address] = len(addresses)
                    addresses.append(address)
                indices.append(address_index[address])
            remap = np.array(indices, dtype=np.uint32)
            remaps.append(remap)
            chunks.append(dict(market=np.full(len(a['trade_time']), item['rank']-1, dtype=np.uint32),
                wallet=remap[a['trade_wallet']], time=a['trade_time'], side=a['trade_side'],
                price=a['trade_price'], size=a['trade_size']))
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    order = np.argsort(arrays['time'], kind='stable')
    arrays = {k: values[order] for k, values in arrays.items()}
    del chunks, order, address_index
    gc.collect()
    with np.load(SOURCE / 'vectors.npz', allow_pickle=False) as a:
        assert a['market_ids'][:count].tolist() == list(markets)
        vectors = a['vectors'][:count]
    return markets, manifests, arrays, remaps, addresses, vectors


def load_history(name, manifests, remaps, addresses, markets):
    history = []
    for manifest, remap in zip(manifests, remaps):
        mid = manifest['market_id']
        with np.load(OUTPUT / 'normalized' / mid / 'data.npz', allow_pickle=False) as a:
            ws = remap[a[f'{name}_history_wallet']]
            scores, notionals = a[f'{name}_history_score'], a[f'{name}_history_notional']
            settled = markets[mid].resolved_at
            history.extend(TraderMarketSettlement(addresses[w], mid, float(s), float(n), settled)
                           for w, s, n in zip(ws, scores, notionals))
    return history


def array_timeline(markets, manifests, arrays, addresses):
    """Stable chronological merge, identical to build_timeline's tie rule.

    Streaming avoids holding tens of millions of duplicate dataclass objects.
    All eligible tape remains available for marks and fills, even before gates.
    """
    resolutions = build_resolution_events(markets)
    for m in manifests:
        market = markets[m['market_id']]
        if m['payout_yes'] is not None and market.resolution is None and market.resolved_at:
            resolutions.append(FractionalResolution(m['market_id'], market.resolved_at, None, m['payout_yes']))
    resolutions = sorted((r for r in resolutions if utc(START) <= r.resolved_at <= utc(END)),
                         key=lambda r: r.resolved_at)
    mids = list(markets)
    cursor = 0
    for i, (t, m, w, s, p, q) in enumerate(zip(arrays['time'], arrays['market'], arrays['wallet'],
                                              arrays['side'], arrays['price'], arrays['size'])):
        ts = utc(t)
        while cursor < len(resolutions) and resolutions[cursor].resolved_at < ts:
            r = resolutions[cursor]
            yield TimelineEvent('resolution', r.resolved_at, r)
            cursor += 1
        trade = TradeEvent(str(i), mids[m], addresses[w], SIDES[s], float(p), float(q), ts)
        yield TimelineEvent('trade', ts, trade)
    for r in resolutions[cursor:]:
        yield TimelineEvent('resolution', r.resolved_at, r)


def run_chunk(count, names):
    remaining = [n for n in names if not (OUTPUT / 'results' / str(count) / n / 'complete.json').exists()]
    if not remaining:
        return
    print(f'Loading {count}: {remaining}', flush=True)
    markets, manifests, arrays, remaps, addresses, vectors = load_tape(count)
    group = read(ROOT / 'config/paper_experiments.json')['groups']['full_window']
    protocol = read(OUTPUT / 'protocol.json')
    assert sha(ROOT / 'config/paper_experiments.json') == protocol['reference_config_sha256']
    config = BacktestConfig(**group['experiments'][PRIMARY]['config'])
    previous_history, estimator = None, None
    for name in remaining:
        gate = None if name == 'baseline' else name.split('__')[1]
        history_name = gate if name.startswith('both__') else 'baseline'
        folder = OUTPUT / 'results' / str(count) / name
        folder.mkdir(parents=True, exist_ok=True)
        begun = time.monotonic()
        if previous_history != history_name:
            del estimator
            gc.collect()
            history = load_history(history_name, manifests, remaps, addresses, markets)
            history_count = len(history)
            estimator = TraderSkillEstimator(markets, history, similarity_config=SimilarityConfig(True, 0),
                similarity_mode='semantic', precomputed_market_vectors=vectors)
            del history
            previous_history = history_name
        activations = None if gate is None else {m['market_id']: utc(m['activations'][gate]['time'])
                       for m in manifests if m['activations'][gate] is not None}
        dump(folder / 'config.json', dict(backtest=asdict(config), gate=asdict(GATES[gate]) if gate else None,
            gate_scope=name.split('__')[0], history=history_name, protocol_sha256=sha(OUTPUT / 'protocol.json')))
        print(f'Running {count}/{name}, history {history_count:,}, activated {len(activations) if activations is not None else count}', flush=True)
        engine = ActivationBacktester(markets, array_timeline(markets, manifests, arrays, addresses),
                                      estimator, config, activation_times=activations)
        result = engine.run()
        summary = _summarize_result(name, result, config.initial_balance)
        summary.update(universe=count, history_rows=history_count, gate=gate,
            activated_markets=len(activations) if activations is not None else count,
            eligible_prints=engine.processed, admitted_prints=engine.admitted,
            losing_positions=sum(p.pnl < 0 for p in result['closed_positions']),
            elapsed_seconds=time.monotonic()-begun, start=utc(START), end=utc(END))
        expected = config.initial_balance + sum(p.pnl for p in result['closed_positions']) - sum(f.fee for f in result['fills']) + result['open_unrealized_pnl']
        assert math.isclose(expected, result['total_equity'], abs_tol=1e-6, rel_tol=0)
        if gate:
            assert all(f.signal_time >= activations[f.market_id] for f in result['fills'])
        for p in result['closed_positions']:
            assert p.resolved_at >= p.opened_at
        _write_experiment_artifacts(name, result, summary, config.initial_balance, folder)
        for key, rows in (('positions', result['closed_positions']), ('fills', result['fills']),
                          ('open_positions', result['open_positions'])):
            dump(folder / f'{key}.json', [asdict(x) for x in rows])
        dump(folder / 'equity.json', result['equity_curve'])
        dump(folder / 'validation.json', dict(accounting=True, signals_after_activation=True,
             settlements_after_entry=True, core_execution_config_unchanged=True))
        dump(folder / 'complete.json', summary)
        print(json.dumps(summary, default=str), flush=True)
        del engine, result
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wait-inputs', action='store_true')
    parser.add_argument('--audit-inputs', action='store_true')
    parser.add_argument('--cases', nargs='+', default=['baseline'] + [f'{scope}__{g}' for scope in ('flow', 'both') for g in GATES])
    args = parser.parse_args()
    if args.wait_inputs:
        cohort = read(SOURCE / 'cohort.json')[:args.count]
        while True:
            missing = [m for m in cohort if not (OUTPUT / 'normalized' / m['market_id'] / 'manifest.json').exists()
                or read(OUTPUT / 'normalized' / m['market_id'] / 'manifest.json').get('normalization_version') != NORMALIZATION_VERSION]
            if not missing:
                break
            time.sleep(5)
    if args.audit_inputs:
        from report_activation_study import input_audit
        audit = input_audit(args.count)
        print(f"Input audit: {audit['unchanged_binary_markets_checked']} unchanged binary markets checked; "
              f"unexpected mismatches {len(audit['unexpected_mismatches'])}", flush=True)
    allowed = {'baseline'} | {f'{s}__{g}' for s in ('flow', 'both') for g in GATES}
    if not set(args.cases) <= allowed:
        parser.error('Unknown case.')
    # Share a full-history estimator/cache across flow-only variants in a chunk.
    flows = [n for n in args.cases if not n.startswith('both__')]
    both = [n for n in args.cases if n.startswith('both__')]
    if args.workers == 1:
        run_chunk(args.count, flows + both)
        return
    n_flow = min(2, max(1, args.workers//2))
    chunks = [flows[i::n_flow] for i in range(n_flow)]
    n_both = max(1, args.workers-n_flow)
    chunks += [both[i::n_both] for i in range(n_both)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(run_chunk, args.count, c) for c in chunks if c]
        for task in as_completed(tasks):
            task.result()


if __name__ == '__main__':
    main()
