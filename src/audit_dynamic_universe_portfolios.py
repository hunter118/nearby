"""Independent persisted-account audit for generic or fast inventory studies.

Never invokes a trading engine or constructs a new strategy. Fast studies read
the full normalized market tapes from their bound preparation directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math

import numpy as np

from expand_market_universe import ROOT, read, dump, sha

SIDES = ('BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO')


def epoch(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp())


def close(a, b, label):
    assert math.isclose(a, b, rel_tol=0, abs_tol=2e-6), (label, a, b)


def validate_signal_quote(yes_prices, direction, reported, confidence, config):
    """Stored signal quote includes slippage; entry/edge tests use raw price."""
    raw = np.asarray(yes_prices) if direction == 'YES' else 1-np.asarray(yes_prices)
    adjusted = np.minimum(1., raw*(1+config['slippage_bps']/10000))
    match = np.isclose(adjusted, reported, rtol=0, atol=1e-12)
    allowed = ((raw >= config['min_entry_price']) & (raw <= config['max_entry_price'])
        & (confidence-raw+1e-12 >= config['min_edge']))
    assert (match & allowed).any(), 'No archived raw signal quote supports recorded adjusted quote and entry guardrails.'
    return int((match & allowed).sum())


def load_layout(output):
    """Resolve bound full arrays without treating thin output as source data."""
    protocol = read(output/'protocol.json')
    kind = protocol.get('kind')
    if kind == 'dynamic_universe_fast_v1':
        prepared = Path(protocol['prepared'])
        assert sha(prepared/'protocol.json') == protocol['prepared_protocol_sha256']
        design = read(prepared/'protocol.json')
        assert sha(prepared/'input_bindings.json') == protocol['prepared_input_bindings_sha256']
        assert sha(output/'prepared_array_bindings.json') == protocol['prepared_array_bindings_sha256']
        assert sha(ROOT/'src/run_dynamic_universe_fast.py') == protocol['fast_source_sha256']
        for key in ('count', 'source', 'gates', 'core_config'):
            assert protocol[key] == design[key], ('fast/prepared mismatch', key)
    elif kind == 'dynamic_universe_inventory_sensitivity_v1':
        prepared, design = output, protocol
    else:
        raise ValueError('Not a supported dynamic inventory study protocol.')
    assert design['kind'] == 'dynamic_universe_inventory_sensitivity_v1'
    assert sha(prepared/'input_bindings.json') == design['input_bindings_sha256']
    for name, digest in design['source_files_sha256'].items():
        assert sha(ROOT/name) == digest, ('changed source', name)
    for name, digest in design['frozen_files_sha256'].items():
        assert sha(ROOT/name) == digest, ('changed frozen artifact', name)
    source = Path(design['source'])
    assert sha(source/'cohort.json') == design['source_cohort_sha256']
    cohort = read(source/'cohort.json')[:design['count']]
    assert len(cohort) == design['count']
    ids = {item['market_id'] for item in cohort}
    assert len(ids) == len(cohort)
    return protocol, design, prepared, ids


class DeepAudit:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.protocol, self.design, self.prepared, self.mids = load_layout(self.output)
        self.protocol_sha = sha(self.output/'protocol.json')
        self.manifests, self.tapes = {}, {}
        self.array_bindings = (read(self.output/'prepared_array_bindings.json')
            if self.protocol['kind'] == 'dynamic_universe_fast_v1' else None)

    def market(self, mid):
        if mid not in self.manifests:
            path = self.prepared/'normalized'/mid
            manifest = read(path/'manifest.json')
            assert manifest['market_id'] == mid
            assert sha(path/'data.npz') == manifest['array_sha256']
            if self.array_bindings is not None:
                assert sha(path/'manifest.json') == self.array_bindings[mid]['manifest_sha256']
                assert manifest['array_sha256'] == self.array_bindings[mid]['array_sha256']
            with np.load(path/'data.npz', allow_pickle=False) as arrays:
                self.tapes[mid] = {k: arrays['trade_'+k] for k in ('time', 'price', 'size', 'side')}
            self.manifests[mid] = manifest
        return self.manifests[mid], self.tapes[mid]

    def case(self, gate):
        path = self.output/'results'/str(self.design['count'])/f'both__{gate}'
        summary, config = read(path/'complete.json'), read(path/'config.json')
        assert config['protocol_sha256'] == self.protocol_sha
        assert config['backtest'] == self.design['core_config']
        assert config['gate'] == self.design['gates'][gate]
        assert config['gate_scope'] == 'both' and config['history'] == gate
        if self.array_bindings is not None:
            assert config['prepared_protocol_sha256'] == sha(self.prepared/'protocol.json')
            assert summary['eligible_prints'] == summary['full_tape_count']
            assert summary['admitted_prints'] <= summary['actually_iterated_prints'] <= summary['full_tape_count']
        cfg = config['backtest']; start, end = self.design['start'], self.design['end']
        fills, closed = read(path/'fills.json'), read(path/'positions.json')
        opens, curve = read(path/'open_positions.json'), read(path/'equity.json')
        assert (len(fills), len(closed), len(opens)) == (summary['fills'], summary['closed_positions'], summary['open_positions'])
        assert len({p['market_id'] for p in closed+opens}) == len(closed)+len(opens)
        groups = defaultdict(list); matches, ratios, cash_events = [], [], []
        for i, fill in enumerate(fills):
            mid = fill['market_id']; assert mid in self.mids
            manifest, tape = self.market(mid)
            signal, filled = epoch(fill['signal_time']), epoch(fill['filled_at'])
            activation = manifest['activations'][gate]; assert activation is not None
            assert start <= activation['time'] <= signal <= filled <= end
            assert filled-signal >= cfg['delay_seconds']
            days = (epoch(manifest['market']['close_time'])-signal)/86400
            assert cfg['min_days_to_resolution'] < days < cfg['max_days_to_resolution']
            gate_cfg = config['gate']
            assert activation['time'] % 3600 == 0
            gate_wait = epoch(manifest['market']['close_time'])-activation['time']
            assert 0 < gate_wait < gate_cfg['max_days_to_scheduled_close']*86400
            assert activation['observed_hours'] >= gate_cfg['min_observed_hours']
            volume_key = 'cumulative_notional' if gate_cfg.get('volume_mode') == 'cumulative' else 'recent_notional'
            threshold_key = 'min_cumulative_notional' if volume_key == 'cumulative_notional' else 'min_recent_notional'
            assert activation[volume_key] >= gate_cfg[threshold_key]
            assert max(activation['recent_yes_vwap'], 1-activation['recent_yes_vwap']) >= gate_cfg['min_conviction']
            assert activation['recent_prints'] >= gate_cfg['min_recent_prints']
            assert fill['signal_confidence'] >= cfg['consensus_threshold']
            assert fill['skilled_trader_count'] >= cfg['min_skilled_traders']
            assert fill['signal_concentration'] <= cfg['max_single_trader_weight']+1e-12
            assert fill['directional_trader_count'] >= cfg['min_directional_traders']
            assert fill['effective_directional_traders']+1e-12 >= cfg['min_effective_directional_traders']
            assert fill['directional_concentration'] <= cfg['max_directional_trader_weight']+1e-12
            assert fill['mean_expert_history_markets']+1e-12 >= cfg['min_signal_mean_expert_history_markets']
            close(fill['quantity']*fill['fill_price'], fill['notional'], 'fill cash/quantity')
            close(fill['fee'], fill['notional']*cfg['trade_fee_bps']/10000, 'fill fee')
            assert fill['notional'] <= fill['order_requested_notional']+2e-6
            lo = np.searchsorted(tape['time'], filled, side='left')
            hi = np.searchsorted(tape['time'], filled, side='right')
            price = tape['price'][lo:hi]
            target = price if fill['direction'] == 'YES' else 1-price
            expected = np.clip(target*(1+cfg['slippage_bps']/10000), 0, 1)
            mask = (np.isclose(expected, fill['fill_price'], rtol=0, atol=1e-12)
                & np.isclose(tape['size'][lo:hi], fill['source_trade_size'], rtol=0, atol=1e-12)
                & (tape['side'][lo:hi] == SIDES.index(fill['source_trade_side'])))
            assert mask.any(), ('no supporting source print', mid, fill)
            matches.append(int(mask.sum()))
            at_signal = tape['price'][tape['time'] == signal]
            validate_signal_quote(at_signal, fill['direction'], fill['signal_token_price'],
                fill['signal_confidence'], cfg)
            if fill['source_trade_size'] > 0:
                ratios.append(fill['quantity']/fill['source_trade_size'])
            groups[mid].append(fill)
            cash_events.append((filled, 0, i, -fill['notional']-fill['fee']))
        for i, position in enumerate(closed+opens):
            mid = position['market_id']; manifest, _ = self.market(mid)
            group = groups[mid]
            assert {f['direction'] for f in group} == {position['direction']}
            close(position['quantity'], sum(f['quantity'] for f in group), 'position inventory')
            notional = sum(f['notional'] for f in group)
            close(position['quantity']*position['avg_entry_price'], notional, 'position cost')
            assert epoch(position['opened_at']) == min(epoch(f['filled_at']) for f in group)
            if 'resolved_at' in position:
                assert start <= epoch(position['opened_at']) <= epoch(position['resolved_at']) <= end
                assert position['resolved_at'] == manifest['market']['resolved_at']
                assert manifest['payout_yes'] is not None
                payout = manifest['payout_yes'] if position['direction'] == 'YES' else 1-manifest['payout_yes']
                close(position['payout'], position['quantity']*payout, 'resolved payout')
                close(position['notional'], notional, 'closed cost')
                close(position['pnl'], position['payout']-position['notional'], 'resolved pnl')
                cash_events.append((epoch(position['resolved_at']), 1, i, position['payout']))
        cash = minimum = cfg['initial_balance']
        for _, _, _, amount in sorted(cash_events):
            cash += amount; minimum = min(minimum, cash)
            assert cash >= -2e-6, 'Negative reconstructed cash.'
        close(cash, summary['cash'], 'final cash')
        fee, pnl = sum(f['fee'] for f in fills), sum(p['pnl'] for p in closed)
        close(pnl, summary['gross_closed_pnl'], 'gross realized pnl')
        close(pnl-fee, summary['net_realized_pnl'], 'net realized pnl')
        close(fee, summary['fees'], 'total fees')
        close(summary['total_return'], summary['total_equity']/cfg['initial_balance']-1, 'return')
        assert sum(p['pnl'] < 0 for p in closed) == summary['loss_count']
        stamps = [epoch(e['ts']) for e in curve]
        assert all(start <= t <= end for t in stamps)
        assert all(b >= a for a, b in zip(stamps, stamps[1:]))
        max_error = 0.; peak = cfg['initial_balance']; dd = 0.
        for snapshot, stamp in zip(curve, stamps):
            past = [f for f in fills if epoch(f['filled_at']) <= stamp]
            paid = [p for p in closed if epoch(p['resolved_at']) <= stamp]
            resolved = {p['market_id'] for p in paid}
            actual_cash = cfg['initial_balance']-sum(f['notional']+f['fee'] for f in past)+sum(p['payout'] for p in paid)
            cost = value = 0.
            for fill in past:
                if fill['market_id'] in resolved:
                    continue
                _, tape = self.market(fill['market_id'])
                idx = int(np.searchsorted(tape['time'], stamp, side='right'))-1
                assert idx >= 0
                price = tape['price'][idx] if fill['direction'] == 'YES' else 1-tape['price'][idx]
                cost += fill['notional']; value += fill['quantity']*price
            expected = dict(cash_balance=actual_cash, open_notional=cost,
                open_market_value=value, open_unrealized_pnl=value-cost, total_equity=actual_cash+value)
            for key, amount in expected.items():
                max_error = max(max_error, abs(snapshot[key]-amount)); close(snapshot[key], amount, 'daily '+key)
            peak = max(peak, actual_cash+value); dd = min(dd, (actual_cash+value)/peak-1)
        close(dd, summary['max_drawdown'], 'daily drawdown')
        if curve:
            close(curve[-1]['total_equity'], summary['total_equity'], 'last equity')
        return dict(case=gate, passed=True, fills=len(fills), closed_positions=len(closed),
            open_positions=len(opens), daily_snapshots=len(curve), minimum_reconstructed_cash=minimum,
            max_reconstructed_daily_error=max_error, source_matched_fills=len(matches),
            ambiguous_equal_second_source_matches=sum(n > 1 for n in matches),
            fill_quantity_divided_by_source_print_quantity=dict(
                median=float(np.median(ratios)) if ratios else None,
                p90=float(np.quantile(ratios, .9)) if ratios else None,
                maximum=max(ratios) if ratios else None, fills_above_one=sum(r > 1 for r in ratios)),
            return_fraction=summary['total_return'], drawdown_fraction=summary['max_drawdown'],
            files_sha256={name: sha(path/name) for name in ('config.json', 'complete.json', 'fills.json',
                'positions.json', 'open_positions.json', 'equity.json')})


def audit(output, allow_pending=False):
    checker = DeepAudit(output)
    cases, errors, pending = [], [], []
    for gate in checker.design['gates']:
        path = checker.output/'results'/str(checker.design['count'])/f'both__{gate}'/'complete.json'
        if not path.exists():
            pending.append(gate)
            continue
        try:
            result = checker.case(gate); cases.append(result)
            print(f'{gate}: passed {result["fills"]} fills / {result["daily_snapshots"]} snapshots', flush=True)
        except Exception as error:
            errors.append(dict(case=gate, error=repr(error))); print(errors[-1], flush=True)
    complete = len(cases) == len(checker.design['gates'])
    result = dict(status='failed' if errors else 'passed' if complete else 'partial_pass',
        complete=complete, expected_cases=len(checker.design['gates']), cases=cases, errors=errors,
        pending=pending, unique_markets_with_verified_tapes=len(checker.tapes),
        protocol_sha256=checker.protocol_sha, prepared=str(checker.prepared),
        source_sha256=sha(Path(__file__)), generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Independent persisted cash/inventory/payout/fee accounting, every recorded daily mark and drawdown, recorded activation and signal guardrails, and supporting archived prints/prices at signal and fill timestamps.',
        limitations='Supporting print existence does not establish available depth, executable price, or achievable size. Equal-second duplicate source matches are recorded. Does not re-estimate every expert score or rerun portfolio decisions; does not certify inventory completeness.')
    dump(checker.output/'deep_portfolio_audit.json', result)
    assert not errors and (complete or allow_pending), result['status']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-pending', action='store_true')
    args = parser.parse_args()
    audit(args.output, args.allow_pending)


if __name__ == '__main__':
    main()
