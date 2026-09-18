"""Independent saved-account/source-print audit; never runs a strategy engine."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from audit_dynamic_universe_portfolios import validate_signal_quote
from backtest.engine import BacktestConfig
from expand_market_universe import ROOT, read, dump, sha

SIDES = ('BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO')


def epoch(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def close(left, right, label):
    if not math.isclose(left, right, rel_tol=0, abs_tol=2e-6):
        raise AssertionError((label, left, right))


def cash_path(initial, fills, closed):
    # New runner publishes finality BEFORE any trade at the same second.
    events = [(epoch(p['resolved_at']), 0, p['market_id'], i, p['payout'])
              for i,p in enumerate(closed)]
    events += [(epoch(f['filled_at']), 1, f['market_id'], i, -f['notional']-f['fee'])
               for i,f in enumerate(fills)]
    cash = minimum = initial
    for *_, amount in sorted(events):
        cash += amount; minimum = min(minimum, cash)
        assert cash >= -2e-6, 'Negative reconstructed cash.'
    return cash, minimum


def validate_execution_limits(fill, cfg):
    """Independently check saved child limits, without invoking the engine."""
    assert fill['quantity'] > 0 and fill['source_trade_size'] > 0
    participation = cfg.get('max_fill_participation')
    if participation is not None:
        limit = participation*fill['source_trade_size']
        assert fill['quantity'] <= limit+1e-9*max(1., limit), 'Child exceeds print participation.'
    trade_filter = cfg.get('execution_trade_filter', 'any')
    if trade_filter in ('target_token', 'target_token_buy', 'target_token_sell'):
        assert fill['source_trade_side'].endswith(fill['direction']), 'Wrong execution token.'
    if trade_filter in ('target_token_buy', 'target_token_sell'):
        action = 'BUY' if trade_filter=='target_token_buy' else 'SELL'
        assert fill['source_trade_side'] == action+'_'+fill['direction'], 'Wrong source taker side.'
    expiry = cfg.get('pending_order_expiry_seconds')
    if expiry is not None:
        deadline = epoch(fill['signal_time'])+cfg['delay_seconds']+expiry
        assert epoch(fill['filled_at']) <= deadline, 'Fill after parent expiry.'
    deterioration = cfg.get('max_price_deterioration_bps')
    if deterioration is not None:
        maximum = fill['signal_token_price']*(1+deterioration/10000.)
        assert fill['fill_price'] <= maximum+1e-12, 'Fill exceeds relative price cap.'


def validate_parent_limits(group, cfg):
    if cfg.get('execution_one_order_per_market'):
        assert len({f['signal_time'] for f in group}) == 1, 'More than one parent signal.'
        requested = group[0]['order_requested_notional']
        assert all(f['order_requested_notional']==requested for f in group), 'Parent budget changed.'
        assert sum(f['notional'] for f in group) <= requested+2e-6, 'Children exceed parent budget.'
        maximum = cfg['execution_max_child_fills'] if cfg['execution_partial_fill'] else cfg['execution_slices']
        assert len(group) <= maximum, 'Too many children.'
        assert [f['child_fill_index'] for f in group] == list(range(1,len(group)+1)), 'Child sequence is not contiguous.'


class AccountAudit:
    def __init__(self, stage):
        self.stage = Path(stage).resolve()
        self.protocol = read(self.stage/'protocol.json')
        assert self.protocol['kind'] in ('causal_rule_search_v1', 'causal_rule_held_position_fast_v1',
                                        'causal_expiry_barrier_fast_v1')
        self.protocol_sha = sha(self.stage/'protocol.json')
        assert sha(self.stage/'input_bindings.json') == self.protocol['input_bindings_sha256']
        self.binding = read(self.stage/'input_bindings.json')
        self.history = Path(self.protocol['history_root'])
        assert self.binding['history_root'] == str(self.history)
        assert self.binding['market_order'] == sorted(set(self.binding['market_order']))
        assert len(self.binding['market_order']) == self.binding['count'] == self.protocol['count']
        assert sha(self.history/'complete.json') == self.binding['historical_completion_sha256']
        source = Path(self.protocol['source'])
        assert sha(source/'cohort.json') == self.binding['cohort_sha256']
        for group in ('source_files_sha256', 'frozen_files_sha256'):
            for filename, digest in self.protocol[group].items():
                assert sha(ROOT/filename) == digest, ('changed code/frozen input', filename)
        self.cache = {}

    def market(self, mid):
        if mid not in self.cache:
            folder = self.history/'normalized'/mid
            manifest = read(folder/'manifest.json')
            record = self.binding['records'][mid]
            assert sha(folder/'manifest.json') == record['manifest_sha256']
            assert sha(folder/'data.npz') == manifest['array_sha256'] == record['array_sha256']
            assert manifest['market_id'] == mid
            with np.load(folder/'data.npz', allow_pickle=False) as arrays:
                tape = {k:arrays['trade_'+k] for k in ('time', 'price', 'token_price', 'size', 'side')}
            assert np.all(np.diff(tape['time']) >= 0)
            self.cache[mid] = manifest, tape
        return self.cache[mid]

    def case(self, name):
        folder = self.stage/'results'/name
        summary, config = read(folder/'complete.json'), read(folder/'config.json')
        cfg = config['backtest']; case = self.protocol['cases'][name]
        assert config['protocol_sha256'] == self.protocol_sha and config['case'] == case
        assert cfg == asdict(BacktestConfig(**(self.protocol['base_config'] | case.get('overrides', {}))))
        assert cfg['equity_record_interval'] <= 0, 'Audit expects complete daily marks.'
        fills, closed, opens, curve = (read(folder/f'{f}.json') for f in
                                      ('fills', 'positions', 'open_positions', 'equity'))
        activation = read(folder/'activations.json')
        assert len(activation) == summary['activated_markets']
        assert (len(fills), len(closed), len(opens)) == (summary['fills'], summary['closed_positions'], summary['open_positions'])
        assert len({p['market_id'] for p in closed+opens}) == len(closed)+len(opens)
        start, end = self.protocol['start'], self.protocol['end']
        groups = defaultdict(list); matches = []; ratios = []; opposite = []; tiers = Counter()
        for i, fill in enumerate(fills):
            validate_execution_limits(fill, cfg)
            mid = fill['market_id']; manifest, tape = self.market(mid)
            signal, filled = epoch(fill['signal_time']), epoch(fill['filled_at'])
            gate = activation[mid]
            assert start <= gate['time'] <= signal <= filled <= end
            assert filled-signal >= cfg['delay_seconds'] >= 300
            available = manifest['label_available_at'] if manifest['label_usable'] else None
            assert available is None or filled < available, 'Fill at/after already-known finality.'
            tiers[manifest['evidence_tier']] += 1
            lo, hi = (np.searchsorted(tape['time'], filled, side=s) for s in ('left', 'right'))
            yes = tape['price'][lo:hi]; sides = tape['side'][lo:hi]
            actual_source = tape['token_price'][lo:hi]
            source_from_yes = np.where((sides==0)|(sides==2), yes, 1-yes)
            assert np.allclose(actual_source, source_from_yes, rtol=0, atol=1e-12)
            target = yes if fill['direction']=='YES' else 1-yes
            adjusted = np.clip(target*(1+cfg['slippage_bps']/10000), 0., 1.)
            mask = (np.isclose(adjusted, fill['fill_price'], rtol=0, atol=1e-12)
                    & np.isclose(tape['size'][lo:hi], fill['source_trade_size'], rtol=0, atol=1e-12)
                    & (sides == SIDES.index(fill['source_trade_side'])))
            assert mask.any(), ('No exact timestamp/side/size/price source support', name, i)
            matches.append(int(mask.sum()))
            same_token = fill['source_trade_side'].endswith(fill['direction'])
            if not same_token: opposite.append(i)
            if cfg['execution_trade_filter'] in ('target_token', 'target_token_buy', 'target_token_sell'):
                assert same_token
            at_signal = tape['price'][tape['time']==signal]
            validate_signal_quote(at_signal, fill['direction'], fill['signal_token_price'], fill['signal_confidence'], cfg)
            assert fill['signal_confidence'] >= cfg['consensus_threshold']
            assert fill['skilled_trader_count'] >= cfg['min_skilled_traders']
            assert fill['directional_trader_count'] >= cfg['min_directional_traders']
            assert fill['effective_directional_traders']+1e-12 >= cfg['min_effective_directional_traders']
            assert fill['signal_concentration'] <= cfg['max_single_trader_weight']+1e-12
            assert fill['directional_concentration'] <= cfg['max_directional_trader_weight']+1e-12
            assert fill['mean_expert_history_markets']+1e-12 >= cfg['min_signal_mean_expert_history_markets']
            assert cfg['min_entry_price'] <= fill['fill_price'] <= cfg['max_entry_price']
            close(fill['notional'], fill['quantity']*fill['fill_price'], 'fill notional')
            close(fill['fee'], fill['notional']*cfg['trade_fee_bps']/10000, 'fill fee')
            assert fill['notional'] <= fill['order_requested_notional']+2e-6
            if fill['source_trade_size'] > 0: ratios.append(fill['quantity']/fill['source_trade_size'])
            groups[mid].append(fill)
        for position in closed+opens:
            mid = position['market_id']; manifest, _ = self.market(mid); group = groups[mid]
            validate_parent_limits(group, cfg)
            assert {f['direction'] for f in group} == {position['direction']}
            close(position['quantity'], sum(f['quantity'] for f in group), 'position quantity')
            cost = sum(f['notional'] for f in group)
            close(position['quantity']*position['avg_entry_price'], cost, 'position cost')
            assert epoch(position['opened_at']) == min(epoch(f['filled_at']) for f in group)
            if 'resolved_at' in position:
                assert manifest['label_usable'] and manifest['payout_yes'] is not None
                assert epoch(position['resolved_at']) == manifest['label_available_at'] <= end
                rate = manifest['payout_yes'] if position['direction']=='YES' else 1-manifest['payout_yes']
                close(position['payout'], position['quantity']*rate, 'resolved payout')
                close(position['notional'], cost, 'closed notional')
                close(position['pnl'], position['payout']-cost, 'closed pnl')
            else:
                assert not manifest['label_usable'] or manifest['label_available_at'] > end
        cash, minimum = cash_path(cfg['initial_balance'], fills, closed)
        close(cash, summary['cash'], 'final cash')
        fees = sum(f['fee'] for f in fills); pnl = sum(p['pnl'] for p in closed)
        close(fees, summary['fees'], 'fees'); close(pnl, summary['gross_closed_pnl'], 'realized pnl')
        close(pnl-fees, summary['net_realized_pnl'], 'net realized pnl')
        assert sum(p['pnl']<0 for p in closed) == summary['loss_count']
        assert {f['market_id'] for f in fills} == {p['market_id'] for p in closed+opens}
        peak = cfg['initial_balance']; drawdown = 0.; max_error = 0.; last = start
        for point in curve:
            stamp = epoch(point['ts']); assert last <= stamp <= end; last = stamp
            past = [f for f in fills if epoch(f['filled_at']) <= stamp]
            paid = [p for p in closed if epoch(p['resolved_at']) <= stamp]
            resolved = {p['market_id'] for p in paid}
            daily_cash = cfg['initial_balance']-sum(f['notional']+f['fee'] for f in past)+sum(p['payout'] for p in paid)
            cost = value = 0.
            for fill in past:
                if fill['market_id'] in resolved: continue
                _, tape = self.market(fill['market_id'])
                index = int(np.searchsorted(tape['time'], stamp, side='right'))-1
                assert index >= 0
                target = tape['price'][index] if fill['direction']=='YES' else 1-tape['price'][index]
                value += fill['quantity']*target; cost += fill['notional']
            expected = dict(cash_balance=daily_cash, open_notional=cost, open_market_value=value,
                            open_unrealized_pnl=value-cost, total_equity=daily_cash+value)
            for key, amount in expected.items():
                max_error = max(max_error, abs(point[key]-amount)); close(point[key], amount, 'daily '+key)
            peak = max(peak, daily_cash+value); drawdown = min(drawdown, (daily_cash+value)/peak-1)
        assert curve
        for field in ('open_notional', 'open_market_value', 'total_equity'):
            close(curve[-1][field], summary[field], 'final '+field)
        close(drawdown, summary['max_drawdown'], 'daily drawdown')
        close(summary['total_return'], summary['total_equity']/cfg['initial_balance']-1, 'total return')
        return dict(case=name, status='passed', fills=len(fills), closed=len(closed), open=len(opens),
            daily_snapshots=len(curve), minimum_cash=minimum, maximum_daily_error=max_error,
            total_return=summary['total_return'], max_drawdown=summary['max_drawdown'],
            execution_limit_checks='Child quantity/token/side, parent time/price bounds and single-parent budget/child count verified when configured; not book-depth proof.',
            exact_source_matched_fills=len(matches), ambiguous_equal_second_matches=sum(n>1 for n in matches),
            opposite_token_source_fills=len(opposite), opposite_token_fill_indices=opposite,
            fill_evidence_tiers=dict(tiers), quantity_to_source_print_ratio={
                'median':float(np.median(ratios)) if ratios else None,
                'maximum':max(ratios) if ratios else None, 'above_one':sum(r>1 for r in ratios)},
            files_sha256={f:sha(folder/f) for f in ('complete.json','config.json','fills.json',
                'positions.json','open_positions.json','equity.json','activations.json')})


def audit(stage, cases, output):
    output = Path(output).resolve(); stage = Path(stage).resolve()
    if not output.is_relative_to(ROOT/'artifacts') or output==stage or stage in output.parents:
        raise ValueError('Use a separate artifact output, outside the bound stage.')
    checker = AccountAudit(stage); records = []; errors = []
    for name in cases:
        try:
            result = checker.case(name); records.append(result)
            print(f'{name}: passed {result["fills"]} fills / {result["daily_snapshots"]} marks; opposite-token source {result["opposite_token_source_fills"]}', flush=True)
        except Exception as error:
            errors.append(dict(case=name, error=repr(error))); print(errors[-1], flush=True)
    report = dict(status='failed' if errors else 'passed', requested=cases, cases=records, errors=errors,
        stage=str(stage), stage_protocol_sha256=checker.protocol_sha, history_root=str(checker.history),
        checked_market_tapes=len(checker.cache), source_sha256=sha(Path(__file__)),
        generated_at=datetime.now(timezone.utc).isoformat(),
        limits='Independent persisted account and source-print consistency only; no score re-estimation. Metadata-proxy finality remains proxy. Opposite-token prints imply complementary target quotes, not observed target-token fills. Quantity-unconstrained source-print prices do not prove book depth, liquidity or executable order size.')
    dump(output/'audit.json', report)
    assert not errors, report
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True); parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.stage,args.cases,args.output)
