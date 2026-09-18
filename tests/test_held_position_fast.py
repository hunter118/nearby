"""Byte-exact synthetic portfolio certificate; no downloaded inputs needed."""
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig
from causal_rule_engine import ObservedFinalityBacktester
from held_position_fast import HeldPositionFastBacktester
from market_activation import FractionalResolution
from models import Direction, Market, ResolutionEvent, Side, TimelineEvent, TradeEvent, TraderMarketSettlement
from run_similarity_kernel_study import Kernel, KernelSkillEstimator


ROOT = Path(__file__).resolve().parents[1]
MODES = ('favorite', 'uniform', 'semantic', 'kernel')
EXECUTIONS = ('single', 'slices', 'partial')


def encode(value):
    def default(x):
        if is_dataclass(x): return asdict(x)
        if isinstance(x, datetime): return x.isoformat()
        raise TypeError(type(x))
    return json.dumps(value, default=default, sort_keys=True, separators=(',', ':')).encode()


def synthetic_case(mode='semantic', execution='single', planning=False, interval=0):
    start = datetime(2025, 1, 1)
    markets = {mid: Market(mid, 'Will a team win the final?', 'sports', start-timedelta(days=30),
                         start+timedelta(days=20), None, None, 0., True)
               for mid in ['h0', 'h1', 'h2', 'h3', 'a', 'b', 'open', 'inactive']}
    history = [TraderMarketSettlement(f'w{w}', f'h{h}', .1*(1+h), 1000.+w,
               start-timedelta(days=10-h) if h<3 else start+timedelta(hours=8))
               for w in range(4) for h in range(4)]
    events = []
    for i in range(300):
        stamp = start+timedelta(minutes=30*i)
        for j, mid in enumerate(('a', 'b', 'open', 'inactive')):
            # Held markets continue supplying marks and opposite-direction
            # observations; other conditions share exactly the same wallets.
            direction = Side.BUY_NO if mid=='b' else Side.BUY_YES
            side = direction if i%7 else (Side.SELL_YES if mid=='b' else Side.SELL_NO)
            if i>=8 and i%4!=0:
                side = Side.BUY_YES if mid=='b' else Side.BUY_NO
            price = (.1 if mid=='b' else .9) + (.02 if i%3 else -.015)
            size = 0. if i==17 and mid=='a' else 25.+i%5
            trade = TradeEvent(f'{i}-{j}', mid, f'w{i%4}', side, price, size, stamp)
            events.append(TimelineEvent('trade', stamp, trade))
    # Finality sorts before same-second trades, followed by repeated finality.
    for mid, day, fraction in (('a', 2, None), ('a', 3, None), ('b', 4, .5)):
        stamp = start+timedelta(days=day)
        resolution = (ResolutionEvent(mid, stamp, Direction.YES) if fraction is None
                      else FractionalResolution(mid, stamp, None, fraction))
        events.append(TimelineEvent('resolution', stamp, resolution))
    events.sort(key=lambda e:(e.ts, 0 if e.event_type=='resolution' else 1))
    config_path = ROOT/'config/paper_experiments.json'
    base = json.loads(config_path.read_text())['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config']
    cfg = replace(BacktestConfig(**base), signal_mode='favorite' if mode=='favorite' else 'expert_flow',
        delay_seconds=60, min_skilled_traders=2, min_directional_traders=2,
        min_effective_directional_traders=1., max_single_trader_weight=1.,
        max_directional_trader_weight=1., min_signal_mean_expert_history_markets=0.,
        min_days_to_resolution=0, min_entry_price=.5, min_user_volume=0.,
        equity_record_interval=interval, min_ticket_size=1., trade_fee_bps=7.,
        pending_order_expiry_seconds=3*3600, flow_lookback_seconds=4*3600,
        execution_reserve_parent_cash=True, execution_wait_for_price=True,
        execution_trade_filter='target_token', execution_confirmation_trades=2,
        execution_slices=3 if execution=='slices' else 1,
        execution_partial_fill=execution=='partial', execution_max_child_fills=8,
        max_fill_participation=.2 if execution=='partial' else None,
        semantic_cluster_similarity_threshold=.7 if mode in ('semantic', 'kernel') else None,
        max_semantic_cluster_exposure_fraction=.5)
    vectors = np.random.default_rng(118).uniform(.3, 1., size=(len(markets), 12)).astype(np.float32)
    kwargs = dict(activation_times={m:start for m in ('a', 'b', 'open')},
                  planning_horizon_days=14 if planning else None,
                  ignore_risk_text=planning, observation_start=start)
    return markets, events, history, vectors, cfg, kwargs


def replay_pair(mode, execution, planning, interval):
    markets, events, history, vectors, cfg, kwargs = synthetic_case(mode, execution, planning, interval)
    engines = []
    results = []
    for cls in (ObservedFinalityBacktester, HeldPositionFastBacktester):
        estimator = None
        if mode!='favorite':
            options = dict(similarity_mode='uniform' if mode=='uniform' else 'semantic',
                           precomputed_market_vectors=vectors)
            estimator = (KernelSkillEstimator(markets, history, kernel=Kernel(.5, 2), **options)
                         if mode=='kernel' else TraderSkillEstimator(markets, history, **options))
            # Force different cache-clear schedules; cached/recomputed outputs
            # must remain identical even when skipped calls would fill a cache.
            estimator._estimate_cache_max_entries = 3
        engine = cls(markets, events, estimator, replace(cfg), **kwargs)
        results.append(engine.run()); engines.append(engine)
    base, fast = engines
    assert encode(results[0]) == encode(results[1])
    assert base.processed == fast.processed
    assert base.admitted == fast.admitted
    assert base.post_resolution_prints_skipped == fast.post_resolution_prints_skipped
    assert base.market_observed_notional == fast.market_observed_notional
    assert base.last_yes_price == fast.last_yes_price
    assert base.markets == fast.markets
    assert encode(base.pending_orders) == encode(fast.pending_orders)
    assert fast.held_position_signal_updates_skipped > 0
    assert len(results[0]['fills']) >= 3
    assert results[0]['closed_positions'] and results[0]['open_positions']
    if execution!='single':
        assert any(f.child_fill_index>1 for f in results[0]['fills'])
    return dict(mode=mode, execution=execution, planning=planning, equity_interval=interval,
                portfolio_sha256=hashlib.sha256(encode(results[0])).hexdigest(),
                exact=True, fills=len(results[0]['fills']),
                closed=len(results[0]['closed_positions']), open=len(results[0]['open_positions']),
                processed=fast.processed, admitted=fast.admitted,
                held_signal_updates_skipped=fast.held_position_signal_updates_skipped)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('execution', EXECUTIONS)
@pytest.mark.parametrize('planning,interval', [(False, 0), (True, 7)])
def test_byte_exact_portfolios(mode, execution, planning, interval):
    replay_pair(mode, execution, planning, interval)


def test_recheck_and_unrecognized_stateful_estimator_rejected():
    markets, events, history, vectors, cfg, kwargs = synthetic_case()
    with pytest.raises(ValueError, match='recheck'):
        HeldPositionFastBacktester(markets, events, None, replace(cfg, execution_recheck_signal=True), **kwargs)
    with pytest.raises(TypeError, match='deterministic'):
        HeldPositionFastBacktester(markets, events, object(), cfg, **kwargs)
    estimator = TraderSkillEstimator(markets, history, similarity_mode='uniform')
    estimator.estimate = lambda *args, **kwargs: None
    with pytest.raises(TypeError, match='overrides'):
        HeldPositionFastBacktester(markets, events, estimator, cfg, **kwargs)


def test_runtime_recheck_change_rejected_before_due_orders():
    markets, events, _, _, cfg, kwargs = synthetic_case(mode='favorite')
    engine = HeldPositionFastBacktester(markets, events, None, cfg, **kwargs)
    engine.config.execution_recheck_signal = True
    with pytest.raises(ValueError, match='recheck'):
        engine.run()
    assert not engine.fills


def write_synthetic_certificate(destination):
    """Explicit offline proof command; not run automatically by unit tests."""
    records = [replay_pair(mode, execution, planning, interval)
               for mode in MODES for execution in EXECUTIONS
               for planning, interval in [(False, 0), (True, 7)]]
    files = ['src/held_position_fast.py', 'src/causal_rule_engine.py', 'src/market_activation.py',
             'src/backtest/engine.py', 'src/alpha/trader_skill.py', 'src/alpha/signal.py',
             'src/run_similarity_kernel_study.py', 'src/models.py',
             'config/paper_experiments.json', 'tests/test_held_position_fast.py']
    document = dict(status='passed', complete=True, byte_exact=True, cases=records,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    bindings={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                    scope='24 synthetic full-result byte comparisons; known offline estimators, hold-to-resolution lifecycle, no execution signal recheck. Not an empirical replay certificate.',
                    intentional_state_difference='Held-market flow accumulators and estimator cache contents may differ; none feed future orders under the required lifecycle.')
    path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2)+'\n')
    return document
