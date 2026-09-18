from datetime import datetime, timedelta
from dataclasses import replace

import numpy as np
import pytest

from backtest.engine import BacktestConfig
from causal_rule_engine import MaturityRule, ObservedFinalityBacktester, prefix_activation, replay_indices
from market_activation import FractionalResolution
from models import Market, Direction, Side, TradeEvent, TimelineEvent, Position
from expand_market_universe import ROOT, read


def config(**changes):
    base=BacktestConfig(**read(ROOT/'config/paper_experiments.json')['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config'])
    return replace(base,**changes)


def tape():
    return np.array([10,3700]),np.array([1_000_000.,1_000_000.]),np.array([.95,.95]),np.array([.95,.95])


def rule(**kw):
    return MaturityRule(min_observed_hours=0,min_recent_prints=1,**kw)


def test_gate_does_not_observe_future_rows_or_partial_hour():
    arrays=tape();r=rule()
    assert prefix_activation(*arrays,rule=r,observed_until=7199) is None
    answer=prefix_activation(*arrays,rule=r,observed_until=7200)
    assert answer['time']==7200 and answer['cumulative_notional']==1_900_000
    extended=[np.r_[a,v] for a,v in zip(arrays,(8000,1e9,.01,.01))]
    assert prefix_activation(*extended,rule=r,observed_until=7200)==answer


def test_terminal_silent_hour_can_satisfy_explicit_warmup_clock():
    arrays=([10],[2_000_000],[.95],[.95])
    r=MaturityRule(min_observed_hours=2,min_recent_prints=1)
    assert prefix_activation(*arrays,rule=r,observed_until=7200) is None
    assert prefix_activation(*arrays,rule=r,observed_until=10800)['time']==10800


def test_deadline_free_gate_ignores_future_schedule_changes():
    arrays=tape();r=rule()
    assert prefix_activation(*arrays,rule=r,observed_until=7200,scheduled_close=0)==prefix_activation(
        *arrays,rule=r,observed_until=7200,scheduled_close=10**12)


def test_schedule_dependent_gate_requires_explicit_schedule():
    assert prefix_activation(*tape(),rule=rule(max_days_to_scheduled_close=28),observed_until=7200) is None


def test_sub_million_rule_rejected():
    with pytest.raises(ValueError,match='requires'):
        rule(min_cumulative_notional=750_000)


def test_expiring_orders_force_full_timeline():
    t=np.array([1,2,3]);m=np.array([0,0,0]);g=np.array([4])
    idx,thin=replay_indices(t,m,g,config(equity_record_interval=0))
    assert idx.tolist()==[2] and thin
    idx,thin=replay_indices(t,m,g,config(equity_record_interval=0,pending_order_expiry_seconds=10))
    assert idx.tolist()==[0,1,2] and not thin


def test_resolution_blocks_future_trades_and_cash_is_not_double_paid():
    start=datetime(2025,1,1)
    market=Market('m','question','unknown',start,start+timedelta(days=20),None,None,0,True)
    engine=ObservedFinalityBacktester({'m':market},[],None,config(),activation_times={'m':start})
    engine.positions['m']=Position('m',Direction.YES,10,.6,start)
    engine.balance=100.
    event=FractionalResolution('m',start+timedelta(days=1),None,.5)
    engine._on_resolution(event)
    assert engine.balance==105 and not engine.positions
    engine._on_resolution(event)
    assert engine.balance==105
    engine._on_trade(TradeEvent('1','m','w',Side.BUY_YES,.9,10,start+timedelta(days=2)))
    assert not engine.pending_orders and engine.post_resolution_prints_skipped==1


def test_policy_horizon_does_not_modify_shared_market_metadata():
    start=datetime(2025,1,1)
    market=Market('m','future revised text','unknown',start,start-timedelta(days=1),None,None,1e9,False)
    cfg=config(signal_mode='favorite',min_ticket_size=1e20)
    source={'m':market}
    engine=ObservedFinalityBacktester(source,[],None,cfg,activation_times={'m':start},
        planning_horizon_days=14,ignore_risk_text=True)
    engine._on_trade(TradeEvent('1','m','w',Side.BUY_YES,.95,10,start))
    assert source['m']==market
    assert engine.markets['m'].close_time==start+timedelta(days=14)
    assert engine.markets['m'].question==''


def test_pre_start_finality_is_known_without_peeking_at_future_finality():
    start=datetime(2025,1,1)
    old=Market('old','old question','unknown',start-timedelta(days=3),None,
        start-timedelta(days=1),Direction.YES,0,False)
    future=replace(old,market_id='future',resolved_at=start+timedelta(days=1))
    engine=ObservedFinalityBacktester({'old':old,'future':future},[],None,config(),
        activation_times={'old':start},observation_start=start)
    assert engine.observed_resolved_markets=={'old'}
    engine._on_trade(TradeEvent('1','old','w',Side.BUY_YES,.9,100,start+timedelta(hours=1)))
    assert not engine.pending_orders and not engine.positions
    assert engine.post_resolution_prints_skipped==1 and engine.balance==10000
