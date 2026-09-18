"""New research primitives: prefix maturity and observed-finality portfolio state.

No function in this module infers label time from a tape's final print. Archived
metadata is not upgraded to historical evidence by these chronological checks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
import math

import numpy as np

from market_activation import ActivationBacktester, rolling_sum
from run_dynamic_universe_fast import retained_indices


@dataclass(frozen=True)
class MaturityRule:
    min_cumulative_notional: float = 1_000_000.
    min_conviction: float = .9
    window_hours: int = 24
    min_recent_prints: int = 100
    min_observed_hours: int = 24
    max_days_to_scheduled_close: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.min_cumulative_notional) or self.min_cumulative_notional<1_000_000:
            raise ValueError('The user requires observed cumulative turnover >= $1m.')
        if not math.isfinite(self.min_conviction) or not .5<=self.min_conviction<=1:
            raise ValueError('Invalid price conviction.')
        if self.window_hours<1 or self.min_recent_prints<1 or self.min_observed_hours<0:
            raise ValueError('Invalid observation window.')
        if self.max_days_to_scheduled_close is not None and (
            not math.isfinite(self.max_days_to_scheduled_close) or self.max_days_to_scheduled_close<=0):
            raise ValueError('Invalid optional scheduled horizon.')


def prefix_activation(times, sizes, yes_prices, token_prices, *, rule,
                      observed_until, scheduled_close=None):
    """First qualifying completed hour, including terminal silent hours.

    `observed_until` is the caller's observation clock, not the final trade time.
    A scheduled close is used only by an explicitly schedule-dependent rule.
    Price/volume arrays may contain future rows; those cannot enter any bar.
    """
    times=np.asarray(times,dtype=np.int64)
    if times.ndim!=1 or np.any(np.diff(times)<0):
        raise ValueError('Expected chronological timestamps.')
    values=[np.asarray(a,dtype=np.float64) for a in (sizes,yes_prices,token_prices)]
    if any(a.shape!=times.shape for a in values):
        raise ValueError('Unaligned trade arrays.')
    if isinstance(observed_until,bool) or not isinstance(observed_until,(int,np.integer)):
        raise ValueError('An explicit integer observation clock is required.')
    if not len(times):return None
    completed_hour=int(observed_until)//3600
    stop=int(np.searchsorted(times,completed_hour*3600,side='left'))
    if not stop:return None
    ts=times[:stop];size,yes,token=(a[:stop] for a in values)
    if (not all(np.isfinite(a).all() for a in (size,yes,token)) or np.any(size<=0)
            or np.any((yes<0)|(yes>1)) or np.any((token<0)|(token>1))):
        raise ValueError('Invalid observed price or quantity.')
    first_hour=int(ts[0]//3600);n=completed_hour-first_hour
    bins=ts//3600-first_hour
    dollar=np.bincount(bins,weights=size*token,minlength=n)
    cumulative=np.cumsum(dollar)
    quantities=rolling_sum(np.bincount(bins,weights=size,minlength=n),rule.window_hours)
    pq=rolling_sum(np.bincount(bins,weights=size*yes,minlength=n),rule.window_hours)
    prints=rolling_sum(np.bincount(bins,minlength=n),rule.window_hours)
    vwap=np.divide(pq,quantities,out=np.full(n,.5),where=quantities>0)
    clocks=(first_hour+np.arange(n)+1)*3600
    eligible=((clocks-int(ts[0])>=rule.min_observed_hours*3600)
        &(cumulative>=rule.min_cumulative_notional)
        &(prints>=rule.min_recent_prints)
        &(np.maximum(vwap,1-vwap)>=rule.min_conviction))
    if rule.max_days_to_scheduled_close is not None:
        if scheduled_close is None:return None
        eligible&=(scheduled_close-clocks>0)&(scheduled_close-clocks<rule.max_days_to_scheduled_close*86400)
    matches=np.flatnonzero(eligible)
    if not len(matches):return None
    i=int(matches[0])
    return dict(time=int(clocks[i]),cumulative_notional=float(cumulative[i]),
        recent_notional=float(rolling_sum(dollar,rule.window_hours)[i]),
        recent_prints=int(prints[i]),recent_yes_vwap=float(vwap[i]),
        observed_hours=float((clocks[i]-int(ts[0]))/3600),
        deadline_used=rule.max_days_to_scheduled_close is not None)


def replay_indices(times, market_index, gate_times, config):
    """Use the proven no-effect shortcut only under its required clock guards."""
    if config.pending_order_expiry_seconds is not None or config.equity_record_interval>0:
        return np.arange(len(times),dtype=np.int64),False
    selected,_=retained_indices(times,market_index,gate_times)
    return selected,True


class ObservedFinalityBacktester(ActivationBacktester):
    """Cancel at observed finality and never open again in that condition.

    Optional fixed planning horizon is a policy constant for allocation, not a
    forecast or claim about realized settlement. In this branch current Gamma
    endDate is not used by entry/sizing. A uniform-score branch can additionally
    blank text for risk grouping, avoiding revised-text dependence there.
    """
    def __init__(self,*args,planning_horizon_days=None,ignore_risk_text=False,
                 observation_start=None,**kwargs):
        if planning_horizon_days is not None and (
            not math.isfinite(planning_horizon_days) or planning_horizon_days<=0):
            raise ValueError('Planning horizon must be a finite positive constant.')
        if args:
            args=(dict(args[0]),*args[1:])
        elif 'markets' in kwargs:
            kwargs['markets']=dict(kwargs['markets'])
        super().__init__(*args,**kwargs)
        self.planning_horizon_days=planning_horizon_days
        self.ignore_risk_text=ignore_risk_text
        # Fresh cash does not mean previously resolved conditions become open
        # again. This set uses only labels available BEFORE the replay clock.
        self.observed_resolved_markets={mid for mid,market in self.markets.items()
            if observation_start is not None and market.resolved_at is not None
            and market.resolved_at<observation_start}
        self.post_resolution_prints_skipped=0

    def _on_resolution(self,resolution):
        self.observed_resolved_markets.add(resolution.market_id)
        super()._on_resolution(resolution)

    def _on_trade(self,trade):
        if trade.market_id in self.observed_resolved_markets:
            self.processed+=1
            self.post_resolution_prints_skipped+=1
            return
        if self.planning_horizon_days is not None or self.ignore_risk_text:
            market=self.markets[trade.market_id]
            changes={}
            if self.planning_horizon_days is not None:
                changes['close_time']=trade.timestamp+timedelta(days=self.planning_horizon_days)
            if self.ignore_risk_text:
                changes.update(question='',category='unknown')
            self.markets[trade.market_id]=replace(market,**changes)
        super()._on_trade(trade)
