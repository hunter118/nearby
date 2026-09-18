from dataclasses import asdict,replace

import numpy as np
import pytest

from alpha.trader_skill import SkillEstimate
from backtest.engine import BacktestConfig
from build_expanded_replay import utc
from expand_market_universe import ROOT,read,dump
from market_activation import ActivationBacktester
from models import Market,Direction
from run_activation_study import array_timeline
from run_dynamic_universe_fast import assert_safe,retained_indices,thin_timeline,portfolio_equality


class FixedSkills:
    def estimate(self,trader_id,target_market_id,as_of):
        return SkillEstimate(trader_id,target_market_id,as_of,.5,1_000_000.,
            supporting_markets=3,effective_history_markets=3,mean_similarity=.8,
            positive_history_weight_fraction=1.,weighted_score_std=.1)


def fixture():
    t=1735689600
    def market(mid,resolved,direction):
        return Market(mid,f'Will {mid} happen?','unknown',utc(t-86400),utc(t+20*86400),
                      utc(resolved),direction,10_000_000.,True)
    markets={'active':market('active',t+90000,Direction.YES),
             'inactive':market('inactive',t+172900,Direction.NO),
             'fractional':market('fractional',t+3*86400+42,None)}
    manifests=[dict(market_id='active',payout_yes=1.),dict(market_id='inactive',payout_yes=0.),
               dict(market_id='fractional',payout_yes=.5)]
    offsets=np.array([1,10,100,110,200,500,800,86399,86500,90000,92600,172810,173000])
    mids=np.array([1,0,0,0,1,1,0,1,1,0,1,1,1],dtype=np.uint32)
    arrays=dict(time=t+offsets,market=mids,wallet=np.array([0,0,0,1,0,0,2,0,0,1,0,0,0],dtype=np.uint32),
        side=np.zeros(len(offsets),dtype=np.uint8),price=np.full(len(offsets),.9),
        size=np.full(len(offsets),100_000.))
    arrays['price'][9]=.85
    gates=np.array([t+100,np.iinfo(np.int64).max,np.iinfo(np.int64).max])
    config=BacktestConfig(**read(ROOT/'config/paper_experiments.json')['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config'])
    return t,markets,manifests,arrays,['a','b','c'],gates,config


def serialized(result,path):
    for key,value in [('positions',[asdict(p) for p in result['closed_positions']]),
                      ('fills',[asdict(p) for p in result['fills']]),
                      ('open_positions',[asdict(p) for p in result['open_positions']]),
                      ('equity',result['equity_curve'])]:
        dump(path/f'{key}.json',value)


def test_retention_keeps_post_gate_and_daily_last_real_prints():
    _,_,_,arrays,_,gates,_=fixture()
    keep,n=retained_indices(arrays['time'],arrays['market'],gates)
    assert keep.tolist()==[2,3,6,7,9,10,12]
    assert n==4
    assert 5 not in keep  # unrelated post-delay clock print does not fill target order


def test_original_trade_ids_and_all_resolution_ties_preserved():
    _,markets,manifests,arrays,addresses,gates,_=fixture()
    keep,_=retained_indices(arrays['time'],arrays['market'],gates)
    thin=list(thin_timeline(markets,manifests,arrays,addresses,keep))
    full=list(array_timeline(markets,manifests,arrays,addresses))
    assert [e.payload.trade_id for e in thin if e.event_type=='trade']==[str(i) for i in keep]
    assert [e.payload for e in thin if e.event_type=='resolution']==[e.payload for e in full if e.event_type=='resolution']
    tie=[e.event_type for e in thin if e.ts==markets['active'].resolved_at]
    assert tie==['trade','resolution']


@pytest.mark.parametrize('no_active',[False,True])
def test_full_and_thin_four_portfolios_are_byte_identical(tmp_path,no_active):
    t,markets,manifests,arrays,addresses,gates,config=fixture()
    if no_active:
        gates[:]=np.iinfo(np.int64).max
    keep,_=retained_indices(arrays['time'],arrays['market'],gates)
    active={} if no_active else {'active':utc(t+100)}
    full=ActivationBacktester(markets,array_timeline(markets,manifests,arrays,addresses),
        FixedSkills(),config,activation_times=active).run()
    thin=ActivationBacktester(markets,thin_timeline(markets,manifests,arrays,addresses,keep),
        FixedSkills(),config,activation_times=active).run()
    serialized(full,tmp_path/'full');serialized(thin,tmp_path/'thin')
    assert len(portfolio_equality(tmp_path/'thin',tmp_path/'full'))==4
    if no_active:
        assert not full['fills']
    else:
        assert len(full['fills'])==1
        assert full['fills'][0].filled_at==utc(t+800)  # not unrelated t+500 print
        assert full['closed_positions'][0].resolved_at==utc(t+90000)
    assert full['equity_curve']==thin['equity_curve']


def test_empty_tape_retains_resolution_only_days():
    _,markets,manifests,arrays,addresses,gates,_=fixture()
    arrays={k:v[:0] for k,v in arrays.items()}
    keep,n=retained_indices(arrays['time'],arrays['market'],gates)
    assert not len(keep) and n==0
    assert list(thin_timeline(markets,manifests,arrays,addresses,keep))==list(array_timeline(markets,manifests,arrays,addresses))


def test_reject_non_daily_expiry_or_other_scope():
    *_,config=fixture()
    assert_safe(config)
    with pytest.raises(ValueError,match='daily'):
        assert_safe(replace(config,equity_record_interval=1))
    with pytest.raises(ValueError,match='expiry'):
        assert_safe(replace(config,pending_order_expiry_seconds=0))
    with pytest.raises(ValueError,match='BOTH'):
        assert_safe(config,'flow')


def test_unsorted_tape_rejected():
    with pytest.raises(ValueError,match='chronological'):
        retained_indices(np.array([2,1]),np.array([0,0]),np.array([0]))
