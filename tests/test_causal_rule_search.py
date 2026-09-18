from dataclasses import asdict,replace

import numpy as np
import pytest

from build_expanded_replay import utc
from causal_rule_engine import MaturityRule
from models import Market
from run_causal_rule_search import validate_cases,timeline,load_inputs,history_rows
from causal_rule_engine import ObservedFinalityBacktester,replay_indices
from test_dynamic_universe_fast import fixture,FixedSkills,serialized
from run_dynamic_universe_fast import portfolio_equality
from expand_market_universe import dump
from prepare_causal_expert_history import VARIANTS


def case(**changes):
    c=dict(maturity=asdict(MaturityRule(max_days_to_scheduled_close=28)),
        history_variant='tail1000',similarity='semantic',overrides={})
    c.update(changes);return c


def test_default_and_deadline_free_cases_validate():
    free=case(maturity=asdict(MaturityRule()),similarity='uniform',
        planning_horizon_days=14,ignore_risk_text=True)
    assert validate_cases({'semantic':case(),'no_text':free})['initial_balance']==10000


@pytest.mark.parametrize('change',[
    dict(overrides={'initial_balance':100000}),dict(overrides={'target_exposure_fraction':2}),
    dict(overrides={'slippage_bps':-5}),dict(overrides={'delay_seconds':0}),
    dict(history_variant='future_tail'),dict(similarity='trained'),
    dict(ignore_risk_text=True),dict(planning_horizon_days=14),
    dict(maturity=asdict(MaturityRule()))])
def test_invalid_or_mislabeled_cases_rejected(change):
    with pytest.raises(ValueError):validate_cases({'bad':case(**change)})


def test_known_resolution_precedes_same_second_new_trade():
    t=1735689600
    markets={'m':Market('m','q','unknown',utc(t-100),utc(t+100000),utc(t+10),None,0,True)}
    manifests=[dict(market_id='m',label_usable=True,payout_yes=.5)]
    arrays=dict(time=np.array([t+9,t+10,t+11]),market=np.array([0,0,0]),
        wallet=np.array([0,0,0]),side=np.array([0,0,0]),price=np.array([.9,.9,.9]),size=np.array([1.,1.,1.]))
    rows=list(timeline(markets,manifests,arrays,['wallet'],np.array([0,1,2])))
    assert [(r.ts,r.event_type) for r in rows if r.ts==utc(t+10)]==[(utc(t+10),'resolution'),(utc(t+10),'trade')]
    assert [r.payload.trade_id for r in rows if r.event_type=='trade']==['0','1','2']


def test_empty_trade_tape_still_contains_known_payout():
    t=1735689600
    markets={'m':Market('m','q','unknown',utc(t-100),None,utc(t),None,0,True)}
    rows=list(timeline(markets,[dict(market_id='m',label_usable=True,payout_yes=.5)],{},[],np.array([],dtype=int)))
    assert len(rows)==1 and rows[0].event_type=='resolution'


@pytest.mark.parametrize('deadline_free',[False,True])
@pytest.mark.parametrize('partial_fill',[False,True])
@pytest.mark.parametrize('no_active',[False,True])
def test_new_finality_engine_full_and_thin_are_byte_identical(tmp_path,deadline_free,partial_fill,no_active):
    t,markets,manifests,arrays,addresses,gates,cfg=fixture()
    manifests=[m|{'label_usable':True} for m in manifests]
    # Include another active-market print strictly after its known resolution.
    arrays['market'][10]=0
    if no_active:gates[:]=np.iinfo(np.int64).max
    active={} if no_active else {'active':utc(t+100)}
    if partial_fill:
        cfg=replace(cfg,max_fill_participation=.001,execution_partial_fill=True,
            execution_max_child_fills=10,execution_trade_filter='target_token',
            execution_reserve_parent_cash=True)
    opts=dict(activation_times=active,planning_horizon_days=14 if deadline_free else None,
        ignore_risk_text=deadline_free)
    keep,thinned=replay_indices(arrays['time'],arrays['market'],gates,cfg)
    assert thinned
    full_engine=ObservedFinalityBacktester(markets,timeline(markets,manifests,arrays,addresses,np.arange(len(arrays['time']))),
        FixedSkills(),cfg,**opts)
    thin_engine=ObservedFinalityBacktester(markets,timeline(markets,manifests,arrays,addresses,keep),
        FixedSkills(),cfg,**opts)
    full,thin=full_engine.run(),thin_engine.run()
    serialized(full,tmp_path/'full');serialized(thin,tmp_path/'thin')
    assert len(portfolio_equality(tmp_path/'thin',tmp_path/'full'))==4
    assert all(f.filled_at<markets[f.market_id].resolved_at for f in full['fills'])
    if not no_active:
        assert full['fills']
        assert thin_engine.post_resolution_prints_skipped==full_engine.post_resolution_prints_skipped==3


def test_loader_remaps_by_condition_and_discards_future_last_print_resolution(tmp_path):
    source,history=tmp_path/'source',tmp_path/'history'
    source.mkdir();t=1735689600
    # Deliberately opposite archive rank / vector order to required lexical ties.
    dump(source/'cohort.json',[{'market_id':'z','rank':1},{'market_id':'a','rank':2}])
    np.savez(source/'vectors.npz',market_ids=np.array(['z','a']),vectors=np.array([[1.,0.],[0.,1.]]))
    for mid,known in [('z',True),('a',False)]:
        folder=history/'normalized'/mid;folder.mkdir(parents=True)
        old=dict(market_id=mid,question='test',category='unknown',
            created_at=str(utc(t-86400)),close_time=str(utc(t+86400*20)),
            resolved_at=str(utc(t+999999)),resolution='YES',volume=1e10,active=False)
        dump(folder/'manifest.json',dict(market_id=mid,binding={'provenance':{'source_market':old}},
            label_usable=known,label_available_at=t+86401 if known else None,
            raw_resolution_boundary=t+86400 if known else None,payout_yes=0 if known else None))
        fields=dict(wallets=np.array(['same_wallet']),trade_time=np.array([t+1,t+3601]),
            trade_wallet=np.array([0,0],dtype=np.uint32),trade_side=np.array([0,0],dtype=np.uint8),
            trade_price=np.array([.95,.95]),trade_token_price=np.array([.95,.95]),
            trade_size=np.array([1e6,1e6]))
        for variant in VARIANTS:
            fields.update({f'{variant}_history_wallet':np.array([0],dtype=np.uint32),
                f'{variant}_history_score':np.array([-.9]),f'{variant}_history_notional':np.array([1e6])})
        np.savez(folder/'data.npz',**fields)
    spec=case(maturity=asdict(MaturityRule(min_recent_prints=1,min_observed_hours=0,
        max_days_to_scheduled_close=28)))
    markets,manifests,arrays,addresses,remaps,vectors,gates=load_inputs(source,history,{'x':spec})
    assert list(markets)==['a','z'] and addresses==['same_wallet']
    assert vectors.tolist()==[[0.,1.],[1.,0.]]
    assert arrays['market'].tolist()==[0,1,0,1]
    assert markets['a'].resolved_at is None and markets['a'].resolution is None
    assert markets['z'].resolved_at==utc(t+86401) and markets['z'].resolution.value=='NO'
    rows=history_rows(history,'tail1000',manifests,remaps,addresses,markets)
    assert len(rows)==1 and rows[0].market_id=='z' and rows[0].settled_at==utc(t+86401)
    assert all(g['time']==t+7200 for g in gates['x'])
