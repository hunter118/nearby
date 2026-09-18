import numpy as np
import pytest

from prepare_causal_expert_history import build_histories, volume_admission, VARIANTS, read_corrected_tape
from expand_market_universe import dump, sha, START, END


def fixture(n=2100):
    times=np.arange(n+1,dtype=np.int64)+1_700_000_000
    sizes=np.r_[2_000_000.,np.full(n,10.)]
    return dict(times=times,wallet=np.zeros(n+1,dtype=np.uint32),
        side=np.zeros(n+1,dtype=np.uint8),size=sizes,
        yes_price=np.full(n+1,.5),token_price=np.full(n+1,.5),
        payout_yes=1.,event_at=int(times[-1])+1,label_available_at=int(times[-1])+2,
        wallet_count=1,observed_until=int(times[-1])+100)


def test_volume_admission_uses_complete_second_and_excludes_trigger_second():
    t=np.array([10,10,10,11],dtype=np.int64)
    admission=volume_admission(t,np.array([800_000.,800_000.,800_000.,1.]),np.full(4,.5))
    assert admission['time']==11 and admission['crossing_timestamp']==10
    assert admission['observed_cumulative_notional']==1_200_000.
    assert admission['crossing_group_last_index']==2


def test_minimum_volume_and_array_validation():
    with pytest.raises(ValueError,match='minimum'):
        volume_admission(np.array([1]),np.array([2e6]),np.array([.5]),750000)
    with pytest.raises(ValueError,match='Chronological'):
        volume_admission(np.array([2,1]),np.ones(2),np.ones(2))
    with pytest.raises(ValueError,match='positive'):
        volume_admission(np.array([1]),np.array([-1.]),np.array([.5]))
    assert volume_admission(np.array([1,2]),np.array([1e6,1e6]),np.array([.2,.2])) is None


def test_exact_past_tail_counts_and_no_crossing_print_in_history():
    arrays,audit=build_histories(**fixture())
    assert audit['admission_time']==1_700_000_001
    assert {k:v['selected_prints'] for k,v in audit['variants'].items()}=={
        'all_after_admission':2100,'tail500':500,'tail1000':1000,'tail2000':2000,'days7':2100,'days28':2100}
    assert arrays['all_after_admission_history_notional'].tolist()==[10500.]
    assert arrays['tail500_history_score'].tolist()==[1.]


def test_resolution_entire_second_is_excluded_and_label_has_strict_delay():
    args=fixture(5)
    args['event_at']=int(args['times'][3]);args['label_available_at']=args['event_at']+1
    _,audit=build_histories(**args)
    assert audit['variants']['all_after_admission']['selected_prints']==2
    assert audit['variants']['all_after_admission']['last_timestamp']<args['event_at']
    args['label_available_at']=args['event_at']
    with pytest.raises(ValueError,match='at least one second'):
        build_histories(**args)


def test_future_suffix_does_not_change_admission_or_past_wallet_labels():
    args=fixture(12); before,ba=build_histories(**args)
    for key in ('times','wallet','side','size','yes_price','token_price'):
        extra=np.array([args['event_at'],args['event_at']+20]) if key=='times' else args[key][-2:]
        args[key]=np.r_[args[key],extra]
    after,aa=build_histories(**args)
    for key in before:
        assert np.array_equal(before[key],after[key])
    assert ba['admission']==aa['admission'] and ba['variants']==aa['variants']
    assert aa['prints_at_or_after_boundary']==2


def test_future_high_volume_cannot_admit_a_past_history():
    args=fixture(5);args['size'][:]=1.
    args['size'][-1]=10_000_000.
    args['event_at']=int(args['times'][-1]);args['label_available_at']=args['event_at']+1
    arrays,audit=build_histories(**args)
    assert audit['admission_time'] is None
    assert all(len(arrays[f'{v}_history_wallet'])==0 for v in VARIANTS)


def test_day_windows_are_relative_to_event_not_dataset_end_or_final_print():
    args=fixture(6);t=1_700_000_000
    args['times']=np.array([t,t+1,t+2*86400,t+20*86400,t+23*86400,t+29*86400,t+30*86400])
    args['event_at']=t+30*86400;args['label_available_at']=args['event_at']+10
    args['observed_until']=args['event_at']+100000
    _,a=build_histories(**args)
    assert a['variants']['all_after_admission']['selected_prints']==5
    assert a['variants']['days7']['selected_prints']==2
    assert a['variants']['days28']['selected_prints']==4


def test_missing_or_future_label_releases_no_contributions():
    args=fixture(5);args['payout_yes']=None;args['label_available_at']=None
    arrays,audit=build_histories(**args)
    assert not audit['label_usable']
    args=fixture(5);args['label_available_at']=args['observed_until']+1
    arrays,audit=build_histories(**args)
    assert not audit['label_usable']
    assert all(len(arrays[f'{v}_history_wallet'])==0 for v in VARIANTS)
    assert all(len(arrays[f'{v}_history_wallet'])==0 for v in VARIANTS)
    args=fixture(5);args['event_at']=args['observed_until']+1;args['label_available_at']=args['event_at']+1
    arrays,audit=build_histories(**args)
    assert not audit['label_usable']


def test_explicit_joint_gate_can_only_delay_verified_volume_admission():
    args=fixture(10);a=args['times'][0]
    with pytest.raises(ValueError,match='precedes'):
        build_histories(**args,admission={'time':int(a)})
    _,audit=build_histories(**args,admission={'time':int(a)+5,'kind':'joint'})
    assert audit['variants']['all_after_admission']['selected_prints']==6
    _,audit=build_histories(**args,admission={'time':None,'kind':'joint_never_activated'})
    assert audit['variants']['all_after_admission']['selected_prints']==0


def test_fractional_signed_incremental_wallet_score():
    t=1_700_000_000
    args=dict(times=np.array([t,t+10,t+20,t+30]),wallet=np.array([0,1,1,1]),
        side=np.array([0,0,2,1]),size=np.array([2e6,10.,4.,5.]),
        yes_price=np.array([.5,.2,.8,.3]),token_price=np.array([.5,.2,.8,.7]),
        payout_yes=.5,event_at=t+40,label_available_at=t+41,wallet_count=2,observed_until=t+50)
    arrays,_=build_histories(**args)
    assert arrays['all_after_admission_history_wallet'].tolist()==[1]
    assert arrays['all_after_admission_history_notional'][0]==pytest.approx(8.7)
    assert arrays['all_after_admission_history_score'][0]==pytest.approx(3.2/8.7)


def test_price_mapping_and_observation_bounds_fail_closed():
    args=fixture(3);args['token_price'][2]=.7
    with pytest.raises(ValueError,match='mapping'):
        build_histories(**args)
    args=fixture(3);args['observed_until']=int(args['times'][-1])-1
    with pytest.raises(ValueError,match='observation bound'):
        build_histories(**args)


def test_empty_tape_is_explicit_empty_histories():
    args=fixture(0)
    for key in ('times','wallet','side','size','yes_price','token_price'):
        args[key]=args[key][:0]
    arrays,audit=build_histories(**args)
    assert audit['admission_time'] is None and audit['prints_before_boundary']==0
    assert all(len(arrays[f'{v}_history_wallet'])==0 for v in VARIANTS)


def test_raw_reader_keeps_full_tape_token_correction_zero_price_and_quarantine(tmp_path,monkeypatch):
    import prepare_causal_expert_history as module
    mid='m'
    dump(tmp_path/'trades'/mid/'complete.json',dict(market_id=mid,status='api_window_exhausted',start=START,end=END,rows=3,pages=[]))
    dump(tmp_path/'normalized'/mid/'manifest.json',dict(market_id=mid,
        source_manifest_sha256=sha(tmp_path/'trades'/mid/'complete.json'),market={'market_id':mid,
        'clob_token_ids':['yes-token','no-token'],'outcome_labels':['Yes','No'],
        'close_time':None}))
    rows=[dict(timestamp=START+1,price=.2,size=10,asset='no-token',outcome='No',outcomeIndex=0,
              side='BUY',proxyWallet='0x'+'a'*40),
          dict(timestamp=START+2,price=0.,size=3,asset='yes-token',outcome='Yes',outcomeIndex=0,
              side='SELL',proxyWallet='0x'+'b'*40),
          dict(timestamp=START+3,price=1.01,size=5,asset='yes-token',outcome='Yes',outcomeIndex=0,
              side='BUY',proxyWallet='0x'+'c'*40)]
    monkeypatch.setattr(module,'iter_selected_rows',lambda *args:iter(rows))
    tape,wallets,provenance=read_corrected_tape(tmp_path,mid)
    assert tape['times'].tolist()==[START+1,START+2]
    assert tape['side'].tolist()==[1,2]
    assert tape['yes_price'].tolist()==[.8,0.]
    assert tape['token_price'].tolist()==[.2,0.]
    assert len(wallets)==2 and provenance['full_valid_prints']==2
    assert provenance['counters']['outcome_index_corrected']==1
    assert provenance['counters']['invalid_price_quarantined']==1
    assert len(provenance['quarantine_sha256'])==64
    assert provenance['quarantined_price_rows'][0]['source_row_index']==2


def test_raw_reader_rejects_wrong_identity_before_reading_rows(tmp_path):
    dump(tmp_path/'trades'/'m'/'complete.json',dict(market_id='other',
        status='api_window_exhausted',start=START,end=END))
    with pytest.raises(ValueError,match='Raw window'):
        read_corrected_tape(tmp_path,'m')


def test_sidecar_refuses_symlinked_market_directory(tmp_path,monkeypatch):
    import prepare_causal_expert_history as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    source=tmp_path/'source';output=tmp_path/'artifacts'/'study'
    mid='0x'+'a'*64; (output/'normalized').mkdir(parents=True)
    other=tmp_path/'other';other.mkdir()
    (output/'normalized'/mid).symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        module.prepare_market(source,output,mid,{},metadata_observed_at=END+1)
