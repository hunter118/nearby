"""Independent data equality, timing, arithmetic and exact-anchor checks."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from itertools import product

import numpy as np

from expand_market_universe import ROOT,read,dump,sha,START,END
from prepare_activation_study import SOURCE,OUTPUT as PARENT
from run_activation_joint import OUTPUT,gate_name
from validate_activation_study import check_case

COMMON=('wallets','trade_time','trade_wallet','trade_side','trade_price','trade_size',
        'baseline_history_wallet','baseline_history_score','baseline_history_notional')


def check_inputs(folder):
    protocol=read(folder/'protocol.json')
    gates=protocol['gates']
    excluded=set(protocol.get('excluded_market_ids',[]))
    anchor=gate_name(1_000_000,40,.90)
    anchor_checked=0
    for i,item in enumerate(read(SOURCE/'cohort.json')[:3000],1):
        mid=item['market_id']; current=folder/'normalized'/mid
        m=read(current/'manifest.json');old=read(PARENT/'normalized'/mid/'manifest.json')
        assert m['market']==old['market'] and m['payout_yes']==old['payout_yes']
        assert m['source_manifest_sha256']==old['source_manifest_sha256']
        assert sha(current/'data.npz')==m['array_sha256']
        with np.load(current/'data.npz',allow_pickle=False) as a,np.load(PARENT/'normalized'/mid/'data.npz',allow_pickle=False) as b:
            for key in COMMON:
                assert np.array_equal(a[key],b[key]),(mid,key)
            if anchor in gates and not excluded:
                ref=PARENT/'cumulative-volume-check/normalized'/mid
                r=read(ref/'manifest.json')
                assert m['activations'][anchor]==r['activations']['p90_cum1000k']
                with np.load(ref/'data.npz',allow_pickle=False) as c:
                    for suffix in ('wallet','score','notional'):
                        assert np.array_equal(a[f'{anchor}_history_{suffix}'],c[f'p90_cum1000k_history_{suffix}'])
                anchor_checked+=1
            for name,spec in gates.items():
                t=m['activations'][name]
                if mid in excluded:
                    assert t is None and len(a[f'{name}_history_wallet'])==0
                    continue
                if t is None:
                    assert len(a[f'{name}_history_wallet'])==0
                    continue
                assert t['cumulative_notional']>=spec['min_cumulative_notional']
                assert max(t['recent_yes_vwap'],1-t['recent_yes_vwap'])>=spec['min_conviction']
                assert t['recent_prints']>=spec['min_recent_prints']
                assert t['observed_hours']>=spec['min_observed_hours']
                assert START<=t['time']<=END and t['time']%3600==0
                close=datetime.fromisoformat(m['market']['close_time']).replace(tzinfo=timezone.utc).timestamp()
                assert 0<close-t['time']<spec['max_days_to_scheduled_close']*86400
        # Every tighter cell can activate no earlier than a looser cell.
        for lo,hi in product(gates,gates):
            s,t=gates[lo],gates[hi]
            tighter=(s['min_cumulative_notional']<=t['min_cumulative_notional']
                and s['min_conviction']<=t['min_conviction']
                and s['max_days_to_scheduled_close']>=t['max_days_to_scheduled_close'])
            if tighter and m['activations'][hi]:
                assert m['activations'][lo] and m['activations'][lo]['time']<=m['activations'][hi]['time']
        if i%1000==0:
            print(f'{folder.name}: exact common inputs and gate checks {i}/3000',flush=True)
    result=dict(markets=3000,common_arrays_exact=True,gate_timing_and_thresholds=True,
        gate_monotonicity=True,exact_previous_anchor_market_histories=anchor_checked,
        excluded_markets_checked=len(excluded))
    dump(folder/'input_validation.json',result)
    return result


def validate(stage,arrays=False,allow_pending=False):
    folder=OUTPUT/stage
    protocol=read(folder/'protocol.json')
    if arrays:
        check_inputs(folder)
    base=read(ROOT/'config/paper_experiments.json')['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config']
    completed=[];pending=[]
    for name,spec in protocol['gates'].items():
        case=f'both__{name}'; path=folder/'results/3000'/case
        if not (path/'complete.json').exists():
            pending.append(case);continue
        result=check_case(folder,3000,case,base)
        assert read(path/'config.json')['gate']==spec
        excluded=set(protocol.get('excluded_market_ids',[]))
        assert not ({p['market_id'] for p in read(path/'fills.json')} & excluded)
        completed.append(result)
    anchor=folder/f'results/3000/both__{gate_name(1_000_000,40,.9)}'
    exact_anchor=False
    if (anchor/'complete.json').exists() and not protocol.get('excluded_market_ids'):
        prior=PARENT/'cumulative-volume-check/results/3000/both__p90_cum1000k'
        for filename in ('positions.json','fills.json','equity.json','open_positions.json'):
            assert read(anchor/filename)==read(prior/filename),filename
        exact_anchor=True
    for key,filename in [('frozen_pdf_sha256','main.pdf'),('frozen_zip_sha256','anonymous_code.zip')]:
        assert sha(ROOT/'reports/polymarket_iclr2027/frozen'/filename)==protocol[key]
    result=dict(stage=stage,expected=len(protocol['gates']),completed=completed,pending=pending,
        exact_previous_anchor_portfolio=exact_anchor,frozen_artifacts_unchanged=True)
    dump(folder/'validation.json',result)
    print(f'{stage}: validated {len(completed)}/{len(protocol["gates"])}; pending {len(pending)}; exact anchor {exact_anchor}',flush=True)
    assert allow_pending or not pending
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='grid')
    parser.add_argument('--arrays',action='store_true')
    parser.add_argument('--allow-pending',action='store_true')
    args=parser.parse_args()
    validate(args.stage,args.arrays,args.allow_pending)


if __name__=='__main__':
    main()
