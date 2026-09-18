"""Independent arithmetic, timing, paired-input and frozen-artifact checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import math
import subprocess

import numpy as np

from expand_market_universe import ROOT, START, END, read, dump, sha
from market_activation import GATES
from prepare_activation_study import SOURCE, OUTPUT


def timestamp(value):
    return datetime.fromisoformat(value)


def close(a,b):
    assert math.isclose(a,b,rel_tol=0,abs_tol=1e-6),(a,b)


def compare_variant_inputs(kind):
    assert kind in ('local-volume-check','cumulative-volume-check')
    common=['wallets','trade_time','trade_wallet','trade_side','trade_price','trade_size',
            'baseline_history_wallet','baseline_history_score','baseline_history_notional']
    count=0
    for item in read(SOURCE/'cohort.json'):
        mid=item['market_id']
        parent=OUTPUT/'normalized'/mid
        neighbor=OUTPUT/kind/'normalized'/mid
        pm,nm=read(parent/'manifest.json'),read(neighbor/'manifest.json')
        assert pm['market']==nm['market'] and pm['payout_yes']==nm['payout_yes']
        assert pm['source_manifest_sha256']==nm['source_manifest_sha256']
        for folder,m in ((parent,pm),(neighbor,nm)):
            assert sha(folder/'data.npz')==m['array_sha256']
        with np.load(parent/'data.npz',allow_pickle=False) as a,np.load(neighbor/'data.npz',allow_pickle=False) as b:
            for key in common:
                assert np.array_equal(a[key],b[key]),(mid,key)
        if kind=='local-volume-check':
            low=nm['activations']['p90_v200k']
            middle=pm['activations']['p90_v250k']
            high=nm['activations']['p90_v300k']
        else:
            low=nm['activations']['p90_cum250k']
            middle=nm['activations']['p90_cum1000k']
            high=nm['activations']['p90_cum5000k']
            rolling=pm['activations']['p90_v250k']
            if rolling:
                assert low and low['time']<=rolling['time']
            for a in (low,middle,high):
                if a:
                    assert a['cumulative_notional']+1e-6>=a['recent_notional']
        if high:
            assert middle and low and low['time']<=middle['time']<=high['time']
        elif middle:
            assert low and low['time']<=middle['time']
        count+=1
        if count%2000==0:
            print(f'Exact parent/{kind} input checks: {count}',flush=True)
    result=dict(markets=count,identical_common_arrays=common,
                identical_market_metadata=True,monotone_volume_cutoffs=True,
                archive_checksums=True)
    result['variant']=kind
    dump(OUTPUT/f'{kind}_input_validation.json',result)
    return result


def check_case(root,count,name,base_config):
    folder=root/'results'/str(count)/name
    summary=read(folder/'complete.json')
    config=read(folder/'config.json')
    assert config['backtest']==base_config
    assert config['protocol_sha256']==sha(root/'protocol.json')
    positions,fills,opens,curve=(read(folder/f'{n}.json') for n in ('positions','fills','open_positions','equity'))
    assert len(positions)==summary['closed_positions']
    assert len(fills)==summary['fills']
    assert len(opens)==summary['open_positions']
    assert len({p['market_id'] for p in positions})==len(positions)
    assert sum(p['pnl']<0 for p in positions)==summary['loss_count']
    profit,fees,cost,payout=0.,sum(f['fee'] for f in fills),sum(f['notional'] for f in fills),0.
    manifests={}
    for p in positions+fills+opens:
        mid=p['market_id']
        if mid not in manifests:
            manifests[mid]=read(root/'normalized'/mid/'manifest.json')
    for p in positions:
        m=manifests[p['market_id']]
        assert m['payout_yes'] is not None
        rate=m['payout_yes'] if p['direction']=='YES' else 1-m['payout_yes']
        close(p['payout'],p['quantity']*rate)
        close(p['notional'],p['quantity']*p['avg_entry_price'])
        close(p['pnl'],p['payout']-p['notional'])
        assert timestamp(p['resolved_at'])==timestamp(m['market']['resolved_at'])
        assert timestamp(p['resolved_at'])>=timestamp(p['opened_at'])
        profit+=p['pnl'];payout+=p['payout']
    for f in fills:
        m=manifests[f['market_id']]
        signal,filled=timestamp(f['signal_time']),timestamp(f['filled_at'])
        assert (filled-signal).total_seconds()>=base_config['delay_seconds']
        days=(timestamp(m['market']['close_time'])-signal).total_seconds()/86400
        assert base_config['min_days_to_resolution']<days<base_config['max_days_to_resolution']
        close(f['notional'],f['quantity']*f['fill_price'])
        close(f['fee'],f['notional']*base_config['trade_fee_bps']/10000)
        if name!='baseline':
            gate=name.split('__')[1]
            activation=m['activations'][gate]
            assert activation is not None
            assert signal>=datetime.fromtimestamp(activation['time'],timezone.utc).replace(tzinfo=None)
            spec=config['gate']
            if spec.get('volume_mode')=='cumulative':
                assert activation['volume_mode']=='cumulative'
                assert activation['cumulative_notional']>=spec['min_cumulative_notional']
            else:
                assert activation['recent_notional']>=spec['min_recent_notional']
            assert activation['recent_prints']>=spec['min_recent_prints']
            assert activation['observed_hours']>=spec['min_observed_hours']
            p=activation['recent_yes_vwap']
            assert max(p,1-p)>=spec['min_conviction']
            assert START<=activation['time']<=END
    open_cost=sum(p['quantity']*p['avg_entry_price'] for p in opens)
    close(cost,sum(p['notional'] for p in positions)+open_cost)
    close(summary['cash'],base_config['initial_balance']-cost-fees+payout)
    close(summary['net_realized_pnl'],profit-fees)
    close(summary['total_return'],summary['total_equity']/base_config['initial_balance']-1)
    close(summary['total_equity'],summary['cash']+summary['open_market_value'])
    close(summary['total_equity'],curve[-1]['total_equity'])
    peak=base_config['initial_balance'];drawdown=0.
    for row in curve:
        peak=max(peak,row['total_equity'])
        drawdown=min(drawdown,row['total_equity']/peak-1)
        close(row['total_equity'],row['cash_balance']+row['open_market_value'])
    close(summary['max_drawdown'],drawdown)
    return dict(root=str(root),universe=count,case=name,positions=len(positions),fills=len(fills),
                open_positions=len(opens),checks='all passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays',action='store_true')
    parser.add_argument('--allow-pending',action='store_true')
    args=parser.parse_args()
    if args.arrays:
        compare_variant_inputs('local-volume-check')
        compare_variant_inputs('cumulative-volume-check')
    base=read(ROOT/'config/paper_experiments.json')['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config']
    cases=[(OUTPUT,3000,n) for n in ['baseline']+[f'{s}__{g}' for s in ('flow','both') for g in GATES]]
    cases += [(OUTPUT,10000,n) for n in ['baseline']+[f'{s}__p90_v{v}k' for s in ('flow','both') for v in (50,250)]]
    cases += [(OUTPUT/'local-volume-check',n,f'both__p90_v{v}k') for n in (3000,10000) for v in (200,300)]
    cases += [(OUTPUT/'cumulative-volume-check',n,f'both__p90_cum{v}k') for n in (3000,10000) for v in (250,1000,5000)]
    completed,missing=[],[]
    for root,count,name in cases:
        if not (root/'results'/str(count)/name/'complete.json').exists():
            missing.append(dict(root=str(root),universe=count,case=name));continue
        completed.append(check_case(root,count,name,base))
    protocol=read(OUTPUT/'protocol.json')
    for key,file in [('frozen_pdf_sha256','main.pdf'),('frozen_zip_sha256','anonymous_code.zip')]:
        assert sha(ROOT/'reports/polymarket_iclr2027/frozen'/file)==protocol[key]
    code=['market_activation.py','prepare_activation_study.py','run_activation_study.py',
          'run_activation_neighborhood.py','run_activation_cumulative.py','audit_activation_example.py','report_activation_study.py',
          'validate_activation_study.py']
    result=dict(expected_cases=len(cases),completed=completed,pending=missing,
        frozen_artifacts_unchanged=True,source_sha256={p:sha(ROOT/'src'/p) for p in code},
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    dump(OUTPUT/'validation.json',result)
    print(f'Validated {len(completed)}/{len(cases)} complete cases; pending={missing}',flush=True)
    assert args.allow_pending or not missing


if __name__=='__main__':
    main()
