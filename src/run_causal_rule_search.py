"""Independent configurable, untrained causal-rule research on a full raw tape.

This runner keeps evidence tiers explicit. Metadata-proxy results, even above
200%, are not reported as a verified no-lookahead or executable-profit success.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import gc
import math
import re
import time

import numpy as np

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig
from build_expanded_replay import SIDES, utc
from causal_rule_engine import MaturityRule, ObservedFinalityBacktester, prefix_activation, replay_indices
from data.build_dataset import build_markets
from expand_market_universe import ROOT, START, END, read, dump, sha
from features.embeddings import SimilarityConfig
from market_activation import FractionalResolution
from models import Direction, TimelineEvent, TradeEvent, TraderMarketSettlement
from prepare_causal_expert_history import VARIANTS
from run_dynamic_universe_study import immutable_json
from run_research import _summarize_result
from run_similarity_kernel_study import Kernel, KernelSkillEstimator

FILES=('src/run_causal_rule_search.py','src/causal_rule_engine.py',
    'src/prepare_causal_expert_history.py','src/resolution_evidence.py',
    'src/run_dynamic_universe_fast.py','src/run_dynamic_universe_study.py',
    'src/run_similarity_kernel_study.py','src/market_activation.py',
    'src/backtest/engine.py','src/alpha/trader_skill.py','src/alpha/signal.py',
    'src/alpha/semantic_risk.py','src/features/embeddings.py','src/models.py')
BASE=ROOT/'config/paper_experiments.json'


def validate_cases(cases):
    base=read(BASE)['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config']
    if not cases:raise ValueError('No research cases supplied.')
    for name,case in cases.items():
        if not re.fullmatch('[a-z0-9_]+',name):raise ValueError('Unsafe case name.')
        rule=MaturityRule(**case['maturity'])
        if case['history_variant'] not in VARIANTS:raise ValueError('Unknown historical evidence window.')
        if case['similarity'] not in ('semantic','uniform'):raise ValueError('Unsupported expert mode.')
        config=BacktestConfig(**(base|case.get('overrides',{})))
        if config.initial_balance!=10000 or not 0<=config.target_exposure_fraction<=1:
            raise ValueError('Fixed $10k initial capital and unlevered target exposure required.')
        if config.trade_fee_bps<0 or config.slippage_bps<0 or config.delay_seconds<1:
            raise ValueError('Do not invent rebates or use instantaneous hypothetical fills.')
        horizon=case.get('planning_horizon_days')
        if horizon is not None:
            if not config.min_days_to_resolution<horizon<config.max_days_to_resolution:
                raise ValueError('Fixed planning horizon must satisfy configured order-policy bounds.')
            if rule.max_days_to_scheduled_close is not None:
                raise ValueError('A deadline-free branch cannot also use a scheduled maturity cutoff.')
        elif rule.max_days_to_scheduled_close is None:
            raise ValueError('Declare a fixed planning horizon for deadline-free entry/sizing.')
        if case.get('ignore_risk_text',False) and case['similarity']!='uniform':
            raise ValueError('Text-independent branch requires uniform scores as well as blank risk text.')
        if 'kernel' in case:
            if case['similarity']!='semantic':raise ValueError('Locality kernels require semantic mode.')
            Kernel(**case['kernel'])
    return base


def input_binding(source,history_root):
    cohort=sorted(read(source/'cohort.json'),key=lambda x:x['market_id'])
    ids=[m['market_id'] for m in cohort]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate source condition.')
    completion=read(history_root/'complete.json')
    if not completion.get('complete') or completion['completed']!=len(ids) or completion['total']!=len(ids):
        raise ValueError('Require a completed full-inventory historical preparation.')
    for name,digest in completion['output_files_sha256'].items():
        if sha(history_root/name)!=digest:raise ValueError('Historical preparation manifest changed.')
    prepared_binding=read(history_root/'input_bindings.json')
    for name,digest in prepared_binding['files_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Source inventory, metadata or vectors changed after preparation.')
    records={};code_hashes={}
    for mid in ids:
        folder=history_root/'normalized'/mid
        manifest=read(folder/'manifest.json')
        if manifest['market_id']!=mid or sha(folder/'data.npz')!=manifest['array_sha256']:
            raise ValueError('Historical sidecar binding failed.')
        bound=manifest['binding']
        if Path(bound['source']).resolve()!=source or bound['min_cumulative_notional']<1e6:
            raise ValueError('History source/admission differs from requested data.')
        if sha(source/'trades'/mid/'complete.json')!=bound['provenance']['raw_complete_sha256']:
            raise ValueError('Raw source changed after history preparation.')
        for filename,digest in bound['source_files_sha256'].items():
            if filename not in code_hashes:code_hashes[filename]=sha(ROOT/filename)
            if code_hashes[filename]!=digest:raise ValueError(f'Bound history code changed: {filename}')
        records[mid]=dict(manifest_sha256=sha(folder/'manifest.json'),array_sha256=manifest['array_sha256'])
    return dict(source=str(source),history_root=str(history_root),count=len(ids),market_order=ids,
        order_policy='ConditionId lexicographic, never archived final-volume rank.',
        historical_completion_sha256=sha(history_root/'complete.json'),
        cohort_sha256=sha(source/'cohort.json'),vectors_sha256=sha(source/'vectors.npz'),records=records)


def load_inputs(source,history_root,cases):
    cohort=sorted(read(source/'cohort.json'),key=lambda x:x['market_id'])
    addresses,address_index=[],{}
    markets,manifests,chunks,remaps={},[],[],[]
    rules={name:MaturityRule(**case['maturity']) for name,case in cases.items()}
    gates={name:[] for name in cases}
    for i,item in enumerate(cohort):
        mid=item['market_id'];folder=history_root/'normalized'/mid
        m=read(folder/'manifest.json');manifests.append(m)
        market=dict(m['binding']['provenance']['source_market'])
        # Explicitly discard inherited future-last-print timing and outcome.
        available=m['label_available_at'] if m['label_usable'] else None
        payout=m['payout_yes'] if m['label_usable'] else None
        market['resolved_at']=utc(available) if available is not None else None
        market['resolution']='YES' if payout==1 else 'NO' if payout==0 else None
        for key in ('created_at','close_time'):
            market[key]=datetime.fromisoformat(market[key]) if market[key] else None
        markets.update(build_markets([market]))
        with np.load(folder/'data.npz',allow_pickle=False) as a:
            local=[]
            for address in a['wallets'].tolist():
                if address not in address_index:
                    address_index[address]=len(addresses);addresses.append(address)
                local.append(address_index[address])
            remap=np.asarray(local,dtype=np.uint32);remaps.append(remap)
            times,sizes,prices,token_prices=(a[k] for k in
                ('trade_time','trade_size','trade_price','trade_token_price'))
            chunk=dict(time=times,wallet=remap[a['trade_wallet']],side=a['trade_side'],
                price=prices,size=sizes,market=np.full(len(times),i,dtype=np.uint32))
            chunks.append(chunk)
            close=(market['close_time']-datetime(1970,1,1)).total_seconds() if market['close_time'] else None
            cache={}
            for name,rule in rules.items():
                if rule not in cache:
                    cache[rule]=prefix_activation(times,sizes,prices,token_prices,
                        rule=rule,observed_until=END,scheduled_close=close)
                gates[name].append(cache[rule])
        if (i+1)%5000==0:print(f'Loaded full causal tape {i+1}/{len(cohort)}',flush=True)
    arrays={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    order=np.argsort(arrays['time'],kind='stable')
    arrays={k:v[order] for k,v in arrays.items()}
    with np.load(source/'vectors.npz',allow_pickle=False) as a:
        by_id={mid:i for i,mid in enumerate(a['market_ids'].tolist())}
        vectors=a['vectors'][[by_id[mid] for mid in markets]]
    del chunks,order,address_index;gc.collect()
    return markets,manifests,arrays,addresses,remaps,vectors,gates


def history_rows(history_root,variant,manifests,remaps,addresses,markets):
    rows=[]
    for m,remap in zip(manifests,remaps):
        if not m['label_usable']:continue
        mid=m['market_id'];available=markets[mid].resolved_at
        if available is None:raise ValueError('Usable history without availability.')
        if m['raw_resolution_boundary']>=m['label_available_at']:
            raise ValueError('Historical labels exposed before evidence.')
        with np.load(history_root/'normalized'/mid/'data.npz',allow_pickle=False) as a:
            wallet=remap[a[f'{variant}_history_wallet']]
            rows.extend(TraderMarketSettlement(addresses[w],mid,float(s),float(n),available)
                for w,s,n in zip(wallet,a[f'{variant}_history_score'],a[f'{variant}_history_notional']))
    return rows


def timeline(markets,manifests,arrays,addresses,indices):
    resolutions=[FractionalResolution(m['market_id'],markets[m['market_id']].resolved_at,None,m['payout_yes'])
        for m in manifests if m['label_usable'] and markets[m['market_id']].resolved_at is not None
        and utc(START)<=markets[m['market_id']].resolved_at<=utc(END)]
    resolutions.sort(key=lambda r:(r.resolved_at,r.market_id));cursor=0
    ids=list(markets)
    columns=('time','market','wallet','side','price','size')
    selected=[arrays[k][indices] for k in columns] if len(indices) else []
    trades=zip(indices,*selected) if len(indices) else ()
    for index,epoch,m,w,s,p,q in trades:
        t=utc(epoch)
        # Evidence already available in this second precedes new trading.
        while cursor<len(resolutions) and resolutions[cursor].resolved_at<=t:
            r=resolutions[cursor];yield TimelineEvent('resolution',r.resolved_at,r);cursor+=1
        tr=TradeEvent(str(int(index)),ids[m],addresses[w],SIDES[s],float(p),float(q),t)
        yield TimelineEvent('trade',t,tr)
    for r in resolutions[cursor:]:yield TimelineEvent('resolution',r.resolved_at,r)


def run_worker(source_text,history_text,output_text,names):
    source,history_root,output=map(Path,(source_text,history_text,output_text))
    protocol=read(output/'protocol.json');cases={n:protocol['cases'][n] for n in names}
    markets,manifests,arrays,addresses,remaps,vectors,gates=load_inputs(source,history_root,cases)
    for name,case in cases.items():
        folder=output/'results'/name
        if (folder/'complete.json').exists():continue
        began=time.monotonic()
        config=BacktestConfig(**(protocol['base_config']|case.get('overrides',{})))
        history=history_rows(history_root,case['history_variant'],manifests,remaps,addresses,markets)
        history_count=len(history)
        opts=dict(similarity_config=SimilarityConfig(True,0),similarity_mode=case['similarity'],precomputed_market_vectors=vectors)
        estimator=(KernelSkillEstimator(markets,history,kernel=Kernel(**case['kernel']),**opts)
            if 'kernel' in case else TraderSkillEstimator(markets,history,**opts))
        del history
        epochs=np.array([g['time'] if g else np.iinfo(np.int64).max for g in gates[name]],dtype=np.int64)
        indices,thinned=replay_indices(arrays['time'],arrays['market'],epochs,config)
        active={mid:utc(t) for mid,t in zip(markets,epochs) if t!=np.iinfo(np.int64).max}
        engine=ObservedFinalityBacktester(markets,timeline(markets,manifests,arrays,addresses,indices),estimator,config,
            activation_times=active,planning_horizon_days=case.get('planning_horizon_days'),
            ignore_risk_text=case.get('ignore_risk_text',False),observation_start=utc(START))
        print(f'Running {name}: history={history_count:,}; gate={len(active):,}; full={len(arrays["time"]):,}; loop={len(indices):,}',flush=True)
        result=engine.run()
        expected=config.initial_balance+sum(p.pnl for p in result['closed_positions'])-sum(f.fee for f in result['fills'])+result['open_unrealized_pnl']
        if not math.isclose(expected,result['total_equity'],rel_tol=0,abs_tol=1e-6):raise ValueError('Cash/inventory accounting mismatch.')
        if result['balance']<-1e-8 or any(r['cash_balance']<-1e-8 for r in result['equity_curve']):raise ValueError('Unfunded borrowing detected.')
        if any(f.signal_time<active[f.market_id] for f in result['fills']):raise ValueError('Trade before market admission.')
        if any(markets[f.market_id].resolved_at is not None and f.filled_at>=markets[f.market_id].resolved_at for f in result['fills']):
            raise ValueError('Fill after observed finality.')
        summary=_summarize_result(name,result,config.initial_balance)
        tiers={tier:sum(m['evidence_tier']==tier for m in manifests) for tier in sorted({m['evidence_tier'] for m in manifests})}
        summary.update(universe=len(markets),history_variant=case['history_variant'],history_rows=history_count,
            activated_markets=len(active),full_valid_prints=len(arrays['time']),iterated_prints=len(indices),
            thinned_inert_prints=thinned,post_resolution_prints_skipped=engine.post_resolution_prints_skipped,
            evidence_tiers=tiers,exceeds_200pct=summary['total_return']>2,
            goal_claim_eligible=False,goal_claim_note='Performance alone is insufficient: independent timing, inventory, text/deadline and execution evidence audit still required.',
            execution_assumption='Quantity-unconstrained observed-price simulation' if config.max_fill_participation is None else 'Historical-print participation scenario; not proof of available book depth.',
            max_drawdown_frequency='daily saved equity',elapsed_seconds=time.monotonic()-began,start=str(utc(START)),end=str(utc(END)))
        dump(folder/'config.json',dict(case=case,backtest=asdict(config),protocol_sha256=sha(output/'protocol.json')))
        for key,rows in (('positions',result['closed_positions']),('fills',result['fills']),('open_positions',result['open_positions'])):
            dump(folder/f'{key}.json',[asdict(x) for x in rows])
        dump(folder/'equity.json',result['equity_curve'])
        dump(folder/'activations.json',{mid:g for mid,g in zip(markets,gates[name]) if g})
        dump(folder/'complete.json',summary)
        print(f'COMPLETE {name}: return={summary["total_return"]:.8f}, DD={summary["max_drawdown"]:.8f}, closed={summary["closed_positions"]}',flush=True)
        del engine,result,estimator,indices;gc.collect()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--history-root',type=Path,required=True)
    p.add_argument('--spec-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--cases',nargs='+')
    args=p.parse_args()
    source,history_root,output=args.source.resolve(),args.history_root.resolve(),args.output.resolve()
    if not 1<=args.workers<=4:raise ValueError('Use 1-4 workers.')
    if output==ROOT/'artifacts' or not output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Dedicated artifact output required.')
    for other in (source,history_root):
        if output==other or output in other.parents or other in output.parents:raise ValueError('Output overlaps source.')
    cases=read(args.spec_file)['cases'];base=validate_cases(cases)
    names=list(cases) if args.cases is None else args.cases
    if len(names)!=len(set(names)) or not set(names)<=set(cases):raise ValueError('Invalid requested cases.')
    binding=input_binding(source,history_root);immutable_json(output/'input_bindings.json',binding)
    protocol=dict(kind='causal_rule_search_v1',source=str(source),history_root=str(history_root),
        count=binding['count'],cases=cases,base_config=base,start=START,end=END,
        input_bindings_sha256=sha(output/'input_bindings.json'),spec_sha256=sha(args.spec_file),
        source_files_sha256={name:sha(ROOT/name) for name in FILES},
        causality='Full valid unfiltered tape; prefix-only maturity; no future-last-print label time; known finality prevents new entries; lexicographic condition tie ordering.',
        evidence_caution='Chronological calculation is not certification of retrospective coverage, text/deadline versions, labels or execution capacity.',
        frozen_files_sha256={name:sha(ROOT/name) for name in ('reports/polymarket_iclr2027/frozen/main.pdf','reports/polymarket_iclr2027/frozen/anonymous_code.zip')})
    immutable_json(output/'protocol.json',protocol)
    pending=[]
    for name in names:
        folder=output/'results'/name
        if (folder/'complete.json').exists():
            if read(folder/'config.json')['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('Result/protocol mismatch.')
        else:pending.append(name)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        tasks=[pool.submit(run_worker,str(source),str(history_root),str(output),pending[i::args.workers])
               for i in range(args.workers) if pending[i::args.workers]]
        for task in as_completed(tasks):task.result()
    if input_binding(source,history_root)!=binding:raise ValueError('Inputs changed during replay.')
    for filename,digest in protocol['source_files_sha256'].items():
        if sha(ROOT/filename)!=digest:raise ValueError('Research code changed during run.')
    for filename,digest in protocol['frozen_files_sha256'].items():
        if sha(ROOT/filename)!=digest:raise ValueError('Frozen artifact changed.')
    completed=[n for n in cases if (output/'results'/n/'complete.json').exists()]
    dump(output/'run_manifest.json',dict(complete=len(completed)==len(cases),completed=completed,
        expected=list(cases),protocol_sha256=sha(output/'protocol.json'),finished_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
