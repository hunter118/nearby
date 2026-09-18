"""Exact daily-portfolio optimization: omit causally inert pre-gate prints.

Gate construction, full input statistics, settled wallet histories, estimator,
and the original trading engine are unchanged. Retain every post-gate print,
the final real print of every UTC day, and every original resolution event.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import gc
import math
import time

import numpy as np

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig
from build_expanded_replay import SIDES, utc
from data.build_dataset import build_resolution_events
from expand_market_universe import ROOT, START, END, read, dump, sha
from features.embeddings import SimilarityConfig
from market_activation import ActivationBacktester, FractionalResolution
from models import TimelineEvent, TradeEvent
import run_dynamic_universe_study as generic
import run_activation_study as original
from run_research import _summarize_result
from run_semantic_risk_study import _write_experiment_artifacts

DEFAULT_PREPARED=ROOT/'artifacts/dynamic-universe-2026-09-18/partial-10000'
DEFAULT_OUTPUT=ROOT/'artifacts/dynamic-universe-2026-09-18/fast-equivalence-10000'
PORTFOLIO_FILES=('positions.json','fills.json','equity.json','open_positions.json')


def assert_safe(config,scope='both'):
    if config.equity_record_interval>0:
        raise ValueError('Exact thinning requires daily equity recording.')
    if config.pending_order_expiry_seconds is not None:
        raise ValueError('Inactive prints can expire pending orders; expiry must be None.')
    if scope!='both':
        raise ValueError('Only BOTH gated flow/history portfolios are supported.')


def retained_indices(times,market_index,gate_times):
    """Original indices, never fabricated clock trades or renumbered IDs."""
    times=np.asarray(times); market_index=np.asarray(market_index)
    if times.ndim!=1 or market_index.shape!=times.shape or np.any(np.diff(times)<0):
        raise ValueError('Expected aligned chronological full tape arrays.')
    if not len(times):
        return np.array([],dtype=np.int64),0
    active=times>=np.asarray(gate_times,dtype=np.int64)[market_index]
    keep=active.copy()
    days=times//86400
    last=np.r_[np.flatnonzero(days[:-1]!=days[1:]),len(times)-1]
    keep[last]=True
    return np.flatnonzero(keep),int(active.sum())


def thin_timeline(markets,manifests,arrays,addresses,indices):
    """Same resolution merge and tie ordering, with original full-array IDs."""
    resolutions=build_resolution_events(markets)
    for m in manifests:
        market=markets[m['market_id']]
        if m['payout_yes'] is not None and market.resolution is None and market.resolved_at:
            resolutions.append(FractionalResolution(m['market_id'],market.resolved_at,None,m['payout_yes']))
    resolutions=sorted((r for r in resolutions if utc(START)<=r.resolved_at<=utc(END)),key=lambda r:r.resolved_at)
    mids=list(markets); cursor=0
    thin={k:arrays[k][indices] for k in ('time','market','wallet','side','price','size')}
    for i,t,m,w,s,p,q in zip(indices,thin['time'],thin['market'],thin['wallet'],thin['side'],thin['price'],thin['size']):
        ts=utc(t)
        while cursor<len(resolutions) and resolutions[cursor].resolved_at<ts:
            r=resolutions[cursor]
            yield TimelineEvent('resolution',r.resolved_at,r)
            cursor+=1
        trade=TradeEvent(str(int(i)),mids[m],addresses[w],SIDES[s],float(p),float(q),ts)
        yield TimelineEvent('trade',ts,trade)
    for r in resolutions[cursor:]:
        yield TimelineEvent('resolution',r.resolved_at,r)


def gate_epochs(markets,manifests,gate):
    by_id={m['market_id']:m['activations'][gate] for m in manifests}
    return np.array([by_id[mid]['time'] if by_id[mid] else np.iinfo(np.int64).max
                     for mid in markets],dtype=np.int64)


def portfolio_equality(current,baseline):
    hashes={}
    for name in PORTFOLIO_FILES:
        a,b=sha(current/name),sha(baseline/name)
        if a!=b:
            raise AssertionError(f'Fast/full portfolio bytes differ: {name}')
        hashes[name]=a
    return hashes


def prepare_inputs(source,prepared,gates,label,workers=4):
    """Preparation-only adapter; never starts the generic slow portfolios."""
    if not 1<=workers<=4:
        raise ValueError('Use 1–4 preparation workers.')
    source,prepared=generic.safe_output(source,prepared)
    gates={k:asdict(v) for k,v in generic.configure(gates).items()}
    cohort=generic.select_cohort(read(source/'cohort.json'))
    binding=generic.input_binding(source,cohort)
    design=generic.protocol(source,cohort,len(cohort),gates,label,None,binding)
    generic.immutable_json(prepared/'input_bindings.json',binding)
    design['input_bindings_sha256']=sha(prepared/'input_bindings.json')
    generic.immutable_json(prepared/'protocol.json',design)
    pending=[m for m in cohort if not generic.prepared_valid(prepared,m,gates,binding)]
    metadata=read(source/'metadata.json.gz'); begun=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        tasks=[pool.submit(generic.prepare_one,m,metadata[m['market_id']],str(source),str(prepared),gates) for m in pending]
        for i,task in enumerate(as_completed(tasks),1):
            task.result()
            if i%1000==0 or i==len(tasks):
                print(f'Prepared-only {i}/{len(tasks)}; {time.monotonic()-begun:.1f}s',flush=True)
    for m in cohort:
        assert generic.prepared_valid(prepared,m,gates,binding)
    assert generic.input_binding(source,cohort)==binding
    dump(prepared/'preparation_only_manifest.json',dict(count=len(cohort),
        protocol_sha256=sha(prepared/'protocol.json'),source_adapter_sha256=sha(Path(__file__)),
        elapsed_seconds=time.monotonic()-begun,complete=True))


def prepared_binding(prepared):
    design=read(prepared/'protocol.json')
    assert design['kind']=='dynamic_universe_inventory_sensitivity_v1'
    source=Path(design['source'])
    cohort=generic.select_cohort(read(source/'cohort.json'),design['diagnostic_count_requested'])
    assert len(cohort)==design['count']
    binding=generic.input_binding(source,cohort)
    assert binding==read(prepared/'input_bindings.json')
    assert sha(prepared/'input_bindings.json')==design['input_bindings_sha256']
    arrays={}
    for item in cohort:
        assert generic.prepared_valid(prepared,item,design['gates'],binding)
        path=prepared/'normalized'/item['market_id']
        arrays[item['market_id']]=dict(manifest_sha256=sha(path/'manifest.json'),
            array_sha256=read(path/'manifest.json')['array_sha256'])
    for name,digest in design['source_files_sha256'].items():
        assert sha(ROOT/name)==digest, f'Bound original source changed: {name}'
    return design,arrays


def verification_certificate(path):
    certificate=read(path/'equivalence_summary.json')
    if not certificate['complete'] or not certificate['all_byte_exact']:
        raise ValueError('Optimization certificate is incomplete or not byte exact.')
    design=read(path/'protocol.json')
    assert design['fast_source_sha256']==sha(Path(__file__))
    for name,digest in design['original_source_files_sha256'].items():
        assert sha(ROOT/name)==digest, f'Validated execution source changed: {name}'
    assert certificate['protocol_sha256']==sha(path/'protocol.json')
    for gate,record in certificate['cases'].items():
        current=path/'results'/str(design['count'])/f'both__{gate}'
        baseline=Path(design['prepared'])/'results'/str(design['count'])/f'both__{gate}'
        assert portfolio_equality(current,baseline)==record['portfolio_files_sha256']
    return dict(path=str(path),sha256=sha(path/'equivalence_summary.json'),
                protocol_sha256=sha(path/'protocol.json'),core_config=design['core_config'])


def run_worker(prepared_text,output_text,names):
    prepared,output=Path(prepared_text),Path(output_text)
    design=read(output/'protocol.json'); count=design['count']
    with generic.temporary_settings(original,SOURCE=Path(design['source']),OUTPUT=prepared):
        markets,manifests,arrays,remaps,addresses,vectors=original.load_tape(count)
        config=BacktestConfig(**design['core_config']); assert_safe(config)
        for gate in names:
            folder=output/'results'/str(count)/f'both__{gate}'
            if (folder/'complete.json').exists():
                continue
            begun=time.monotonic()
            history=original.load_history(gate,manifests,remaps,addresses,markets)
            history_count=len(history)
            estimator=TraderSkillEstimator(markets,history,
                similarity_config=SimilarityConfig(True,0),similarity_mode='semantic',
                precomputed_market_vectors=vectors)
            del history
            epochs=gate_epochs(markets,manifests,gate)
            indices,active_count=retained_indices(arrays['time'],arrays['market'],epochs)
            activations={mid:utc(t) for mid,t in zip(markets,epochs) if t!=np.iinfo(np.int64).max}
            print(f'FAST {gate}: full={len(arrays["time"]):,}, actually_iterated={len(indices):,}, admitted={active_count:,}',flush=True)
            engine=ActivationBacktester(markets,thin_timeline(markets,manifests,arrays,addresses,indices),
                estimator,config,activation_times=activations)
            assert not engine.positions and not engine.pending_orders
            result=engine.run()
            assert engine.processed==len(indices) and engine.admitted==active_count
            expected=config.initial_balance+sum(p.pnl for p in result['closed_positions'])-sum(f.fee for f in result['fills'])+result['open_unrealized_pnl']
            assert math.isclose(expected,result['total_equity'],rel_tol=0,abs_tol=1e-6)
            assert all(f.signal_time>=activations[f.market_id] for f in result['fills'])
            summary=_summarize_result(f'both__{gate}',result,config.initial_balance)
            summary.update(universe=count,history_rows=history_count,gate=gate,
                activated_markets=len(activations),eligible_prints=len(arrays['time']),
                full_tape_count=len(arrays['time']),actually_iterated_prints=engine.processed,
                eligible_prints_definition='Full corrected input count, NOT number of loop iterations.',
                admitted_prints=engine.admitted,losing_positions=sum(p.pnl<0 for p in result['closed_positions']),
                elapsed_seconds=time.monotonic()-begun,start=utc(START),end=utc(END))
            dump(folder/'config.json',dict(backtest=asdict(config),gate=design['gates'][gate],
                gate_scope='both',history=gate,protocol_sha256=sha(output/'protocol.json'),
                prepared_protocol_sha256=design['prepared_protocol_sha256']))
            _write_experiment_artifacts(f'both__{gate}',result,summary,config.initial_balance,folder)
            for key,rows in (('positions',result['closed_positions']),('fills',result['fills']),('open_positions',result['open_positions'])):
                dump(folder/f'{key}.json',[asdict(x) for x in rows])
            dump(folder/'equity.json',result['equity_curve'])
            baseline=prepared/'results'/str(count)/f'both__{gate}'
            if design['verify_full_portfolios']:
                try:
                    hashes=portfolio_equality(folder,baseline)
                    old=read(baseline/'complete.json')
                    for key,value in old.items():
                        if key not in ('elapsed_seconds','start','end'):
                            assert summary[key]==value, f'Economic summary differs: {key}'
                    assert str(summary['start'])==old['start'] and str(summary['end'])==old['end']
                except Exception as error:
                    dump(folder/'equivalence_failure.json',dict(error=repr(error)))
                    raise
                validation=dict(byte_exact=True,portfolio_files_sha256=hashes,
                    baseline_complete_sha256=sha(baseline/'complete.json'),summary_exact_except_timing=True)
            else:
                validation=dict(byte_exact=None,reason='New inventory; compute optimization validated on separate reference study.',
                    prior_certificate=design['validated_optimization'])
            dump(folder/'validation.json',validation)
            dump(folder/'complete.json',summary)
            print(f'COMPLETE FAST {gate}: return={summary["total_return"]:.8f}, iterated={engine.processed:,}, elapsed={summary["elapsed_seconds"]:.1f}s',flush=True)
            del engine,result,estimator,indices
            gc.collect()


def run(prepared=DEFAULT_PREPARED,output=DEFAULT_OUTPUT,cases=None,workers=4,validated_optimization=None):
    if not 1<=workers<=4:
        raise ValueError('Use 1–4 replay workers.')
    prepared,output=Path(prepared).resolve(),Path(output).resolve()
    if (not output.is_relative_to(ROOT/'artifacts') or output==ROOT/'artifacts'
            or output==prepared or output in prepared.parents or prepared in output.parents):
        raise ValueError('Use disjoint dedicated prepared/output artifact folders.')
    original_design,arrays=prepared_binding(prepared)
    raw_source=Path(original_design['source']).resolve()
    if output==raw_source or raw_source in output.parents or output in raw_source.parents:
        raise ValueError('Fast output must not overlap raw source inputs.')
    if (output/'protocol.json').exists() and read(output/'protocol.json').get('kind')!='dynamic_universe_fast_v1':
        raise ValueError('Refusing to write into another study output.')
    config=BacktestConfig(**original_design['core_config']); assert_safe(config)
    gates=original_design['gates']; requested=list(gates) if cases is None else cases
    if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(gates):
        raise ValueError('Requested cases must be unique prepared gate names.')
    certificate=verification_certificate(Path(validated_optimization)) if validated_optimization else None
    if certificate:
        assert certificate['core_config']==original_design['core_config']
    else:
        for gate in gates:
            assert (prepared/'results'/str(original_design['count'])/f'both__{gate}'/'complete.json').exists(), 'Full baseline required, or pass a verified optimization certificate.'
    design=dict(kind='dynamic_universe_fast_v1',prepared=str(prepared),source=original_design['source'],
        count=original_design['count'],gates=gates,core_config=original_design['core_config'],
        prepared_protocol_sha256=sha(prepared/'protocol.json'),
        prepared_input_bindings_sha256=sha(prepared/'input_bindings.json'),
        fast_source_sha256=sha(Path(__file__)),original_source_files_sha256=original_design['source_files_sha256'],
        verify_full_portfolios=certificate is None,validated_optimization=certificate,
        safety='Fresh empty portfolio; BOTH flow/history gates; daily equity; expiry=None; retain all post-gate prints, each day final real print, all resolutions and original trade indices.',
        statistics='Every source market/gate/history is processed; full_tape_count and actually_iterated_prints are distinct. This is a computational optimization, not an alternative strategy or data filter.',
        inventory_label=original_design['inventory_label'])
    generic.immutable_json(output/'prepared_array_bindings.json',arrays)
    design['prepared_array_bindings_sha256']=sha(output/'prepared_array_bindings.json')
    generic.immutable_json(output/'protocol.json',design)
    begun=time.monotonic()
    pending=[]
    for gate in requested:
        folder=output/'results'/str(design['count'])/f'both__{gate}'
        if (folder/'complete.json').exists():
            assert read(folder/'config.json')['protocol_sha256']==sha(output/'protocol.json')
            if design['verify_full_portfolios']:
                assert portfolio_equality(folder,prepared/'results'/str(design['count'])/f'both__{gate}')==read(folder/'validation.json')['portfolio_files_sha256']
        else:
            pending.append(gate)
    chunks=[pending[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        tasks=[pool.submit(run_worker,str(prepared),str(output),chunk) for chunk in chunks if chunk]
        for task in as_completed(tasks):
            task.result()
    after,after_arrays=prepared_binding(prepared)
    assert after==original_design and after_arrays==arrays
    assert sha(Path(__file__))==design['fast_source_sha256']
    records={g:read(output/'results'/str(design['count'])/f'both__{g}'/'validation.json') for g in gates
        if (output/'results'/str(design['count'])/f'both__{g}'/'complete.json').exists()}
    result=dict(protocol_sha256=sha(output/'protocol.json'),expected=len(gates),completed=len(records),
        complete=len(records)==len(gates),all_byte_exact=bool(design['verify_full_portfolios'] and all(r['byte_exact'] for r in records.values())),
        cases=records,fast_source_sha256=sha(Path(__file__)))
    dump(output/'equivalence_summary.json',result)
    invocation=dict(completed_at=datetime.now(timezone.utc).isoformat(),requested=requested,
        newly_run=pending,elapsed_seconds=time.monotonic()-begun,workers=workers,
        protocol_sha256=sha(output/'protocol.json'),equivalence_summary_sha256=sha(output/'equivalence_summary.json'))
    dump(output/'invocations'/f'{time.time_ns()}.json',invocation)
    print(f'FAST STUDY verified {len(records)}/{len(gates)}; this invocation {invocation["elapsed_seconds"]:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=DEFAULT_PREPARED)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--validated-optimization',type=Path)
    parser.add_argument('--source',type=Path,help='When supplied, prepare FULL source inventory into --prepared first.')
    parser.add_argument('--spec-file',type=Path)
    parser.add_argument('--inventory-label',help='Required when preparing another source inventory.')
    parser.add_argument('--prepare-workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.source:
        if not args.spec_file or not args.inventory_label:
            parser.error('Source preparation needs --spec-file and --inventory-label.')
        spec=read(args.spec_file)
        prepare_inputs(args.source,args.prepared,spec.get('gates',spec),args.inventory_label,args.prepare_workers)
    elif args.prepare_only:
        parser.error('--prepare-only needs --source.')
    if not args.prepare_only:
        run(args.prepared,args.output,args.cases,args.workers,args.validated_optimization)


if __name__=='__main__':
    main()
