"""Isolated driver for the held-position dead-signal optimization.

The original runner/engine remain byte-unchanged. Each spawned worker replaces
only its own in-memory constructor with the separately audited subclass; every
input, strategy parameter, output account and event stream remains unchanged.
Real reference comparisons are required before using this adapter for new cases.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
from multiprocessing import get_context
from pathlib import Path

import run_causal_rule_search as original
from expand_market_universe import ROOT,START,END,read,dump,sha
from held_position_fast import HeldPositionFastBacktester
from run_dynamic_universe_study import immutable_json
from run_dynamic_universe_fast import portfolio_equality

FILES=(*original.FILES,'src/held_position_fast.py','src/run_causal_rule_fast.py')


def fast_worker(source,history,output,names):
    original.ObservedFinalityBacktester=HeldPositionFastBacktester
    original.run_worker(source,history,output,names)


def certificate(path):
    c=read(path/'reference_validation.json')
    if not c['all_byte_exact'] or not c['cases']:raise ValueError('Missing real-portfolio validation.')
    protocol=read(path/'protocol.json')
    reference=Path(c['reference'])
    reference_protocol=read(reference/'protocol.json')
    if (c['protocol_sha256']!=sha(path/'protocol.json')
            or protocol['kind']!='causal_rule_held_position_fast_v1'
            or set(c['source_files_sha256'])!=set(FILES)
            or c['source_files_sha256']!=protocol['source_files_sha256']
            or protocol['reference']!=str(reference)
            or c['reference_protocol_sha256']!=sha(reference/'protocol.json')
            or protocol['input_bindings_sha256']!=sha(path/'input_bindings.json')
            or reference_protocol['input_bindings_sha256']!=sha(reference/'input_bindings.json')
            or set(reference_protocol['source_files_sha256'])!=set(original.FILES)):
        raise ValueError('Optimization certificate/protocol binding changed.')
    for name,digest in c['source_files_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Validated implementation changed.')
    for name,row in c['cases'].items():
        own,ref=path/'results'/name,reference/'results'/name
        if portfolio_equality(own,ref)!=row['portfolio_files_sha256']:
            raise ValueError('Reference portfolio bytes changed.')
        for filename in ('complete.json','config.json'):
            if (sha(own/filename)!=row['output_sha256'][filename]
                    or sha(ref/filename)!=row['reference_sha256'][filename]):
                raise ValueError('Certified economic summary/configuration changed.')
    return dict(path=str(path),sha256=sha(path/'reference_validation.json'),cases=list(c['cases']))


def run(source,history,spec,output,workers,names=None,reference=None,validated=None):
    source,history,spec,output=map(lambda p:Path(p).resolve(),(source,history,spec,output))
    if not 1<=workers<=4:raise ValueError('Use one to four workers.')
    if (output==ROOT/'artifacts' or not output.is_relative_to(ROOT/'artifacts')
            or any(output==p or output in p.parents or p in output.parents for p in (source,history))):
        raise ValueError('Use a dedicated disjoint artifact root.')
    if bool(reference)==bool(validated):raise ValueError('Supply exactly one reference validation or existing certificate.')
    cases=read(spec)['cases'];base=original.validate_cases(cases)
    for case in cases.values():
        if (base|case.get('overrides',{}))['execution_recheck_signal']:
            raise ValueError('Held-position optimization forbids signal-based position/fill rechecks.')
    names=list(cases) if names is None else list(names)
    if not names or len(names)!=len(set(names)) or not set(names)<=set(cases):raise ValueError('Invalid requested cases.')
    binding=original.input_binding(source,history)
    evidence=certificate(Path(validated).resolve()) if validated else None
    if reference:
        reference=Path(reference).resolve()
        ref=read(reference/'protocol.json')
        if output==reference or output in reference.parents or reference in output.parents:
            raise ValueError('Reference and new output must be disjoint.')
        if (ref['kind']!='causal_rule_search_v1' or ref['base_config']!=base
                or set(ref['source_files_sha256'])!=set(original.FILES)
                or ref['input_bindings_sha256']!=sha(reference/'input_bindings.json')
                or read(reference/'input_bindings.json')!=binding):
            raise ValueError('Reference uses another history, source or base configuration.')
        for name,digest in ref['source_files_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Original reference implementation changed.')
        for name in names:
            if ref['cases'].get(name)!=cases[name] or not (reference/'results'/name/'complete.json').exists():
                raise ValueError('Only already completed exact reference cases can validate this shortcut.')
            cfg=read(reference/'results'/name/'config.json')
            if (cfg['protocol_sha256']!=sha(reference/'protocol.json') or cfg['case']!=cases[name]
                    or cfg['backtest']!=asdict(original.BacktestConfig(**(base|cases[name].get('overrides',{}))))):
                raise ValueError('Reference case config is not bound to its original protocol.')
    immutable_json(output/'input_bindings.json',binding)
    protocol=dict(kind='causal_rule_held_position_fast_v1',source=str(source),history_root=str(history),
        count=binding['count'],cases=cases,base_config=base,start=START,end=END,
        input_bindings_sha256=sha(output/'input_bindings.json'),spec_sha256=sha(spec),
        source_files_sha256={name:sha(ROOT/name) for name in FILES},
        reference=str(reference) if reference else None,validated_optimization=evidence,
        optimization='Skip only dead skill/flow updates once a position exists; preserve marks, pending fills, observed volume, clocks, and known finality. No exits/reentry, online learning or signal rechecks.',
        evidence_caution='This is a computation optimization, not validation of metadata-proxy labels, historical coverage or executable liquidity.',
        frozen_files_sha256={name:sha(ROOT/name) for name in ('reports/polymarket_iclr2027/frozen/main.pdf','reports/polymarket_iclr2027/frozen/anonymous_code.zip')})
    immutable_json(output/'protocol.json',protocol)
    pending=[]
    for name in names:
        folder=output/'results'/name
        if (folder/'complete.json').exists():
            if read(folder/'config.json')['protocol_sha256']!=sha(output/'protocol.json'):
                raise ValueError('Existing case belongs to another protocol.')
        else:pending.append(name)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        tasks=[pool.submit(fast_worker,str(source),str(history),str(output),pending[i::workers])
            for i in range(workers) if pending[i::workers]]
        for task in as_completed(tasks):task.result()
    if original.input_binding(source,history)!=binding:raise ValueError('Inputs changed during run.')
    for group in ('source_files_sha256','frozen_files_sha256'):
        for name,digest in protocol[group].items():
            if sha(ROOT/name)!=digest:raise ValueError('Bound implementation or frozen artifact changed.')
    if reference:
        checks={}
        for name in names:
            folder=output/'results'/name;ref_folder=reference/'results'/name
            hashes=portfolio_equality(folder,ref_folder)
            a,b=read(folder/'complete.json'),read(ref_folder/'complete.json')
            left={k:v for k,v in a.items() if k!='elapsed_seconds'}
            right={k:v for k,v in b.items() if k!='elapsed_seconds'}
            if left!=right:raise ValueError(f'Reference summary differs: {name}')
            checks[name]=dict(portfolio_files_sha256=hashes,economic_summary_exact=True,
                output_sha256={f:sha(folder/f) for f in ('complete.json','config.json')},
                reference_sha256={f:sha(ref_folder/f) for f in ('complete.json','config.json')})
        immutable_json(output/'reference_validation.json',dict(all_byte_exact=True,reference=str(reference),
            cases=checks,source_files_sha256=protocol['source_files_sha256'],
            protocol_sha256=sha(output/'protocol.json'),reference_protocol_sha256=sha(reference/'protocol.json')))
        print(f'VALIDATED: {len(checks)} real full-inventory portfolios byte-identical',flush=True)
    completed=[n for n in cases if (output/'results'/n/'complete.json').exists()]
    dump(output/'run_manifest.json',dict(complete=len(completed)==len(cases),completed=completed,
        expected=list(cases),protocol_sha256=sha(output/'protocol.json'),finished_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','history-root','spec-file','output'):p.add_argument(f'--{arg}',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--cases',nargs='+')
    p.add_argument('--reference',type=Path);p.add_argument('--validated-fast',type=Path)
    a=p.parse_args();run(a.source,a.history_root,a.spec_file,a.output,a.workers,a.cases,a.reference,a.validated_fast)
