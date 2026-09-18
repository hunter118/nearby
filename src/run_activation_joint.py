"""Exploratory cumulative-volume/deadline/price grid on the same 3k universe.

The frozen paper and original research inputs are read-only. Each stage has
its own immutable protocol and normalized arrays. A parameter change creates
a new stage, never overwrites the results used to choose it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path
import time

from expand_market_universe import ROOT, read, dump, sha
from market_activation import CumulativeActivationSpec
from prepare_activation_study import SOURCE, OUTPUT as PARENT

OUTPUT=ROOT/'artifacts/joint-activation-2026-09-18'


def gate_name(volume,days,price):
    return f'p{round(price*1000):03d}_cum{volume/1000:g}k_d{days:g}'


def initial_grid():
    return {gate_name(v,d,p):asdict(CumulativeActivationSpec(
        min_cumulative_notional=v,min_conviction=p,max_days_to_scheduled_close=d))
        for v,d,p in product((500_000.,1_000_000.,2_000_000.),(14.,28.,40.),(.85,.90,.95))}


def configure(gates):
    return {k:CumulativeActivationSpec(**v) for k,v in gates.items()}


def prepare_one(item,raw,output,gates):
    import prepare_activation_study as preparation
    preparation.GATES=configure(gates)
    return preparation.prepare_market(item,raw,str(SOURCE),output)


def replay_chunk(output,gates,names):
    import run_activation_study as replay
    replay.OUTPUT=Path(output)
    replay.GATES=configure(gates)
    replay.run_chunk(3000,names)


def run_stage(stage,gates,reason,prepare_workers=6,replay_workers=8):
    if not stage.replace('-','').isalnum():
        raise ValueError('Stage must be a simple name.')
    folder=OUTPUT/stage
    protocol=dict(gates=gates,stage=stage,count=3000,
        reference_config_sha256=sha(ROOT/'config/paper_experiments.json'),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),
        source_cohort_sha256=sha(SOURCE/'cohort.json'),
        selection=reason,
        deadline='Same archived scheduled close (Gamma endDate), NOT future actual resolution. Historical field revisions are unavailable.',
        cutoff='First completed UTC hour meeting observed cumulative dollar volume, trailing-24h price conviction, and remaining scheduled time.',
        fixed='24h observed warmup; 100 trailing-24h prints; BOTH current flow and historical wallet contributions cut. BGE, wallet filters, sizing, and execution unchanged.',
        entry='Original >5 and <40 days to scheduled close remains; a <=5-day activation has no eligible later entry.',
        volume='Observed actual-token notional since available source floor, not final or unobserved lifetime volume.',
        caution='Exploratory same-window tuning; retrospective universe and metadata; quantity-unconstrained execution.',
        frozen_pdf_sha256=sha(ROOT/'reports/polymarket_iclr2027/frozen/main.pdf'),
        frozen_zip_sha256=sha(ROOT/'reports/polymarket_iclr2027/frozen/anonymous_code.zip'))
    if (folder/'protocol.json').exists() and read(folder/'protocol.json')!=protocol:
        raise ValueError('Protocol changed. Use a new stage.')
    dump(folder/'protocol.json',protocol)
    cohort=read(SOURCE/'cohort.json')[:3000]
    metadata=read(SOURCE/'metadata.json.gz')
    pending=[m for m in cohort if not (folder/'normalized'/m['market_id']/'manifest.json').exists()]
    begun=time.monotonic()
    with ProcessPoolExecutor(max_workers=prepare_workers) as pool:
        tasks=[pool.submit(prepare_one,m,metadata[m['market_id']],str(folder),gates) for m in pending]
        for i,task in enumerate(as_completed(tasks),1):
            task.result()
            if i%250==0 or i==len(tasks):
                print(f'{stage}: prepared {i}/{len(tasks)}; {time.monotonic()-begun:.0f}s',flush=True)
    # Stripe ordered cells across workers. Each worker loads the common tape
    # once; every parameter cell receives its own estimator and portfolio.
    names=[f'both__{g}' for g in gates]
    chunks=[names[i::replay_workers] for i in range(replay_workers)]
    with ProcessPoolExecutor(max_workers=replay_workers) as pool:
        tasks=[pool.submit(replay_chunk,str(folder),gates,c) for c in chunks if c]
        for task in as_completed(tasks):
            task.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='grid')
    parser.add_argument('--spec-file',type=Path)
    parser.add_argument('--prepare-workers',type=int,default=6)
    parser.add_argument('--replay-workers',type=int,default=8)
    args=parser.parse_args()
    spec=read(args.spec_file) if args.spec_file else dict(gates=initial_grid(),
        reason='Initial 27-cell Cartesian grid: volume $0.5m/$1m/$2m, deadline horizon 14/28/40d, price conviction .85/.90/.95. Selected before this joint grid ran, informed by the previous cumulative-volume experiment. Retain all results.')
    run_stage(args.stage,spec['gates'],spec['reason'],args.prepare_workers,args.replay_workers)


if __name__=='__main__':
    main()
