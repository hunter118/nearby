"""Resumable full-inventory adapters around the unchanged activation replay.

The default is a PARTIAL inventory sensitivity study: the available 10,000
markets still come from a retrospective volume-ranked archive. No top-K filter
is applied here. Expanding that inventory does not establish its completeness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from expand_market_universe import ROOT, DEFAULT, START, END, read, dump, sha
from market_activation import ActivationSpec, CumulativeActivationSpec
from run_rule_cutoff_return import local_grid, ANCHOR, compare_anchor


OUTPUT = ROOT / 'artifacts/dynamic-universe-2026-09-18/partial-10000'
REFERENCE = ROOT / 'artifacts/rule-cutoff-return-2026-09-18/local-grid'
OLD_REFERENCE = ROOT / 'artifacts/joint-activation-2026-09-18/grid'
SOURCE_FILES = (
    'src/run_dynamic_universe_study.py', 'src/run_rule_cutoff_return.py',
    'src/run_activation_study.py', 'src/prepare_activation_study.py',
    'src/market_activation.py', 'src/build_expanded_replay.py',
    'src/expand_market_universe.py', 'src/run_research.py',
    'src/run_semantic_risk_study.py', 'src/plot_style.py', 'src/models.py',
    'src/alpha/trader_skill.py', 'src/alpha/signal.py', 'src/alpha/semantic_risk.py',
    'src/features/embeddings.py', 'src/backtest/engine.py',
    'src/data/build_dataset.py', 'src/data/polymarket_client.py',
    'src/validate_activation_study.py',
)
INVENTORY_LABEL = 'partial inventory sensitivity: archived volume-ranked 10,000 candidates'


@contextmanager
def temporary_settings(module, **settings):
    """Restore imported module state even on errors; workers are also spawned."""
    previous = {key: getattr(module, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


def configure(gates):
    if not gates:
        raise ValueError('At least one gate is required.')
    configured = {}
    for name, record in gates.items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for c in name):
            raise ValueError('Gate names must be simple identifiers without path separators.')
        cls = CumulativeActivationSpec if record.get('volume_mode') == 'cumulative' else ActivationSpec
        configured[name] = cls(**record)
    return configured


def select_cohort(cohort, count=None):
    """No volume sorting/filtering: consume supplied complete inventory order."""
    ids = [m['market_id'] for m in cohort]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Inventory must be nonempty with unique market IDs.')
    if [m['rank'] for m in cohort] != list(range(1, len(cohort)+1)):
        raise ValueError('rank must be a dense 1-based array index in supplied order.')
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(cohort)):
        raise ValueError('Diagnostic count must be between 1 and inventory length.')
    return cohort if count is None else cohort[:count]


def safe_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be disjoint directories.')
    if not output.is_relative_to(ROOT / 'artifacts') or output == ROOT / 'artifacts':
        raise ValueError('Use a dedicated new output below artifacts/.')
    if (output / 'normalized').exists() and not (output / 'protocol.json').exists():
        raise ValueError('Refusing an unbound existing normalized dataset.')
    if (output / 'protocol.json').exists() and read(output/'protocol.json').get('kind') != 'dynamic_universe_inventory_sensitivity_v1':
        raise ValueError('Refusing to write into another study output.')
    return source, output


def immutable_json(path, document):
    if path.exists():
        if read(path) != document:
            raise ValueError(f'Immutable protocol/input binding changed: {path}')
    else:
        dump(path, document)


def input_binding(source, cohort):
    """Bind exact metadata, vectors and each exhausted raw-window checkpoint."""
    with np.load(source / 'vectors.npz', allow_pickle=False) as a:
        ids = [m['market_id'] for m in cohort]
        if a['market_ids'][:len(ids)].tolist() != ids:
            raise ValueError('Embedding order does not match supplied inventory.')
        if a['vectors'].shape[1:] != (1024,) or not np.isfinite(a['vectors'][:len(ids)]).all():
            raise ValueError('Expected finite, existing 1024-dimensional BGE vectors.')
    markets = {}
    for item in cohort:
        mid = item['market_id']
        checkpoint = source / 'trades' / mid / 'complete.json'
        complete = read(checkpoint)
        if complete['status'] != 'api_window_exhausted' or (complete['start'], complete['end']) != (START, END):
            raise ValueError(f'Incomplete or unmatched source window: {mid}')
        normalized = source / 'normalized' / mid / 'manifest.json'
        m = read(normalized)
        if m['market_id'] != mid:
            raise ValueError('Source normalized metadata has a different market ID.')
        markets[mid] = dict(raw_checkpoint_sha256=sha(checkpoint),
                           metadata_manifest_sha256=sha(normalized),
                           inherited_array_sha256=m['array_sha256'])
    return dict(files_sha256={name: sha(source / name)
        for name in ('cohort.json', 'metadata.json.gz', 'vectors.npz')}, markets=markets)


def protocol(source, cohort, total_count, gates, inventory_label, diagnostic_count, bindings):
    return dict(kind='dynamic_universe_inventory_sensitivity_v1', source=str(source),
        count=len(cohort), source_inventory_count=total_count,
        diagnostic_count_requested=diagnostic_count, uses_full_supplied_inventory=diagnostic_count is None,
        inventory_label=inventory_label, gates=gates, start=START, end=END,
        source_cohort_sha256=bindings['files_sha256']['cohort.json'],
        reference_config_sha256=sha(ROOT/'config/paper_experiments.json'),
        core_config=read(ROOT/'config/paper_experiments.json')['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config'],
        source_files_sha256={name: sha(ROOT/name) for name in SOURCE_FILES},
        membership='All supplied cohort records, with no top-K or final-volume screening in this runner. A count argument is explicitly only a prefix diagnostic.',
        gate_scope='BOTH current consensus flow and settled wallet-market contributions use each market cutoff.',
        fixed='Existing BGE vectors, semantic scores, original expert/consensus filters, .8 entry floor, >5/<40-day entry rule, sizing and execution unchanged.',
        volume='Cumulative actual-token notional in available source window, never archived final Gamma volume.',
        clock='Preserve existing completed-hour activation mechanics including terminal silent-hour limitation; no live-helper substitution.',
        time='Archived scheduled close, inherited payout availability and same-second event conventions unchanged.',
        tape='Full corrected eligible tape supplies marks/fills, including prints before activation; raw history supplies gate and suffix labels.',
        selection=f'Fixed supplied set of {len(gates)} rules; retain every result. The default seven reuse the preceding 3,000-market sensitivity set. Same-window exploratory, no claim of untouched holdout.',
        limitations=['Partial or retrospective inventory remains biased even with causal admission.',
            'Unknown scheduled-date/question revisions and market discovery omissions remain.',
            'Quantity-unconstrained print-price execution does not prove liquidity capacity.'],
        frozen_files_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/'reports/polymarket_iclr2027/frozen/main.pdf',
            ROOT/'reports/polymarket_iclr2027/frozen/anonymous_code.zip')})


def prepared_valid(output, item, gates, bindings):
    from prepare_activation_study import NORMALIZATION_VERSION
    folder = output / 'normalized' / item['market_id']
    path = folder / 'manifest.json'
    if not path.exists():
        return False
    m = read(path)
    expected = bindings['markets'][item['market_id']]
    if (m['normalization_version'] != NORMALIZATION_VERSION
            or set(m['activations']) != set(gates)
            or m['source_manifest_sha256'] != expected['raw_checkpoint_sha256']
            or m['old_normalized_sha256'] != expected['inherited_array_sha256']
            or m['rank'] != item['rank'] or m['market_id'] != item['market_id']
            or sha(folder/'data.npz') != m['array_sha256']):
        raise ValueError(f'Existing prepared market failed provenance: {item["market_id"]}')
    return True


def prepare_one(item, raw, source, output, gates):
    import prepare_activation_study as preparation
    with temporary_settings(preparation, GATES=configure(gates)):
        return preparation.prepare_market(item, raw, source, output)


def replay_chunk(source, output, gates, count, names):
    import run_activation_study as replay
    with temporary_settings(replay, SOURCE=Path(source), OUTPUT=Path(output), GATES=configure(gates)):
        replay.run_chunk(count, names)


def validate_cases(output, count, gates, allow_pending=True):
    from validate_activation_study import check_case
    base = read(output/'protocol.json')['core_config']
    complete, pending = [], []
    for gate, spec in gates.items():
        name = f'both__{gate}'
        folder = output/'results'/str(count)/name
        if not (folder/'complete.json').exists():
            pending.append(name)
            continue
        result = check_case(output, count, name, base)
        cfg = read(folder/'config.json')
        assert cfg['gate'] == spec and cfg['gate_scope'] == 'both' and cfg['history'] == gate
        complete.append(result)
    result = dict(count=count, expected=len(gates), completed=complete, pending=pending)
    dump(output/'validation.json', result)
    if pending and not allow_pending:
        raise ValueError(f'Incomplete replay cases: {pending}')
    return result


def reference_anchor_check():
    """Reuse two independently saved old 3k runs; do not repeat seven runs."""
    case = f'results/3000/both__{ANCHOR}'
    result = compare_anchor(REFERENCE/case, OLD_REFERENCE/case,
                            REFERENCE/'protocol.json', OLD_REFERENCE/'protocol.json')
    prior_source = read(REFERENCE.parent/'study_protocol.json')['source_files_sha256']
    shared = sorted(set(SOURCE_FILES) & set(prior_source) - {'src/run_rule_cutoff_return.py'})
    for name in shared:
        assert sha(ROOT/name) == prior_source[name], f'Original execution source changed: {name}'
    result['unchanged_shared_source_files'] = shared
    result['reference_complete_sha256'] = sha(REFERENCE/case/'complete.json')
    return result


def reference_input_check(output, cohort, gates):
    """Exact 3k normalized overlap confirms no preparation-mechanics drift."""
    old_gates = read(REFERENCE/'protocol.json')['gates']
    names = set(gates) & set(old_gates)
    if not names:
        return dict(markets=0, reason='No gate shared with saved 3k reference.')
    assert all(gates[n] == old_gates[n] for n in names)
    checked, byte_exact = 0, 0
    for item in cohort:
        old_folder = REFERENCE/'normalized'/item['market_id']
        if not (old_folder/'manifest.json').exists():
            continue
        folder = output/'normalized'/item['market_id']
        a, b = read(folder/'manifest.json'), read(old_folder/'manifest.json')
        assert a['market'] == b['market'] and a['payout_yes'] == b['payout_yes']
        assert all(a['activations'][n] == b['activations'][n] for n in names)
        keys = ['wallets','trade_time','trade_wallet','trade_side','trade_price','trade_size']
        keys += [f'{n}_history_{suffix}' for n in ('baseline', *sorted(names))
                 for suffix in ('wallet','score','notional')]
        if a['array_sha256'] == b['array_sha256']:
            byte_exact += 1
        else:
            with np.load(folder/'data.npz',allow_pickle=False) as x, np.load(old_folder/'data.npz',allow_pickle=False) as y:
                assert all(np.array_equal(x[k],y[k]) for k in keys)
        checked += 1
    result = dict(markets=checked, shared_gates=sorted(names),
                  exact_common_arrays_and_wallet_histories=True, byte_exact_archives=byte_exact)
    dump(output/'reference_input_validation.json',result)
    return result


def run(source=DEFAULT, output=OUTPUT, gates=None, count=None, inventory_label=INVENTORY_LABEL,
        prepare_workers=4, replay_workers=4, audit_only=False):
    if not 1 <= prepare_workers <= 4 or not 1 <= replay_workers <= 4:
        raise ValueError('Preparation and replay are each limited to 1–4 workers.')
    source, output = safe_output(source, output)
    gates = local_grid() if gates is None else gates
    gates = {k: asdict(v) for k,v in configure(gates).items()}
    all_cohort = read(source/'cohort.json')
    cohort = select_cohort(all_cohort, count)
    started = datetime.now(timezone.utc).isoformat()
    begun = time.monotonic()
    bindings = input_binding(source, cohort)
    design = protocol(source, cohort, len(all_cohort), gates, inventory_label, count, bindings)
    immutable_json(output/'input_bindings.json', bindings)
    design['input_bindings_sha256'] = sha(output/'input_bindings.json')
    immutable_json(output/'protocol.json', design)
    anchor = reference_anchor_check()
    immutable_json(output/'reference_anchor_validation.json', anchor)
    validate_cases(output, len(cohort), gates)
    pending = [item for item in cohort if not prepared_valid(output,item,gates,bindings)]
    if audit_only and pending:
        raise ValueError('Audit-only mode requires all prepared inputs.')
    if not audit_only:
        metadata = read(source/'metadata.json.gz')
        with ProcessPoolExecutor(max_workers=prepare_workers, mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(prepare_one,item,metadata[item['market_id']],str(source),str(output),gates)
                     for item in pending]
            for i,task in enumerate(as_completed(tasks),1):
                task.result()
                if i % 500 == 0 or i == len(tasks):
                    print(f'Prepared {i}/{len(tasks)}; {time.monotonic()-begun:.1f}s',flush=True)
        del metadata
    for item in cohort:
        assert prepared_valid(output,item,gates,bindings)
    reference_inputs = reference_input_check(output,cohort,gates)
    prepared_seconds = time.monotonic()-begun
    print(f'Prepared and verified {len(cohort)}; reference overlap {reference_inputs["markets"]}; starting replays.',flush=True)
    if not audit_only:
        names = [f'both__{g}' for g in gates]
        chunks = [names[i::replay_workers] for i in range(replay_workers)]
        with ProcessPoolExecutor(max_workers=replay_workers, mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(replay_chunk,str(source),str(output),gates,len(cohort),chunk)
                     for chunk in chunks if chunk]
            for task in as_completed(tasks):
                task.result()
    validation = validate_cases(output,len(cohort),gates,allow_pending=False)
    # All bound inputs and execution code stay immutable, including on resume.
    assert input_binding(source,cohort) == bindings
    after = protocol(source,cohort,len(all_cohort),gates,inventory_label,count,bindings)
    after['input_bindings_sha256'] = sha(output/'input_bindings.json')
    assert after == design
    assert reference_anchor_check() == anchor
    manifest = dict(started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-begun,preparation_and_input_audit_seconds=prepared_seconds,
        prepare_workers=prepare_workers,replay_workers=replay_workers,audit_only=audit_only,
        protocol_sha256=sha(output/'protocol.json'),input_bindings_sha256=sha(output/'input_bindings.json'),
        source_files_sha256=design['source_files_sha256'],
        reference_anchor_exact=True,reference_inputs=reference_inputs,
        all_cases_validated=len(validation['completed'])==len(gates),
        cases={g:sha(output/'results'/str(len(cohort))/f'both__{g}'/'complete.json') for g in gates})
    dump(output/'run_manifest.json',manifest)
    print(f'COMPLETE {len(gates)} cases / {len(cohort)} supplied markets; {manifest["elapsed_seconds"]:.1f}s',flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--spec-file',type=Path,help='JSON object with gates (or directly a gate mapping).')
    parser.add_argument('--count',type=int,help='Explicit prefix diagnostic only; default uses the FULL inventory.')
    parser.add_argument('--inventory-label',default=INVENTORY_LABEL)
    parser.add_argument('--prepare-workers',type=int,default=4)
    parser.add_argument('--replay-workers',type=int,default=4)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    specs=read(args.spec_file) if args.spec_file else None
    gates=specs.get('gates',specs) if specs is not None else None
    run(args.source,args.output,gates,args.count,args.inventory_label,
        args.prepare_workers,args.replay_workers,args.audit_only)


if __name__=='__main__':
    main()
