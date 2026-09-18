"""Return to seven hand-set rules on the full historical 3,000-market window.

This is a new exploratory research stage, not a learned cutoff, fresh holdout,
or change to the frozen ICLR results. Existing joint-activation mechanics are
called without alteration; the old 28-day anchor must reproduce exactly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time

from expand_market_universe import ROOT, START, END, read, dump, sha
from market_activation import CumulativeActivationSpec
from prepare_activation_study import SOURCE, OUTPUT as PARENT
import run_activation_joint as joint


OUTPUT = ROOT / 'artifacts/rule-cutoff-return-2026-09-18'
OLD_OUTPUT = ROOT / 'artifacts/joint-activation-2026-09-18'
STAGE = 'local-grid'
ANCHOR = joint.gate_name(1_000_000., 28., .90)
PORTFOLIO_FILES = ('positions.json', 'fills.json', 'equity.json', 'open_positions.json')
SOURCE_FILES = (
    'src/run_rule_cutoff_return.py', 'src/run_activation_joint.py',
    'src/prepare_activation_study.py', 'src/run_activation_study.py',
    'src/market_activation.py', 'src/alpha/trader_skill.py',
    'src/backtest/engine.py', 'src/models.py', 'src/data/build_dataset.py',
    'src/build_expanded_replay.py', 'src/run_research.py',
    'src/run_semantic_risk_study.py', 'src/features/embeddings.py',
    'src/validate_activation_study.py', 'src/validate_activation_joint.py',
)
REASON = (
    'User requested a return to simple manually adjusted rules without fitted '
    'cutoffs or train/test splits. Retain the previously observed p=.90, '
    'cumulative $1m, 28-day anchor and six predeclared one-axis neighbors: '
    '$750k/$1.25m, 21/35 days, .875/.925 conviction. All seven are rerun and '
    'retained. These are exploratory same-window sensitivity results; no '
    'automatic replacement of the anchor by the largest new return.'
)


def local_grid():
    """Exactly one anchor and six single-coordinate perturbations."""
    cells = (
        (1_000_000., 28., .90),
        (750_000., 28., .90), (1_250_000., 28., .90),
        (1_000_000., 21., .90), (1_000_000., 35., .90),
        (1_000_000., 28., .875), (1_000_000., 28., .925),
    )
    return {joint.gate_name(v, d, p): asdict(CumulativeActivationSpec(
        min_cumulative_notional=v, max_days_to_scheduled_close=d,
        min_conviction=p)) for v, d, p in cells}


def immutable_json(path, document):
    """Resume an identical design, never silently change its experiment."""
    if path.exists():
        if read(path) != document:
            raise ValueError(f'Immutable experiment document changed: {path}')
    else:
        dump(path, document)


def study_protocol():
    old = OLD_OUTPUT / 'grid'
    old_anchor = old / 'results/3000' / f'both__{ANCHOR}'
    gates = local_grid()
    assert read(old / 'protocol.json')['gates'][ANCHOR] == gates[ANCHOR]
    reference = read(old_anchor / 'complete.json')
    return dict(
        version=1, stage=STAGE, anchor=ANCHOR, gates=gates, selection=REASON,
        count=3000, start=START, end=END, train_test_split=None,
        scope='BOTH: each rule cuts current wallet flow AND settled historical wallet contributions at its own gate.',
        history='Raw archived trades on/after that market gate; outcome scores become visible only at its inherited effective resolution timestamp. Unknown outcomes provide no scores.',
        tape='Unchanged full corrected source tape satisfying 0 < archived scheduled-close minus print time < 40 days; gates do not remove pre-gate price marks or execution prints.',
        fixed='BGE representation, semantic skill, expert and consensus filters, .8 entry price floor, >5/<40-day entry calendar, position sizes, execution and $10,000 starting balance unchanged.',
        hourly_clock='Keep existing completed-hour implementation, including empty interior hours. It stops at the final observed print hour, so is not a fully clock-driven live API in terminal silent periods.',
        availability='Keep inherited second-precision settlement convention, including equal-second last-print/settlement ties. Do not substitute the learned-cutoff experiment +1s amendment.',
        limitations=[
            'Same-window manual sensitivity analysis, not an unbiased estimate after model or rule selection.',
            'Retrospectively fixed market universe and archived scheduled dates; point-in-time revisions are unavailable.',
            'Cumulative amount means observed actual-token dollar turnover from the archived source floor, not unseen lifetime volume.',
            'Quantity-unconstrained execution remains a historical-price scenario, not a market-depth or capacity validation.',
        ],
        old_grid_protocol_sha256=sha(old / 'protocol.json'),
        old_anchor_files_sha256={name: sha(old_anchor / name)
            for name in (*PORTFOLIO_FILES, 'config.json', 'complete.json')},
        old_anchor_summary=reference,
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            SOURCE / 'cohort.json', SOURCE / 'metadata.json.gz', SOURCE / 'vectors.npz',
            PARENT / 'protocol.json', ROOT / 'config/paper_experiments.json',
            ROOT / 'reports/polymarket_iclr2027/frozen/main.pdf',
            ROOT / 'reports/polymarket_iclr2027/frozen/anonymous_code.zip')},
        source_files_sha256={name: sha(ROOT / name) for name in SOURCE_FILES},
    )


def verified_protocol():
    """Keep the launch design immutable; permit only an explicit audit fix."""
    current = study_protocol()
    path = OUTPUT / 'study_protocol.json'
    if not path.exists():
        return current
    original = read(path)
    if original == current:
        return original
    amendment_path = OUTPUT / 'validation_amendment.json'
    if not amendment_path.exists():
        raise ValueError('Protected source/input changed without an explicit audit amendment.')
    amendment = read(amendment_path)
    wrapper = 'src/run_rule_cutoff_return.py'
    assert amendment['initial_study_protocol_sha256'] == sha(path)
    assert amendment['initial_wrapper_sha256'] == original['source_files_sha256'][wrapper]
    assert amendment['corrected_wrapper_sha256'] == current['source_files_sha256'][wrapper]
    current['source_files_sha256'][wrapper] = original['source_files_sha256'][wrapper]
    if original != current:
        raise ValueError('Audit amendment cannot change experiment inputs or execution code.')
    return original


def compare_anchor(folder, old_folder, protocol_path=None, old_protocol_path=None):
    """Economic trajectory equality, not merely similar headline returns."""
    files = {}
    for name in PORTFOLIO_FILES:
        current, previous = sha(folder / name), sha(old_folder / name)
        if current != previous:
            raise AssertionError(f'Old anchor portfolio changed: {name}')
        files[name] = current
    current, previous = read(folder / 'complete.json'), read(old_folder / 'complete.json')
    economic = lambda row: {k: v for k, v in row.items() if k != 'elapsed_seconds'}
    assert economic(current) == economic(previous), 'Old anchor economics changed.'
    current_config, old_config = read(folder / 'config.json'), read(old_folder / 'config.json')
    economic_config = lambda row: {k: v for k, v in row.items() if k != 'protocol_sha256'}
    assert economic_config(current_config) == economic_config(old_config)
    if protocol_path is not None or old_protocol_path is not None:
        assert protocol_path is not None and old_protocol_path is not None
        assert current_config['protocol_sha256'] == sha(protocol_path)
        assert old_config['protocol_sha256'] == sha(old_protocol_path)
    return dict(exact_portfolio_bytes=True, exact_summary_except_elapsed_seconds=True,
                exact_config_except_protocol_sha256=True,
                each_protocol_hash_verified=protocol_path is not None, files_sha256=files)


def run(prepare_workers=4, replay_workers=6, audit_only=False):
    if not 1 <= prepare_workers <= 6 or not 1 <= replay_workers <= 7:
        raise ValueError('Use 1–6 preparation workers and 1–7 replay workers.')
    begun = time.monotonic()
    protocol = verified_protocol()
    immutable_json(OUTPUT / 'study_protocol.json', protocol)
    # Explicit module redirection is the existing stage-runner interface. Its
    # spawned workers receive paths and specs; no source file is patched.
    if not audit_only:
        previous_output = joint.OUTPUT
        try:
            joint.OUTPUT = OUTPUT
            joint.run_stage(STAGE, local_grid(), REASON, prepare_workers, replay_workers)
        finally:
            joint.OUTPUT = previous_output
    replay_seconds = time.monotonic() - begun
    import validate_activation_joint as validation
    previous_validation_output = validation.OUTPUT
    try:
        validation.OUTPUT = OUTPUT
        result = validation.validate(STAGE, arrays=False)
    finally:
        validation.OUTPUT = previous_validation_output
    anchor = compare_anchor(OUTPUT / STAGE / 'results/3000' / f'both__{ANCHOR}',
                            OLD_OUTPUT / 'grid/results/3000' / f'both__{ANCHOR}',
                            OUTPUT / STAGE / 'protocol.json', OLD_OUTPUT / 'grid/protocol.json')
    # Input, existing code and frozen output hashes must be unchanged after run.
    assert verified_protocol() == protocol, 'A protected input/source changed during the run.'
    manifest = dict(
        completed_at=datetime.now(timezone.utc).isoformat(),
        study_protocol_sha256=sha(OUTPUT / 'study_protocol.json'),
        stage_protocol_sha256=sha(OUTPUT / STAGE / 'protocol.json'),
        source_files_sha256={name: sha(ROOT / name) for name in SOURCE_FILES},
        initial_source_files_sha256=protocol['source_files_sha256'],
        validation_amendment_sha256=sha(OUTPUT / 'validation_amendment.json')
            if (OUTPUT / 'validation_amendment.json').exists() else None,
        audit_only=audit_only,
        prepare_workers=prepare_workers, replay_workers=replay_workers,
        preparation_and_replay_seconds=None if audit_only else replay_seconds,
        elapsed_seconds=time.monotonic() - begun,
        all_case_accounting_passed=len(result['completed']) == 7 and not result['pending'],
        anchor=anchor,
        cases={name: sha(OUTPUT / STAGE / 'results/3000' / f'both__{name}' / 'complete.json')
               for name in local_grid()},
        independent_array_audit='Performed separately; see local-grid/input_validation.json and independent_audit.json when available.',
    )
    dump(OUTPUT / 'run_manifest.json', manifest)
    print(f'Completed seven cases and exact old-anchor check in {manifest["elapsed_seconds"]:.1f}s.', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-workers', type=int, default=4)
    parser.add_argument('--replay-workers', type=int, default=6)
    parser.add_argument('--audit-only', action='store_true',
                        help='Validate existing completed outputs without running any portfolio.')
    args = parser.parse_args()
    run(args.prepare_workers, args.replay_workers, args.audit_only)


if __name__ == '__main__':
    main()
