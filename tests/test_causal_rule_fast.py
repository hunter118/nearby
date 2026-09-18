"""Adapter/spawn and persisted-certificate boundaries, without replaying data."""
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

import run_causal_rule_fast as fast
from expand_market_universe import read, dump, sha


def _spawn_constructor_probe():
    import run_causal_rule_fast as child
    captured = []
    child.original.run_worker = lambda *args: captured.append(child.original.ObservedFinalityBacktester.__name__)
    child.fast_worker('source', 'history', 'output', ['a'])
    return captured


def test_spawn_replacement_is_local_and_visible_to_original_worker():
    before = fast.original.ObservedFinalityBacktester
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as pool:
        assert pool.submit(_spawn_constructor_probe).result(timeout=30) == ['HeldPositionFastBacktester']
    assert fast.original.ObservedFinalityBacktester is before


@pytest.fixture
def validation_run(tmp_path, monkeypatch):
    base = read(fast.ROOT/'config/paper_experiments.json')['groups']['full_window']['experiments']['tiered_position_cap_15pct']['config']
    monkeypatch.setattr(fast, 'ROOT', tmp_path)
    files = ('src/original.py', 'src/held.py', 'src/adapter.py')
    monkeypatch.setattr(fast, 'FILES', files)
    monkeypatch.setattr(fast.original, 'FILES', files[:1])
    frozen = ('reports/polymarket_iclr2027/frozen/main.pdf',
              'reports/polymarket_iclr2027/frozen/anonymous_code.zip')
    for filename in (*files, *frozen):
        path = tmp_path/filename; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
    source, history = tmp_path/'source', tmp_path/'history'
    source.mkdir(); history.mkdir()
    cases = {'a': {'history_variant': 'fixed'}, 'b': {'history_variant': 'fixed'}}
    binding = {'count': 2, 'source': str(source), 'history_root': str(history)}
    monkeypatch.setattr(fast.original, 'validate_cases', lambda values: base)
    monkeypatch.setattr(fast.original, 'input_binding', lambda s,h: binding)
    spec = tmp_path/'spec.json'; dump(spec, {'cases': cases})
    reference = tmp_path/'artifacts/reference'
    output = tmp_path/'artifacts/fast'
    dump(reference/'input_bindings.json', binding)
    ref_protocol = dict(kind='causal_rule_search_v1', base_config=base, cases=cases,
        input_bindings_sha256=sha(reference/'input_bindings.json'),
        source_files_sha256={f:sha(tmp_path/f) for f in files[:1]},
        frozen_files_sha256={f:sha(tmp_path/f) for f in frozen},
        source=str(source), history_root=str(history), count=2, start=fast.START, end=fast.END,
        spec_sha256=sha(spec))
    dump(reference/'protocol.json', ref_protocol)
    for name in cases:
        folder = reference/'results'/name
        for filename in ('positions.json', 'fills.json', 'open_positions.json', 'equity.json'):
            dump(folder/filename, [])
        dump(folder/'complete.json', {'total_return': .2, 'elapsed_seconds': 2.})
        dump(folder/'config.json', dict(case=cases[name], backtest=base,
            protocol_sha256=sha(reference/'protocol.json')))

    def fake_run(source, history, destination, names):
        dest = Path(destination)
        p = read(dest/'protocol.json')
        for name in names:
            folder = dest/'results'/name
            for filename in ('positions.json', 'fills.json', 'open_positions.json', 'equity.json'):
                dump(folder/filename, [])
            dump(folder/'complete.json', {'total_return': .2, 'elapsed_seconds': 1.})
            dump(folder/'config.json', dict(case=p['cases'][name], backtest=p['base_config'],
                protocol_sha256=sha(dest/'protocol.json')))

    class ImmediatePool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args):
            result = Future()
            try: fake_run(*args); result.set_result(None)
            except Exception as error: result.set_exception(error)
            return result
    monkeypatch.setattr(fast, 'ProcessPoolExecutor', ImmediatePool)
    fast.run(source, history, spec, output, 1, reference=reference)
    return output, reference, source, history, spec


def test_real_certificate_reuses_unchanged_matching_artifacts(validation_run):
    output, reference, *_ = validation_run
    result = fast.certificate(output)
    assert result['cases'] == ['a', 'b']
    assert read(output/'reference_validation.json')['all_byte_exact']


def test_certificate_rejects_modified_protocol(validation_run):
    output, *_ = validation_run
    protocol = read(output/'protocol.json'); protocol['kind'] = 'altered'
    dump(output/'protocol.json', protocol)
    with pytest.raises((ValueError, AssertionError)):
        fast.certificate(output)


@pytest.mark.parametrize('which', ['fast', 'reference'])
def test_certificate_rejects_modified_economic_summary(validation_run, which):
    output, reference, *_ = validation_run
    folder = (output if which=='fast' else reference)/'results/a'
    summary = read(folder/'complete.json'); summary['total_return'] = 999.
    dump(folder/'complete.json', summary)
    with pytest.raises((ValueError, AssertionError)):
        fast.certificate(output)


def test_certificate_rejects_incomplete_implementation_binding(validation_run):
    output, *_ = validation_run
    certificate = read(output/'reference_validation.json')
    certificate['source_files_sha256'].pop('src/held.py')
    dump(output/'reference_validation.json', certificate)
    with pytest.raises((ValueError, AssertionError)):
        fast.certificate(output)


def test_reference_validation_refuses_modified_config_binding(validation_run):
    output, reference, source, history, spec = validation_run
    path = reference/'results/a/config.json'; config = read(path)
    config['protocol_sha256'] = 'changed'; dump(path, config)
    with pytest.raises((ValueError, AssertionError)):
        fast.run(source, history, spec, output.parent/'other', 1, reference=reference)


def test_certificate_rejects_modified_reference_input_manifest(validation_run):
    output, reference, *_ = validation_run
    document = read(reference/'input_bindings.json'); document['count'] = 999
    dump(reference/'input_bindings.json', document)
    with pytest.raises((ValueError, AssertionError)):
        fast.certificate(output)


def test_reference_refuses_missing_original_source_binding(validation_run):
    output, reference, source, history, spec = validation_run
    path = reference/'protocol.json'; protocol = read(path)
    protocol['source_files_sha256'] = {}; dump(path, protocol)
    for name in ('a', 'b'):
        cfg_path = reference/'results'/name/'config.json'; cfg = read(cfg_path)
        cfg['protocol_sha256'] = sha(path); dump(cfg_path, cfg)
    with pytest.raises((ValueError, AssertionError)):
        fast.run(source, history, spec, output.parent/'other', 1, reference=reference)


def test_reference_summary_key_sets_must_match_even_when_value_is_null(validation_run):
    output, reference, source, history, spec = validation_run
    path = reference/'results/a/complete.json'; summary = read(path)
    summary['extra_null_field'] = None; dump(path, summary)
    with pytest.raises((ValueError, AssertionError)):
        fast.run(source, history, spec, output.parent/'other', 1, reference=reference)
