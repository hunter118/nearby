"""Portable replay correctness and relocation tests; no empirical tape needed."""
from dataclasses import asdict
import shutil

import numpy as np
import pytest

import paper_v2_inputs as inputs
import paper_v2_replay as replay


@pytest.fixture
def prepared(tmp_path):
    root = inputs.demo(tmp_path / "fixture")
    return root, inputs.read(root / "spec.json")


def reseal(root):
    m = inputs.read(root / "inputs/manifest.json")
    return inputs.seal_inputs(root / "inputs", m["market_order"], m["start"], m["end"], m["source_identity"], kind=m["kind"])


def test_demo_runs_real_reference_and_fast_with_fills(prepared, tmp_path):
    root, spec = prepared
    assert spec["cases"]["demo"]["backtest"]["position_sizing"] == "target_exposure_annualized"
    out = replay.run(root / "inputs", root / "spec.json", tmp_path / "run", engine="fast-checked")
    summary = inputs.read(out / "results/demo/complete.json")
    assert summary["closed_positions"] == 1
    assert summary["total_return"] > 0
    assert len(inputs.read(out / "results/demo/fills.json")) == 1
    assert inputs.read(out / "reference_comparison.json")["all_byte_exact"] is True
    assert inputs.read(out / "complete.json")["complete"] is True


def test_relocation_preserves_portfolio_bytes(prepared, tmp_path):
    root, _ = prepared
    moved = tmp_path / "another location" / "fixture"
    shutil.copytree(root, moved)
    first = replay.run(root / "inputs", root / "spec.json", tmp_path / "first")
    second = replay.run(moved / "inputs", moved / "spec.json", tmp_path / "second")
    for name in replay.PORTFOLIO_FILES:
        assert inputs.sha(first / "results/demo" / name) == inputs.sha(second / "results/demo" / name)
    assert inputs.sha(first / "protocol.json") == inputs.sha(second / "protocol.json")
    assert str(tmp_path) not in (second / "protocol.json").read_text()


def test_corrupt_array_rejected_before_run(prepared, tmp_path):
    root, _ = prepared
    path = root / "inputs/normalized/target/data.npz"
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        replay.run(root / "inputs", root / "spec.json", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_missing_implementation_binding_rejected(prepared):
    root, _ = prepared
    path = root / "inputs/manifest.json"
    m = inputs.read(path)
    m["implementation_sha256"].pop("src/backtest/engine.py")
    inputs.write(path, m)
    with pytest.raises(ValueError, match="implementation"):
        inputs.verify_inputs(root / "inputs")


def test_market_order_is_explicit_and_lexicographic(prepared):
    root, _ = prepared
    path = root / "inputs/manifest.json"
    m = inputs.read(path)
    m["market_order"].reverse()
    inputs.write(path, m)
    with pytest.raises(ValueError, match="lexicographic"):
        inputs.verify_inputs(root / "inputs")


def test_wrong_source_and_bounds_rejected(prepared, tmp_path):
    root, spec = prepared
    spec["source_identity_sha256"] = "a" * 64
    inputs.write(root / "bad_source.json", spec)
    with pytest.raises(ValueError, match="source identity"):
        replay.run(root / "inputs", root / "bad_source.json", tmp_path / "run1")
    spec["source_identity_sha256"] = inputs.read(root / "spec.json")["source_identity_sha256"]
    spec["end"] += 1
    inputs.write(root / "bad_end.json", spec)
    with pytest.raises(ValueError, match="observation bounds"):
        replay.run(root / "inputs", root / "bad_end.json", tmp_path / "run2")


def test_source_identity_cannot_disagree_with_window(prepared):
    root, _ = prepared
    path = root / "inputs/manifest.json"
    m = inputs.read(path)
    m["source_identity"]["end"] = m["end"] + 1
    m["source_identity_sha256"] = inputs.digest(m["source_identity"])
    inputs.write(path, m)
    with pytest.raises(ValueError, match="inventory/window"):
        inputs.verify_inputs(root / "inputs")


def test_complete_config_required_and_signal_rechecks_reject_fast(prepared, tmp_path):
    root, spec = prepared
    spec["cases"]["demo"]["backtest"].pop("slippage_bps")
    inputs.write(root / "missing.json", spec)
    with pytest.raises(ValueError, match="fully resolved"):
        replay.run(root / "inputs", root / "missing.json", tmp_path / "missing")
    spec = inputs.read(root / "spec.json")
    spec["cases"]["demo"]["backtest"]["execution_recheck_signal"] = True
    inputs.write(root / "recheck.json", spec)
    with pytest.raises(ValueError, match="execution_recheck_signal"):
        replay.run(root / "inputs", root / "recheck.json", tmp_path / "recheck", engine="fast-checked")


def test_output_cannot_overwrite_inputs_or_existing_run(prepared, tmp_path):
    root, _ = prepared
    with pytest.raises(ValueError, match="disjoint"):
        replay.run(root / "inputs", root / "spec.json", root / "inputs/run")
    out = tmp_path / "already"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        replay.run(root / "inputs", root / "spec.json", out)


def test_manifest_path_traversal_and_escaping_symlink_rejected(prepared, tmp_path):
    root, _ = prepared
    with pytest.raises(ValueError, match="relative"):
        inputs.relative_file(root / "inputs", "../outside")
    outside = tmp_path / "outside"
    outside.write_text("test")
    link = root / "inputs/link"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        inputs.relative_file(root / "inputs", "link")


def test_array_checks_reject_out_of_window_even_after_resealing(prepared):
    root, spec = prepared
    folder = root / "inputs/normalized/target"
    with np.load(folder / "data.npz", allow_pickle=False) as a:
        arrays = {key: a[key] for key in a.files}
    arrays["trade_time"][-1] = spec["end"] + 1
    np.savez_compressed(folder / "data.npz", **arrays)
    row = inputs.read(folder / "market.json")
    row["array_sha256"] = inputs.sha(folder / "data.npz")
    inputs.write(folder / "market.json", row)
    m = reseal(root)
    inputs.verify_inputs(root / "inputs")
    with pytest.raises(ValueError, match="observation bounds"):
        replay.load_inputs(root / "inputs", m, spec["cases"])


def test_label_time_must_follow_boundary(prepared):
    root, spec = prepared
    path = root / "inputs/normalized/target/market.json"
    row = inputs.read(path)
    row["label_available_at"] = row["raw_resolution_boundary"]
    inputs.write(path, row)
    m = reseal(root)
    with pytest.raises(ValueError, match="settlement boundary"):
        replay.load_inputs(root / "inputs", m, spec["cases"])


def test_loader_history_and_timeline_equal_original_helpers(prepared, tmp_path):
    import run_causal_rule_search as original
    root, spec = prepared
    manifest = inputs.verify_inputs(root / "inputs")
    portable = replay.load_inputs(root / "inputs", manifest, spec["cases"])
    legacy = tmp_path / "legacy_fixture"
    (legacy / "normalized").mkdir(parents=True)
    inputs.write(legacy / "cohort.json", [{"market_id": mid} for mid in manifest["market_order"]])
    shutil.copyfile(root / "inputs/vectors.npz", legacy / "vectors.npz")
    for mid in manifest["market_order"]:
        src, dst = root / "inputs/normalized" / mid, legacy / "normalized" / mid
        dst.mkdir()
        row = inputs.read(src / "market.json")
        row["binding"] = {"provenance": {"source_market": row["market"]}}
        inputs.write(dst / "manifest.json", row)
        shutil.copyfile(src / "data.npz", dst / "data.npz")
    old_cases = {name: dict(maturity=c["maturity"]) for name, c in spec["cases"].items()}
    old = original.load_inputs(legacy, legacy, old_cases)
    assert portable[0] == old[0]
    assert portable[3] == old[3]
    assert portable[6] == old[6]
    for key in portable[2]:
        np.testing.assert_array_equal(portable[2][key], old[2][key])
    np.testing.assert_array_equal(portable[5], old[5])
    past = replay.history_rows(root / "inputs", "all_after_admission", portable[1], portable[4], portable[3], portable[0])
    past_old = original.history_rows(legacy, "all_after_admission", old[1], old[4], old[3], old[0])
    assert past == past_old
    indices = np.arange(len(portable[2]["time"]))
    events = list(replay.timeline(portable[0], portable[1], portable[2], portable[3], indices, manifest["start"], manifest["end"]))
    old_events = list(original.timeline(old[0], old[1], old[2], old[3], indices))
    assert [asdict(e) for e in events] == [asdict(e) for e in old_events]


def test_same_second_resolution_precedes_trade_and_ids_are_not_renumbered(prepared):
    root, spec = prepared
    m = inputs.verify_inputs(root / "inputs")
    markets, metadata, arrays, addresses, *_ = replay.load_inputs(root / "inputs", m, spec["cases"])
    arrays["time"][4] = metadata[0]["label_available_at"]
    events = list(replay.timeline(markets, metadata, arrays, addresses, np.array([4]), m["start"], m["end"]))
    assert events[0].event_type == "resolution"
    assert events[1].event_type == "trade"
    assert events[1].payload.trade_id == "4"


def test_dependency_binding_is_explicit_local_closure():
    files = inputs.implementation_files()
    assert "src/backtest/engine.py" in files
    assert "src/held_position_fast.py" in files
    assert "src/paper_v2_replay.py" in files
    assert "src/prepare_dynamic_universe_data.py" not in files
    assert "src/__init__.py" not in files
