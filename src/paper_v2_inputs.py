"""Portable, content-addressed prepared inputs for the second paper version.

The export command verifies the original bound research inputs once, then
creates a NEW manifest. It never rewrites the original manifests or weakens
their validation. Prepared arrays contain the full observed trade tape, not
just paper tables. A synthetic demo is a software check, not empirical data.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

import numpy as np

from backtest.engine import BacktestConfig
from causal_rule_engine import MaturityRule
from prepare_causal_expert_history import VARIANTS, build_histories
from run_similarity_kernel_study import Kernel

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = "semantic_expert_portable_inputs_v1"
SPEC_SCHEMA = "semantic_expert_portable_spec_v1"
MARKET_PATTERN = re.compile(r"[a-zA-Z0-9_-]+")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_files(root=ROOT):
    """Static local import closure, including function-local export imports.

    Binding the complete closure catches altered transitive algorithms; it does
    not pretend that a checksum proves the economics or the metadata assumptions.
    """
    root = Path(root)
    pending = [root / "src/paper_v2_inputs.py", root / "src/paper_v2_replay.py"]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise ValueError(f"Missing implementation file: {path.relative_to(root)}")
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([x.name for x in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                       else [])
            for module in modules:
                if not module:
                    continue
                target = root / "src" / (module.replace(".", "/") + ".py")
                if target.is_file():
                    pending.append(target)
                    for parent in target.parents:
                        if parent == root / "src":
                            break
                        init = parent / "__init__.py"
                        if init.is_file():
                            pending.append(init)
                package = root / "src" / module.replace(".", "/") / "__init__.py"
                if package.is_file():
                    pending.append(package)
    return sorted(str(path.relative_to(root)) for path in found)


def implementation_binding(root=ROOT):
    return {name: sha(Path(root) / name) for name in implementation_files(root)}


def relative_file(root, name):
    pure = PurePosixPath(name)
    if not isinstance(name, str) or pure.is_absolute() or ".." in pure.parts or str(pure) != name:
        raise ValueError("Manifest paths must be canonical relative paths.")
    root = Path(root).resolve()
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Input symlink escapes the portable directory.")
    return path


def require_new_output(output, protected=()):
    output = Path(output).resolve()
    for other in protected:
        other = Path(other).resolve()
        if output == other or output in other.parents or other in output.parents:
            raise ValueError("Output must be disjoint from input directories.")
    if output.exists():
        raise ValueError("Output already exists; use a new directory, never overwrite a run.")
    return output


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer UTC second.")
    return value


def seal_inputs(root, market_order, start, end, source_identity, *, kind="prepared_observed_tape"):
    """Write a portable manifest after all array/metadata files exist."""
    root = Path(root)
    files = {"vectors.npz": sha(root / "vectors.npz")}
    for mid in market_order:
        for suffix in ("market.json", "data.npz"):
            name = f"normalized/{mid}/{suffix}"
            files[name] = sha(relative_file(root, name))
    manifest = dict(schema=INPUT_SCHEMA, kind=kind, start=start, end=end,
        market_order=market_order, market_order_policy="ConditionId lexicographic; stable original order within market and timestamp.",
        source_identity=source_identity, source_identity_sha256=digest(source_identity),
        implementation_sha256=implementation_binding(), files_sha256=files,
        scope="Metadata-proxy label availability; archived text/deadline fields; quantity-unconstrained delayed observed-price simulation. No live fill-capacity claim.")
    write(root / "manifest.json", manifest)
    return manifest


def verify_inputs(root, *, expected_identity=None):
    """Verify every consumed byte and code dependency before loading arrays."""
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("Unsupported portable input schema.")
    start, end = (_integer(manifest[k], k) for k in ("start", "end"))
    if start >= end:
        raise ValueError("Observation start must precede end.")
    ids = manifest["market_order"]
    if (not ids or ids != sorted(set(ids))
            or any(not isinstance(mid, str) or not MARKET_PATTERN.fullmatch(mid) for mid in ids)):
        raise ValueError("Markets require unique canonical lexicographic order.")
    required = {"vectors.npz"} | {f"normalized/{mid}/{suffix}" for mid in ids for suffix in ("market.json", "data.npz")}
    if set(manifest["files_sha256"]) != required:
        raise ValueError("Input manifest does not bind exactly the consumed files.")
    if manifest["implementation_sha256"] != implementation_binding():
        raise ValueError("Portable replay implementation binding changed.")
    if manifest["source_identity_sha256"] != digest(manifest["source_identity"]):
        raise ValueError("Source identity digest mismatch.")
    for key, value in (("start", start), ("end", end), ("market_count", len(ids))):
        if key in manifest["source_identity"] and manifest["source_identity"][key] != value:
            raise ValueError("Source identity and input inventory/window disagree.")
    if expected_identity is not None and manifest["source_identity_sha256"] != expected_identity:
        raise ValueError("Spec was prepared for another source identity.")
    for name, expected in manifest["files_sha256"].items():
        path = relative_file(root, name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Portable input checksum mismatch: {name}")
    return manifest


def validate_spec(spec, manifest):
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("Use a fully resolved portable spec, not a legacy override spec.")
    if (spec.get("start"), spec.get("end")) != (manifest["start"], manifest["end"]):
        raise ValueError("Spec observation bounds differ from the prepared inputs.")
    if spec.get("source_identity_sha256") != manifest["source_identity_sha256"]:
        raise ValueError("Spec was prepared for another source identity.")
    cases = spec.get("cases", {})
    if not cases:
        raise ValueError("No cases supplied.")
    for name, case in cases.items():
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError("Unsafe case name.")
        if set(case) != {"maturity", "history_variant", "similarity", "kernel", "planning_horizon_days", "ignore_risk_text", "backtest"}:
            raise ValueError("Each case must fully declare its algorithm and backtest configuration.")
        if set(case["backtest"]) != {f.name for f in fields(BacktestConfig)}:
            raise ValueError("Backtest configuration is not fully resolved.")
        config = BacktestConfig(**case["backtest"])
        if config.initial_balance != 10000 or not 0 <= config.target_exposure_fraction <= 1:
            raise ValueError("Replay requires the paper's unlevered 10,000 initial capital.")
        if config.trade_fee_bps < 0 or config.slippage_bps < 0 or config.delay_seconds < 1:
            raise ValueError("Negative costs or instantaneous fills are unsupported.")
        if config.entry_start_ts is not None or config.entry_end_ts is not None:
            raise ValueError("Paper portable specs do not use extra entry-time overrides.")
        rule = MaturityRule(**case["maturity"])
        if set(case["maturity"]) != {f.name for f in fields(MaturityRule)}:
            raise ValueError("Maturity configuration is not fully resolved.")
        if case["history_variant"] not in VARIANTS or case["similarity"] not in ("semantic", "uniform"):
            raise ValueError("Unsupported history or similarity mode.")
        if case["kernel"] is not None:
            Kernel(**case["kernel"])
            if case["similarity"] != "semantic":
                raise ValueError("Kernel requires semantic similarity.")
        horizon = case["planning_horizon_days"]
        if horizon is None:
            if rule.max_days_to_scheduled_close is None:
                raise ValueError("Deadline-free policy requires a fixed planning horizon.")
        elif (rule.max_days_to_scheduled_close is not None
              or not config.min_days_to_resolution < horizon < config.max_days_to_resolution):
            raise ValueError("Planning horizon conflicts with maturity or sizing bounds.")
        if case["ignore_risk_text"] and case["similarity"] != "uniform":
            raise ValueError("Text-independent policy requires uniform scoring.")
    return cases


def export_inputs(source, history_root, output, mode="copy"):
    from run_causal_rule_search import input_binding
    from expand_market_universe import START, END

    source, history_root = Path(source).resolve(), Path(history_root).resolve()
    output = require_new_output(output, (source, history_root))
    if mode not in ("hardlink", "copy"):
        raise ValueError("Export mode must be hardlink or copy.")
    print("Verifying original research input bindings before export...", flush=True)
    original = input_binding(source, history_root)
    copy = os.link if mode == "hardlink" else shutil.copyfile
    output.mkdir(parents=True)
    copy(source / "vectors.npz", output / "vectors.npz")
    for i, mid in enumerate(original["market_order"]):
        old = history_root / "normalized" / mid
        target = output / "normalized" / mid
        target.mkdir(parents=True)
        m = read(old / "manifest.json")
        provenance = m["binding"]["provenance"]
        if (provenance["observation_start"], provenance["observation_end"], m["source_observed_until"]) != (START, END, END):
            raise ValueError("Prepared market uses different observation bounds.")
        row = {key: m[key] for key in ("market_id", "label_usable", "label_available_at", "raw_resolution_boundary", "payout_yes", "evidence_tier", "full_valid_prints")}
        row.update(market=provenance["source_market"], observation_start=START,
            observation_end=END, array_sha256=m["array_sha256"],
            provenance=dict(original_manifest_sha256=sha(old / "manifest.json"),
                raw_complete_sha256=provenance["raw_complete_sha256"],
                original_array_sha256=m["array_sha256"]))
        write(target / "market.json", row)
        copy(old / "data.npz", target / "data.npz")
        if (i + 1) % 5000 == 0:
            print(f"Exported {i + 1}/{original['count']} prepared markets", flush=True)
    identity = dict(cohort_sha256=original["cohort_sha256"], vectors_sha256=original["vectors_sha256"],
        historical_completion_sha256=original["historical_completion_sha256"],
        market_count=original["count"], start=START, end=END)
    manifest = seal_inputs(output, original["market_order"], START, END, identity)
    verify_inputs(output)
    print(f"Portable export complete: {output}; source identity {manifest['source_identity_sha256']}", flush=True)
    return manifest


def resolve_spec(old_spec, base_path, output):
    """Resolve defaults exactly once; replay never reads a hidden baseline file."""
    old = read(old_spec)
    baseline = read(base_path)["groups"]["full_window"]["experiments"]["tiered_position_cap_15pct"]["config"]
    if old.get("base_config", {}).get("sha256") and sha(base_path) != old["base_config"]["sha256"]:
        raise ValueError("Legacy base configuration checksum differs.")
    from expand_market_universe import START, END
    source_hashes = old["inputs"]["source_files_sha256"]
    identity = dict(cohort_sha256=source_hashes["cohort.json"], vectors_sha256=source_hashes["vectors.npz"],
        historical_completion_sha256=old["inputs"]["history_files_sha256"]["complete.json"],
        market_count=41891, start=START, end=END)
    cases = {}
    for name, case in old["cases"].items():
        cases[name] = dict(maturity=asdict(MaturityRule(**case["maturity"])),
            history_variant=case["history_variant"], similarity=case["similarity"],
            kernel=asdict(Kernel(**case["kernel"])) if "kernel" in case else None,
            planning_horizon_days=case.get("planning_horizon_days"),
            ignore_risk_text=case.get("ignore_risk_text", False),
            backtest=asdict(BacktestConfig(**(baseline | case.get("overrides", {})))))
    spec = dict(schema=SPEC_SCHEMA, start=START, end=END, source_identity_sha256=digest(identity),
        description=old.get("description", "Fully resolved paper replay configurations."),
        cases=cases, paper_methods=old.get("paper_methods", {}))
    validate_spec(spec, dict(start=START, end=END, source_identity_sha256=digest(identity)))
    if Path(output).exists():
        raise ValueError("Resolved spec output already exists.")
    write(output, spec)
    return spec


def demo(output):
    """Create a tiny deterministic full-tape fixture with two past experts."""
    output = require_new_output(output)
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    start = 1735689600
    end = start + 12 * 86400
    mids = ["history_a", "history_b", "target"]
    wallets = np.array(["0x" + "1" * 40, "0x" + "2" * 40])
    for j, mid in enumerate(mids):
        base = start + j * 2 * 86400
        times = np.array([base + 10, base + 3700, base + 7400, base + 7800, base + 8300], dtype=np.int64)
        sizes = np.array([2e6, 2e6, 100., 100., 10.])
        prices = np.array([.95, .95, .90, .90, .90])
        wallet = np.array([0, 1, 0, 1, 0], dtype=np.uint32)
        side = np.zeros(5, dtype=np.uint8)
        event = base + 86400
        history, audit = build_histories(times=times, wallet=wallet, side=side, size=sizes,
            yes_price=prices, token_price=prices, payout_yes=1., event_at=event,
            label_available_at=event + 1, wallet_count=2, observed_until=end)
        folder = inputs / "normalized" / mid
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "data.npz", wallets=wallets, trade_time=times,
            trade_wallet=wallet, trade_side=side, trade_price=prices, trade_token_price=prices,
            trade_size=sizes, **history)
        market = dict(market_id=mid, question="Synthetic temperature threshold", category="unknown",
            created_at=datetime.fromtimestamp(start, timezone.utc).replace(tzinfo=None).isoformat(),
            close_time=datetime.fromtimestamp(base + 10 * 86400, timezone.utc).replace(tzinfo=None).isoformat(),
            resolved_at=None, resolution=None, active=True, volume=0)
        write(folder / "market.json", dict(market_id=mid, market=market, label_usable=True,
            label_available_at=event + 1, raw_resolution_boundary=event, payout_yes=1.,
            evidence_tier="synthetic_demo", full_valid_prints=5, observation_start=start,
            observation_end=end, array_sha256=sha(folder / "data.npz"), provenance={}))
    np.savez_compressed(inputs / "vectors.npz", market_ids=np.array(mids), vectors=np.array([[1., 0.]] * 3))
    manifest = seal_inputs(inputs, mids, start, end, dict(synthetic_fixture="two_past_markets_v1"), kind="synthetic_demo_not_empirical_data")
    # Explicit fixture settings; no research baseline is needed to generate it.
    mandatory = dict(delay_seconds=300, skill_threshold=.03, consensus_threshold=.7,
        min_skilled_traders=1, max_single_trader_weight=1., min_edge=0., min_user_volume=10.,
        max_trades_per_market=1, stable_min_price=.6, lottery_min_price=0., lottery_max_price=.1,
        stable_balance_fraction=.1, lottery_lot_size=10., lottery_max_exposure_fraction=0.,
        min_days_to_resolution=5., max_days_to_resolution=40., trade_fee_bps=0., slippage_bps=10.,
        min_entry_price=0., max_entry_price=.98, dynamic_price_at_consensus=.98,
        dynamic_price_at_high_confidence=.98, dynamic_high_confidence=1., max_market_fraction=1.,
        max_balance_fraction=1., max_loss_per_trade_fraction=1., min_ticket_size=10., initial_balance=10000.,
        position_sizing="target_exposure_annualized", target_exposure_fraction=.97, cash_buffer_fraction=.05,
        min_target_order_fraction=.05, max_target_order_fraction=.5, annualized_edge_multiplier=1.,
        equity_record_interval=0, apply_market_volume_cap=False, apply_balance_cap=False,
        apply_loss_cap=False, min_directional_traders=2, min_effective_directional_traders=1.25,
        max_directional_trader_weight=.75, min_signal_mean_expert_history_markets=1.5,
        max_position_exposure_fraction=.25)
    case = dict(maturity=asdict(MaturityRule(min_observed_hours=1, min_recent_prints=1, max_days_to_scheduled_close=28)),
        history_variant="all_after_admission", similarity="semantic", kernel=asdict(Kernel(.5, 2, True)),
        planning_horizon_days=None, ignore_risk_text=False, backtest=asdict(BacktestConfig(**mandatory)))
    spec = dict(schema=SPEC_SCHEMA, start=start, end=end,
        source_identity_sha256=manifest["source_identity_sha256"], cases=dict(demo=case))
    validate_spec(spec, manifest)
    write(output / "spec.json", spec)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("export")
    for name in ("source", "history-root", "output"):
        q.add_argument(f"--{name}", type=Path, required=True)
    q.add_argument("--mode", choices=("hardlink", "copy"), default="copy",
        help="copy isolates exported bytes (default); hardlink is space-saving but both paths must remain immutable")
    q = sub.add_parser("resolve-spec")
    for name in ("spec", "base-config", "output"):
        q.add_argument(f"--{name}", type=Path, required=True)
    q = sub.add_parser("demo")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("verify")
    q.add_argument("--inputs", type=Path, required=True)
    a = p.parse_args()
    if a.command == "export":
        export_inputs(a.source, a.history_root, a.output, a.mode)
    elif a.command == "resolve-spec":
        resolve_spec(a.spec, a.base_config, a.output)
    elif a.command == "demo":
        demo(a.output)
    else:
        verify_inputs(a.inputs)
        print("All portable input and implementation checksums match.")


if __name__ == "__main__":
    main()
