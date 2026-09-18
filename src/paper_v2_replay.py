"""Replay fully resolved paper-v2 configurations from portable prepared inputs.

Reference is the default. ``--engine fast-checked`` runs BOTH engines and
publishes a same-input, same-configuration equality certificate only after their
portfolio bytes and economic summaries match. No old certificate or frozen PDF
is accepted in place of that comparison. Paths may be relocated freely.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import time

import numpy as np

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig
from causal_rule_engine import MaturityRule, ObservedFinalityBacktester, prefix_activation, replay_indices
from data.build_dataset import build_markets
from features.embeddings import SimilarityConfig
from held_position_fast import HeldPositionFastBacktester
from market_activation import FractionalResolution
from models import Side, TimelineEvent, TradeEvent, TraderMarketSettlement
from paper_v2_inputs import read, write, sha, verify_inputs, validate_spec, require_new_output
from run_research import _summarize_result
from run_similarity_kernel_study import Kernel, KernelSkillEstimator

SIDES = (Side.BUY_YES, Side.BUY_NO, Side.SELL_YES, Side.SELL_NO)
PORTFOLIO_FILES = ("positions.json", "fills.json", "open_positions.json", "equity.json", "activations.json")


def utc(epoch):
    return datetime.fromtimestamp(int(epoch), timezone.utc).replace(tzinfo=None)


def _naive_date(value):
    if value is None:
        return None
    d = datetime.fromisoformat(value)
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d


def _validate_arrays(a, row, manifest):
    keys = ("trade_time", "trade_wallet", "trade_side", "trade_price", "trade_token_price", "trade_size")
    if any(k not in a for k in (*keys, "wallets")):
        raise ValueError("Prepared tape is missing required arrays.")
    times = a["trade_time"]
    n = len(times)
    if any(a[k].ndim != 1 or len(a[k]) != n for k in keys):
        raise ValueError("Unaligned prepared trade arrays.")
    if (not np.issubdtype(times.dtype, np.integer) or np.any(np.diff(times) < 0)
            or (n and (times[0] < manifest["start"] or times[-1] > manifest["end"]))):
        raise ValueError("Tape timestamps are unordered or outside the declared observation bounds.")
    if n != row["full_valid_prints"]:
        raise ValueError("Tape count differs from market metadata.")
    wallets = a["wallets"]
    if wallets.ndim != 1 or len(set(wallets.tolist())) != len(wallets):
        raise ValueError("Invalid local wallet dictionary.")
    for key, upper in (("trade_wallet", len(wallets)), ("trade_side", 4)):
        values = a[key]
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= upper)):
            raise ValueError("Invalid trade wallet or side index.")
    for key in ("trade_price", "trade_token_price"):
        if not np.isfinite(a[key]).all() or np.any((a[key] < 0) | (a[key] > 1)):
            raise ValueError("Invalid prepared price.")
    if not np.isfinite(a["trade_size"]).all() or np.any(a["trade_size"] <= 0):
        raise ValueError("Invalid prepared quantity.")
    expected = np.where((a["trade_side"] == 0) | (a["trade_side"] == 2), a["trade_price"], 1 - a["trade_price"])
    if not np.allclose(expected, a["trade_token_price"], rtol=0, atol=1e-12):
        raise ValueError("Actual-token prices disagree with YES-equivalent prices.")


def load_inputs(root, manifest, cases):
    root = Path(root)
    addresses, address_index = [], {}
    markets, metadata, chunks, remaps = {}, [], [], []
    rules = {name: MaturityRule(**case["maturity"]) for name, case in cases.items()}
    gates = {name: [] for name in cases}
    for i, mid in enumerate(manifest["market_order"]):
        folder = root / "normalized" / mid
        row = read(folder / "market.json")
        if (row["market_id"] != mid or row["market"]["market_id"] != mid
                or (row["observation_start"], row["observation_end"]) != (manifest["start"], manifest["end"])
                or row["array_sha256"] != manifest["files_sha256"][f"normalized/{mid}/data.npz"]):
            raise ValueError("Market metadata does not bind its prepared tape or window.")
        market = dict(row["market"])
        if row["label_usable"]:
            boundary, available = row["raw_resolution_boundary"], row["label_available_at"]
            if (not isinstance(boundary, int) or not isinstance(available, int)
                    or boundary >= available or available > manifest["end"]
                    or row["payout_yes"] is None or not 0 <= row["payout_yes"] <= 1):
                raise ValueError("Invalid settlement boundary, availability or payout.")
        available = row["label_available_at"] if row["label_usable"] else None
        payout = row["payout_yes"] if row["label_usable"] else None
        market["resolved_at"] = utc(available) if available is not None else None
        market["resolution"] = "YES" if payout == 1 else "NO" if payout == 0 else None
        for key in ("created_at", "close_time"):
            market[key] = _naive_date(market[key])
        parsed = build_markets([market])
        if list(parsed) != [mid]:
            raise ValueError("Market lacks the required creation timestamp.")
        markets.update(parsed)
        metadata.append(row)
        with np.load(folder / "data.npz", allow_pickle=False) as a:
            _validate_arrays(a, row, manifest)
            local = []
            for address in a["wallets"].tolist():
                if address not in address_index:
                    address_index[address] = len(addresses)
                    addresses.append(address)
                local.append(address_index[address])
            remap = np.asarray(local, dtype=np.uint32)
            remaps.append(remap)
            times, sizes, prices, token = (a[k] for k in ("trade_time", "trade_size", "trade_price", "trade_token_price"))
            chunks.append(dict(time=times, wallet=remap[a["trade_wallet"]], side=a["trade_side"],
                price=prices, size=sizes, market=np.full(len(times), i, dtype=np.uint32)))
            close = ((market["close_time"] - datetime(1970, 1, 1)).total_seconds()
                     if market["close_time"] else None)
            cache = {}
            for name, rule in rules.items():
                if rule not in cache:
                    cache[rule] = prefix_activation(times, sizes, prices, token, rule=rule,
                        observed_until=manifest["end"], scheduled_close=close)
                gates[name].append(cache[rule])
        if (i + 1) % 5000 == 0:
            print(f"Loaded {i + 1}/{len(manifest['market_order'])} portable markets", flush=True)
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    order = np.argsort(arrays["time"], kind="stable")
    arrays = {k: values[order] for k, values in arrays.items()}
    with np.load(root / "vectors.npz", allow_pickle=False) as a:
        ids = a["market_ids"].tolist()
        values = a["vectors"]
        if (len(ids) != len(set(ids)) or set(ids) != set(markets) or values.ndim != 2
                or values.shape[0] != len(ids) or not np.isfinite(values).all()):
            raise ValueError("Embedding ids, shape or finite values do not match the inventory.")
        by_id = {mid: i for i, mid in enumerate(ids)}
        vectors = values[[by_id[mid] for mid in markets]]
    del chunks, order, address_index
    gc.collect()
    return markets, metadata, arrays, addresses, remaps, vectors, gates


def history_rows(root, variant, metadata, remaps, addresses, markets):
    rows = []
    for m, remap in zip(metadata, remaps):
        if not m["label_usable"]:
            continue
        mid = m["market_id"]
        with np.load(Path(root) / "normalized" / mid / "data.npz", allow_pickle=False) as a:
            w, s, n = (a[f"{variant}_history_{key}"] for key in ("wallet", "score", "notional"))
            if (w.ndim != 1 or s.shape != w.shape or n.shape != w.shape
                    or not np.issubdtype(w.dtype, np.integer) or np.any((w < 0) | (w >= len(remap)))
                    or not np.isfinite(s).all() or np.any(np.abs(s) > 1)
                    or not np.isfinite(n).all() or np.any(n <= 0)):
                raise ValueError("Invalid prepared wallet history arrays.")
            rows.extend(TraderMarketSettlement(addresses[wallet], mid, float(score), float(notional), markets[mid].resolved_at)
                for wallet, score, notional in zip(remap[w], s, n))
    return rows


def timeline(markets, metadata, arrays, addresses, indices, start, end):
    resolutions = [FractionalResolution(m["market_id"], markets[m["market_id"]].resolved_at, None, m["payout_yes"])
        for m in metadata if m["label_usable"] and start <= m["label_available_at"] <= end]
    resolutions.sort(key=lambda r: (r.resolved_at, r.market_id))
    cursor, ids = 0, list(markets)
    selected = [arrays[k][indices] for k in ("time", "market", "wallet", "side", "price", "size")]
    for index, epoch, m, w, s, p, q in zip(indices, *selected):
        now = utc(epoch)
        while cursor < len(resolutions) and resolutions[cursor].resolved_at <= now:
            r = resolutions[cursor]
            yield TimelineEvent("resolution", r.resolved_at, r)
            cursor += 1
        trade = TradeEvent(str(int(index)), ids[m], addresses[w], SIDES[s], float(p), float(q), now)
        yield TimelineEvent("trade", now, trade)
    for r in resolutions[cursor:]:
        yield TimelineEvent("resolution", r.resolved_at, r)


def _run_case(root, manifest, loaded, name, case, folder, engine_class):
    markets, metadata, arrays, addresses, remaps, vectors, gates = loaded
    config = BacktestConfig(**case["backtest"])
    began = time.monotonic()
    history = history_rows(root, case["history_variant"], metadata, remaps, addresses, markets)
    history_count = len(history)
    options = dict(similarity_config=SimilarityConfig(True, 0), similarity_mode=case["similarity"], precomputed_market_vectors=vectors)
    estimator = (KernelSkillEstimator(markets, history, kernel=Kernel(**case["kernel"]), **options)
                 if case["kernel"] is not None else TraderSkillEstimator(markets, history, **options))
    del history
    epochs = np.array([g["time"] if g else np.iinfo(np.int64).max for g in gates[name]], dtype=np.int64)
    indices, thinned = replay_indices(arrays["time"], arrays["market"], epochs, config)
    active = {mid: utc(t) for mid, t in zip(markets, epochs) if t != np.iinfo(np.int64).max}
    engine = engine_class(markets, timeline(markets, metadata, arrays, addresses, indices, manifest["start"], manifest["end"]),
        estimator, config, activation_times=active, planning_horizon_days=case["planning_horizon_days"],
        ignore_risk_text=case["ignore_risk_text"], observation_start=utc(manifest["start"]))
    print(f"Running {name} ({engine_class.__name__}): {len(indices):,} replay prints, {history_count:,} histories", flush=True)
    result = engine.run()
    expected = config.initial_balance + sum(p.pnl for p in result["closed_positions"]) - sum(f.fee for f in result["fills"]) + result["open_unrealized_pnl"]
    if not math.isclose(expected, result["total_equity"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Cash/inventory accounting mismatch.")
    if result["balance"] < -1e-8 or any(r["cash_balance"] < -1e-8 for r in result["equity_curve"]):
        raise ValueError("Unfunded borrowing detected.")
    if any(f.signal_time < active[f.market_id] for f in result["fills"]):
        raise ValueError("Signal before market admission.")
    if any(markets[f.market_id].resolved_at is not None and f.filled_at >= markets[f.market_id].resolved_at for f in result["fills"]):
        raise ValueError("Fill at or after observed finality.")
    summary = _summarize_result(name, result, config.initial_balance)
    summary.update(universe=len(markets), history_rows=history_count, activated_markets=len(active),
        full_valid_prints=len(arrays["time"]), iterated_prints=len(indices), thinned_inert_prints=thinned,
        post_resolution_prints_skipped=engine.post_resolution_prints_skipped,
        start=str(utc(manifest["start"])), end=str(utc(manifest["end"])),
        execution_assumption="Quantity-unconstrained delayed observed-price simulation; not a live fill-capacity estimate.",
        input_kind=manifest["kind"], elapsed_seconds=time.monotonic() - began)
    write(folder / "config.json", case)
    for key, rows in (("positions", result["closed_positions"]), ("fills", result["fills"]), ("open_positions", result["open_positions"])):
        write(folder / f"{key}.json", [asdict(x) for x in rows])
    write(folder / "equity.json", result["equity_curve"])
    write(folder / "activations.json", {mid: g for mid, g in zip(markets, gates[name]) if g})
    write(folder / "complete.json", summary)
    print(f"COMPLETE {name}: return={summary['total_return']:.8f}, DD={summary['max_drawdown']:.8f}", flush=True)
    del engine, estimator, result
    gc.collect()
    return summary


def run(inputs, spec_file, output, names=None, engine="reference"):
    inputs, spec_file = Path(inputs).resolve(), Path(spec_file).resolve()
    output = require_new_output(output, (inputs,))
    if engine not in ("reference", "fast-checked"):
        raise ValueError("Only reference or independently compared fast-checked engines are supported.")
    spec = read(spec_file)
    manifest = verify_inputs(inputs, expected_identity=spec.get("source_identity_sha256"))
    all_cases = validate_spec(spec, manifest)
    names = list(all_cases) if names is None else list(names)
    if not names or len(names) != len(set(names)) or not set(names) <= set(all_cases):
        raise ValueError("Invalid requested cases.")
    cases = {name: all_cases[name] for name in names}
    if engine == "fast-checked" and any(c["backtest"]["execution_recheck_signal"] for c in cases.values()):
        raise ValueError("Fast comparison requires execution_recheck_signal=False.")
    protocol = dict(schema="semantic_expert_portable_run_v1", engine=engine,
        input_manifest_sha256=sha(inputs / "manifest.json"), source_identity_sha256=manifest["source_identity_sha256"],
        implementation_sha256=manifest["implementation_sha256"], spec_sha256=sha(spec_file),
        cases=cases, start=manifest["start"], end=manifest["end"])
    output.mkdir(parents=True)
    write(output / "protocol.json", protocol)
    loaded = load_inputs(inputs, manifest, cases)
    checks = {}
    for name, case in cases.items():
        own = output / "results" / name
        reference = _run_case(inputs, manifest, loaded, name, case, own, ObservedFinalityBacktester)
        if engine == "fast-checked":
            optimized = output / "fast_comparison" / name
            fast = _run_case(inputs, manifest, loaded, name, case, optimized, HeldPositionFastBacktester)
            hashes = {}
            for filename in PORTFOLIO_FILES:
                if sha(own / filename) != sha(optimized / filename):
                    raise ValueError(f"Fast/reference portfolio mismatch: {name}/{filename}")
                hashes[filename] = sha(own / filename)
            if {k: v for k, v in reference.items() if k != "elapsed_seconds"} != {k: v for k, v in fast.items() if k != "elapsed_seconds"}:
                raise ValueError(f"Fast/reference economic summary mismatch: {name}")
            checks[name] = hashes
    if verify_inputs(inputs) != manifest or sha(spec_file) != protocol["spec_sha256"]:
        raise ValueError("Inputs, spec or implementation changed while replaying.")
    if checks:
        write(output / "reference_comparison.json", dict(all_byte_exact=True, cases=checks,
            protocol_sha256=sha(output / "protocol.json"), limitation="Portfolio equality is computational validation, not validation of metadata availability or executable liquidity."))
    write(output / "complete.json", dict(complete=True, cases=names,
        protocol_sha256=sha(output / "protocol.json"), finished_at=datetime.now(timezone.utc).isoformat()))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "spec", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--engine", choices=("reference", "fast-checked"), default="reference")
    a = p.parse_args()
    run(a.inputs, a.spec, a.output, a.cases, a.engine)


if __name__ == "__main__":
    main()
