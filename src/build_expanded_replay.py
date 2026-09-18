"""Normalize the expanded public tape and replay nested, fixed-parameter cohorts.

Source responses and per-market checkpoints are retained. No legacy 1,000-row
histories are mixed into the full-history comparison. One wallet namespace is
used throughout all markets, including the history-only expansion control.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
import gc
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from alpha.trader_skill import TraderSkillEstimator, build_trader_market_settlements_from_records
from backtest.engine import BacktestConfig, EventDrivenBacktester
from data.build_dataset import build_markets, build_resolution_events, build_timeline
from data.polymarket_client import PolymarketClient
from expand_market_universe import DEFAULT, ROOT, START, END, dump, read, sha
from features.embeddings import SimilarityConfig
from models import Direction, Side, TradeEvent, TraderMarketSettlement
from run_research import _summarize_result
from run_semantic_risk_study import _write_experiment_artifacts

SIDES = tuple(Side)
EPOCH = datetime(1970, 1, 1)
PRIMARY = "tiered_position_cap_15pct"


class ProgressBacktester(EventDrivenBacktester):
    """Logging only; dispatch every trade to the unchanged core implementation."""
    def _on_trade(self, trade):
        super()._on_trade(trade)
        self.research_processed = getattr(self, "research_processed", 0) + 1
        if self.research_processed % 250000 == 0:
            print(f"Processed {self.research_processed:,} trades; market time {trade.timestamp}", flush=True)


def utc(seconds):
    return EPOCH + timedelta(seconds=int(seconds))


def iter_selected_rows(folder, complete):
    total = 0
    # Fetch cursor goes backwards; process oldest page first. A fixed public
    # execution-field tie order makes rebuilds and nested universes deterministic.
    for page in reversed(complete["pages"]):
        source = folder / page["file"]
        if sha(source) != page["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source}")
        rows = read(source)["rows"]
        selected = ([] if page["count"] == 0 else
                    [r for r in rows if page["min_time"] <= int(r["timestamp"]) <= page["max_time"]])
        if len(selected) != page["count"]:
            raise ValueError("Selected source count differs from acquisition checkpoint")
        selected.sort(key=lambda r: (int(r["timestamp"]), str(r.get("transactionHash", "")),
                      str(r.get("asset", "")), str(r.get("proxyWallet", "")),
                      str(r.get("side", "")), float(r.get("price", 0)), float(r.get("size", 0))))
        total += len(selected)
        yield from selected
    if total != complete["rows"]:
        raise ValueError("Market trade count mismatch")


def normalize_market_tape(output_string, item, raw_metadata):
    output = Path(output_string)
    mid = item["market_id"]
    target = output / "normalized" / mid
    if (target / "manifest.json").exists():
        return read(target / "manifest.json")
    folder = output / "trades" / mid
    complete = read(folder / "complete.json")
    if complete["status"] != "api_window_exhausted" or (complete["start"], complete["end"]) != (START, END):
        raise ValueError("Incomplete or mismatched tape")
    market = PolymarketClient.normalize_market(raw_metadata)
    if market["market_id"] != mid or not market["created_at"]:
        raise ValueError("Market identity/date missing")
    tokens = market["clob_token_ids"]
    if len(market["outcome_labels"]) != 2 or (len(tokens) != 2 and complete["rows"] > 0):
        raise ValueError("Market is not binary with two identified tokens")
    # Reuse historical question text with the corresponding frozen BGE vector.
    current_question = market["question"]
    market["question"] = item["question"]
    adjusted = False
    if market["resolution"] and market["resolved_at"] and complete["last_timestamp"] is not None:
        last = utc(complete["last_timestamp"])
        if market["resolved_at"] < last:
            market["resolved_at"] = last + timedelta(seconds=1)
            adjusted = True
    mapping = {str(token): index for index, token in enumerate(tokens)}
    labels = {mid: market["outcome_labels"]}
    counters = Counter()
    eligible = []
    quarantined = []
    wallet_index = {}

    def wallet_id(wallet):
        if wallet not in wallet_index:
            wallet_index[wallet] = len(wallet_index)
        return wallet_index[wallet]

    def normalized_rows():
        for raw in iter_selected_rows(folder, complete):
            counters["source_rows"] += 1
            row = PolymarketClient.normalize_trade(raw, mapping, labels)
            if row["side"] is None:
                counters["unknown_outcome"] += 1
                continue
            if (not re.fullmatch(r"0x[0-9a-fA-F]{40}", str(raw.get("proxyWallet") or ""))
                    or not math.isfinite(row["size"]) or row["size"] <= 0):
                counters["invalid_wallet_or_size"] += 1
                continue
            price = float(raw.get("price", -1))
            if not math.isfinite(price) or not 0 <= price <= 1:
                counters["invalid_price"] += 1
                quarantined.append(dict(reason="raw outcome price outside finite [0,1]", row=raw))
                continue
            counters["normalized_rows"] += 1
            timestamp = int(raw["timestamp"])
            wait = (market["close_time"] - row["timestamp"]).total_seconds() if market["close_time"] else None
            if wait is not None and 0 < wait < 40 * 86400:
                eligible.append((timestamp, wallet_id(row["trader_id"]),
                                 SIDES.index(Side(row["side"])), row["price_yes"], row["size"]))
            yield row

    resolution = {mid: Direction(market["resolution"])} if market["resolution"] else {}
    resolution_ts = {mid: market["resolved_at"]} if market["resolved_at"] else {}
    history = build_trader_market_settlements_from_records(normalized_rows(), resolution, resolution_ts)
    for row in history:
        wallet_id(row.trader_id)
    if counters["unknown_outcome"] or counters["invalid_wallet_or_size"]:
        raise ValueError(f"Invalid trade records, do not silently omit: {dict(counters)}")
    target.mkdir(parents=True, exist_ok=True)
    if quarantined:
        dump(target / "quarantined_rows.json.gz", quarantined)
    arrays = dict(
        wallets=np.array(list(wallet_index), dtype="U42"),
        trade_time=np.array([r[0] for r in eligible], dtype=np.int64),
        trade_wallet=np.array([r[1] for r in eligible], dtype=np.uint32),
        trade_side=np.array([r[2] for r in eligible], dtype=np.uint8),
        trade_price=np.array([r[3] for r in eligible], dtype=np.float64),
        trade_size=np.array([r[4] for r in eligible], dtype=np.float64),
        history_wallet=np.array([wallet_index[r.trader_id] for r in history], dtype=np.uint32),
        history_score=np.array([r.score for r in history], dtype=np.float64),
        history_notional=np.array([r.notional for r in history], dtype=np.float64))
    temporary = target / "data.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target / "data.npz")
    manifest = dict(market_id=mid, rank=item["rank"], market=market,
                    source_rows=complete["rows"], counters=dict(counters),
                    eligible_trades=len(eligible), settlements=len(history),
                    quarantined_price_rows=len(quarantined),
                    wallets=len(wallet_index), resolution_time_adjusted=adjusted,
                    created_before_history_window=market["created_at"] < utc(START),
                    current_question=current_question, historical_question=item["question"],
                    array_sha256=sha(target / "data.npz"),
                    source_manifest_sha256=sha(folder / "complete.json"))
    dump(target / "manifest.json", manifest)
    return manifest


def normalize_available(output, workers, watch=False):
    cohort = read(output / "cohort.json")
    metadata = read(output / "metadata.json.gz")
    failed = {}
    while True:
        missing = [m for m in cohort if not (output / "normalized" / m["market_id"] / "manifest.json").exists()]
        pending = [m for m in missing if (output / "trades" / m["market_id"] / "complete.json").exists()
                   and m["market_id"] not in failed]
        if pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(normalize_market_tape, str(output), m, metadata[m["market_id"]]): m
                           for m in pending}
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        future.result()
                    except Exception as exc:
                        m = futures[future]
                        failed[m["market_id"]] = dict(rank=m["rank"], error=repr(exc))
                    if i % 25 == 0 or i == len(pending):
                        print(f"Normalized batch {i}/{len(pending)}; errors={len(failed)}", flush=True)
            dump(output / "normalization_errors.json", failed)
        remaining = [m for m in cohort if not (output / "normalized" / m["market_id"] / "manifest.json").exists()]
        dump(output / "normalization_progress.json", dict(completed=10000-len(remaining),
                                                          remaining=len(remaining), errors=failed))
        if not remaining:
            return
        if not watch or failed:
            print(f"Normalization waiting: {len(remaining)} markets not ready", flush=True)
            return
        time.sleep(20)


def load_inputs(output, history_count, trade_count):
    cohort = read(output / "cohort.json")[:history_count]
    markets, trade_chunks, history_chunks = {}, [], []
    wallet_index, wallets = {}, []
    statistics = Counter()
    manifests = []
    for item in cohort:
        mid = item["market_id"]
        folder = output / "normalized" / mid
        manifest = read(folder / "manifest.json")
        if sha(folder / "data.npz") != manifest["array_sha256"]:
            raise ValueError("Normalized array checksum mismatch")
        manifests.append(manifest)
        record = manifest["market"]
        for key in ("created_at", "close_time", "resolved_at"):
            if record[key]:
                record[key] = datetime.fromisoformat(record[key])
        markets.update(build_markets([record]))
        with np.load(folder / "data.npz", allow_pickle=False) as archive:
            local_wallets = archive["wallets"].tolist()
            indexes = []
            for wallet in local_wallets:
                if wallet not in wallet_index:
                    wallet_index[wallet] = len(wallets)
                    wallets.append(wallet)
                indexes.append(wallet_index[wallet])
            remap = np.array(indexes, dtype=np.uint32)
            if item["rank"] <= trade_count:
                trade_chunks.append(dict(
                    market=np.full(len(archive["trade_time"]), item["rank"]-1, dtype=np.uint32),
                    wallet=remap[archive["trade_wallet"]], time=archive["trade_time"],
                    side=archive["trade_side"], price=archive["trade_price"], size=archive["trade_size"]))
            if len(archive["history_wallet"]):
                history_chunks.append(dict(
                    market=np.full(len(archive["history_wallet"]), item["rank"]-1, dtype=np.uint32),
                    wallet=remap[archive["history_wallet"]], score=archive["history_score"],
                    notional=archive["history_notional"]))
        statistics.update(source_rows=manifest["source_rows"],
                          quarantined_price_rows=manifest.get("quarantined_price_rows", 0),
                          settlements=manifest["settlements"],
                          markets_created_before_history_window=int(record["created_at"] < utc(START)),
                          resolution_time_adjusted=int(manifest["resolution_time_adjusted"]))
    trade_arrays = {key: np.concatenate([chunk[key] for chunk in trade_chunks])
                    for key in ("market", "wallet", "time", "side", "price", "size")}
    history_arrays = {key: np.concatenate([chunk[key] for chunk in history_chunks])
                      for key in ("market", "wallet", "score", "notional")}
    del trade_chunks, history_chunks, wallet_index
    gc.collect()
    mids = list(markets)
    order = np.argsort(trade_arrays["time"], kind="stable")
    trades = [TradeEvent(str(i), mids[trade_arrays["market"][i]], wallets[trade_arrays["wallet"][i]],
                        SIDES[trade_arrays["side"][i]], float(trade_arrays["price"][i]),
                        float(trade_arrays["size"][i]), utc(trade_arrays["time"][i])) for i in order]
    history = [TraderMarketSettlement(wallets[w], mids[m], float(s), float(n), markets[mids[m]].resolved_at)
               for w, m, s, n in zip(history_arrays["wallet"], history_arrays["market"],
                                     history_arrays["score"], history_arrays["notional"])]
    history.sort(key=lambda r: r.settled_at)
    statistics.update(eligible_trades=len(trades), markets=len(markets), wallets=len(wallets),
                      markets_with_trades=sum(m["source_rows"] > 0 for m in manifests),
                      markets_with_eligible_trades=sum(m["eligible_trades"] > 0 for m in manifests))
    with np.load(output / "vectors.npz", allow_pickle=False) as archive:
        assert archive["market_ids"][:history_count].tolist() == mids
        vectors = archive["vectors"][:history_count]
    return markets, trades, history, vectors, dict(statistics)


def replay(output, case):
    variants = dict(markets3000_history3000=(3000, 3000),
                    markets3000_history10000=(10000, 3000),
                    markets10000_history10000=(10000, 10000))
    history_count, trade_count = variants[case]
    study=read(output / "study_manifest.json")
    if sha(ROOT / "config/paper_experiments.json") != study["reference_config_sha256"]:
        raise ValueError("Original configuration file changed after protocol freeze")
    folder = output / "results" / case
    if (folder / "complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Loading {case}", flush=True)
    markets, trades, history, vectors, statistics = load_inputs(output, history_count, trade_count)
    dump(folder / "data_statistics.json", statistics)
    timeline = build_timeline(trades, build_resolution_events(markets), utc(START), utc(END))
    estimator = TraderSkillEstimator(markets, history, similarity_config=SimilarityConfig(True, 0),
                                     similarity_mode="semantic", precomputed_market_vectors=vectors)
    reference = read(ROOT / "config/paper_experiments.json")["groups"]["full_window"]
    config = BacktestConfig(**reference["experiments"][PRIMARY]["config"])
    dump(folder / "config.json", asdict(config))
    begun = time.monotonic()
    print(f"Replaying {case}: {len(trades):,} eligible trades, {len(history):,} settlements", flush=True)
    result = ProgressBacktester(markets, timeline, estimator, config).run()
    summary = _summarize_result(case, result, config.initial_balance)
    summary.update(trade_universe=trade_count, history_universe=history_count,
                   elapsed_seconds=time.monotonic()-begun, start=utc(START), end=utc(END))
    _write_experiment_artifacts(case, result, summary, config.initial_balance, folder)
    dump(folder / "positions.json", [asdict(p) for p in result["closed_positions"]])
    dump(folder / "fills.json", [asdict(f) for f in result["fills"]])
    dump(folder / "equity.json", result["equity_curve"])
    dump(folder / "complete.json", summary)
    print(json.dumps(summary, default=str), flush=True)


def wait_for_normalized(output, count):
    cohort = read(output / "cohort.json")[:count]
    previous = None
    while True:
        pending = sum(not (output / "normalized" / m["market_id"] / "manifest.json").exists()
                      for m in cohort)
        if pending == 0:
            return
        if pending != previous:
            print(f"Waiting for normalized inputs: {count-pending}/{count}", flush=True)
            previous = pending
        failure = output / "normalization_errors.json"
        if failure.exists() and read(failure):
            raise RuntimeError("Normalization reported errors; inspect before replay")
        time.sleep(30)


def run_case_when_ready(output, case, wait):
    if wait:
        wait_for_normalized(output, 3000 if case == "markets3000_history3000" else 10000)
    replay(output, case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["normalize", "replay"])
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for all requested per-market inputs")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--case", choices=["markets3000_history3000", "markets3000_history10000", "markets10000_history10000"])
    args = parser.parse_args()
    args.output=args.output.resolve()
    if not args.output.is_relative_to((ROOT/"artifacts").resolve()) or args.output==ROOT/"artifacts":
        parser.error("Use a dedicated research artifact subdirectory")
    if args.stage == "normalize":
        normalize_available(args.output, args.workers, args.watch)
    else:
        if not args.case and not args.all_cases:
            parser.error("Replay requires an explicit --case")
        cases = (["markets3000_history3000", "markets3000_history10000", "markets10000_history10000"]
                 if args.all_cases else [args.case])
        if args.all_cases:
            with ProcessPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(run_case_when_ready,args.output,case,args.wait) for case in cases]
                for future in as_completed(futures):
                    future.result()
        else:
            run_case_when_ready(args.output,cases[0],args.wait)
        if args.all_cases:
            from report_expanded_universe import main as report
            report(args.output)


if __name__ == "__main__":
    main()
