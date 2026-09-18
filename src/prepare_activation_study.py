"""Build causal activation timestamps and corrected full/suffix wallet labels.

Reads the exhausted public trade-window archive once. Output is a NEW research
dataset; never overwrites the expanded-universe arrays or frozen paper files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import time

import numpy as np

from build_expanded_replay import iter_selected_rows, utc
from expand_market_universe import DEFAULT as SOURCE, ROOT, START, END, read, dump, sha
from market_activation import GATES, first_activation, settlement_arrays

OUTPUT = ROOT / "artifacts/market-activation-2026-09-18"
BIDEN = "0xaa24b4f151f61324ac02d49a27f30454d3698d13cfef0463e111f835d94d0a7a"
NORMALIZATION_VERSION = 3


def trade_outcome(row, tokens, labels):
    """Anchor direction to token identity, cross-check the human outcome label.

    Some archived API rows say outcomeIndex=0 even for the independently
    identified NO token. These are retained, corrected, and counted, not dropped.
    """
    token_index = tokens.get(str(row.get('asset')))
    label = str(row.get('outcome') or '')
    label_index = labels.index(label) if label in labels else None
    if token_index is not None:
        if label_index is not None and label_index != token_index:
            raise ValueError('Token identity and outcome label disagree.')
        return token_index, row.get('outcomeIndex') not in (None, token_index)
    raise ValueError('Trade token absent from market metadata.')


def payout_record(mid, raw, manifest, complete):
    market = dict(manifest["market"])
    payout = {"YES": 1., "NO": 0.}.get(market["resolution"])
    evidence = "existing binary resolution metadata"
    if mid == BIDEN:
        payout, evidence = .5, "https://polymarket.com/event/biden-senile-during-the-debate (Final outcome: Yes 0.50, No 0.50)"
    elif raw.get("closed") and raw.get("umaResolutionStatus") == "resolved":
        prices = raw.get("outcomePrices", [])
        prices = json.loads(prices) if isinstance(prices, str) else prices
        if len(prices) == 2 and all(abs(float(p) - .5) < 1e-12 for p in prices):
            payout, evidence = .5, "archived Gamma closed + UMA resolved + outcomePrices [0.5,0.5]"
    if payout is not None and not market["resolved_at"]:
        # A final label without a known availability date is not a valid as-of
        # expert label. Do not substitute endDate/updatedAt for resolution time.
        payout = None
        evidence += '; resolution availability date unknown: no event/history label'
    # Preserve existing conservative timestamp handling, including fractional
    # markets omitted by the original binary-only normalization.
    if payout is not None and complete["last_timestamp"] is not None:
        settled = datetime.fromisoformat(market["resolved_at"])
        if utc(complete["last_timestamp"]) > settled:
            market["resolved_at"] = str(utc(complete["last_timestamp"]) + timedelta(seconds=1))
            evidence += "; effective settlement delayed to after last recorded print"
    if raw.get("closed") and payout is None and complete["rows"] and market['resolution'] is None:
        raise ValueError(f"Unmapped closed market with observable tape: {mid}")
    return market, payout, evidence


def prepare_market(item, raw, source_string, output_string):
    source, output = Path(source_string), Path(output_string)
    mid = item["market_id"]
    folder = output / "normalized" / mid
    if (folder / "manifest.json").exists():
        previous = read(folder / "manifest.json")
        if previous.get('normalization_version') == NORMALIZATION_VERSION:
            return previous
    old = read(source / "normalized" / mid / "manifest.json")
    complete = read(source / "trades" / mid / "complete.json")
    if complete["status"] != "api_window_exhausted" or (complete["start"], complete["end"]) != (START, END):
        raise ValueError("Source window incomplete.")
    market, payout, evidence = payout_record(mid, raw, old, complete)
    tokens = {str(t): i for i, t in enumerate(market["clob_token_ids"])}
    addresses, address_index = [], {}
    records, counters = [], Counter()
    for row in iter_selected_rows(source / "trades" / mid, complete):
        counters["source_rows"] += 1
        price, quantity = float(row["price"]), float(row["size"])
        if not np.isfinite(price) or not 0 <= price <= 1:
            counters["invalid_price_quarantined"] += 1
            continue
        address = str(row.get("proxyWallet") or "")
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", address) or not np.isfinite(quantity) or quantity <= 0:
            raise ValueError("Invalid wallet or quantity.")
        outcome, corrected = trade_outcome(row, tokens, market['outcome_labels'])
        counters['outcome_index_corrected'] += int(corrected)
        if outcome not in (0, 1) or row.get("side") not in ("BUY", "SELL"):
            raise ValueError("Unmapped token or direction.")
        if address not in address_index:
            address_index[address] = len(addresses)
            addresses.append(address)
        side = int(outcome) + (2 if row["side"] == "SELL" else 0)
        records.append((int(row["timestamp"]), address_index[address], side,
                        price if outcome == 0 else 1 - price, quantity, price))
        counters["zero_price_corrected"] += int(price == 0)
    times = np.array([r[0] for r in records], dtype=np.int64)
    wallet = np.array([r[1] for r in records], dtype=np.uint32)
    side = np.array([r[2] for r in records], dtype=np.uint8)
    yes_price = np.array([r[3] for r in records], dtype=float)
    sizes = np.array([r[4] for r in records], dtype=float)
    token_price = np.array([r[5] for r in records], dtype=float)
    del records, address_index
    close = (datetime.fromisoformat(market["close_time"]) - datetime(1970, 1, 1)).total_seconds() if market["close_time"] else None
    eligible = np.zeros(len(times), dtype=bool) if close is None else ((close - times > 0) & (close - times < 40 * 86400))
    arrays = dict(wallets=np.array(addresses, dtype="U42"), trade_time=times[eligible],
                  trade_wallet=wallet[eligible], trade_side=side[eligible],
                  trade_price=yes_price[eligible], trade_size=sizes[eligible])
    activations, history_counts = {}, {}
    for name in ("baseline", *GATES):
        if name == "baseline":
            keep = slice(None)
        else:
            activation = first_activation(times, sizes, yes_price, token_price, close, GATES[name], observed_until=END)
            activations[name] = activation
            keep = times >= activation["time"] if activation else np.zeros(len(times), dtype=bool)
        history = settlement_arrays(wallet[keep], side[keep], sizes[keep], yes_price[keep], payout, len(addresses))
        for key, values in history.items():
            arrays[f"{name}_history_{key}"] = values
        history_counts[name] = len(history["wallet"])
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "data.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(folder / "data.npz")
    manifest = dict(normalization_version=NORMALIZATION_VERSION, market_id=mid, rank=item["rank"], market=market, payout_yes=payout,
        payout_evidence=evidence, counters=dict(counters), activations=activations,
        settlements=history_counts, eligible_trades=int(eligible.sum()), wallets=len(addresses),
        source_manifest_sha256=sha(source / "trades" / mid / "complete.json"),
        old_normalized_sha256=old["array_sha256"], array_sha256=sha(folder / "data.npz"))
    dump(folder / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cohort, metadata = read(SOURCE / "cohort.json"), read(SOURCE / "metadata.json.gz")
    protocol = dict(source=str(SOURCE), start=START, end=END, gates={n: asdict(s) for n, s in GATES.items()},
        calibration="Exploratory grid; all variants retained. No claim of held-out tuning.",
        scopes=["flow_only", "flow_and_wallet_history"],
        trigger="End of first qualifying completed UTC hour. Only subsequent prints count.",
        history="Suffix signed incremental trading contribution, not total wallet account P&L.",
        execution="Unchanged paper primary; trade-print replay does NOT establish live fill capacity.",
        normalization_version=NORMALIZATION_VERSION,
        corrections=["fractional payouts in portfolio and wallet labels", "preserve actual zero token price",
                     "token identity + outcome label override conflicting API outcomeIndex"],
        universe="Archived volume-ranked nested universe, not point-in-time selection.",
        source_cohort_sha256=sha(SOURCE / "cohort.json"),
        reference_config_sha256=sha(ROOT / "config/paper_experiments.json"),
        frozen_pdf_sha256=sha(ROOT / "reports/polymarket_iclr2027/frozen/main.pdf"),
        frozen_zip_sha256=sha(ROOT / "reports/polymarket_iclr2027/frozen/anonymous_code.zip"))
    if (OUTPUT / "protocol.json").exists() and read(OUTPUT / "protocol.json") != protocol:
        if (OUTPUT / 'results').exists():
            raise ValueError("Existing protocol differs after replay began; use a new study.")
        if not (OUTPUT / 'protocol_before_input_audit.json').exists():
            dump(OUTPUT / 'protocol_before_input_audit.json', read(OUTPUT / 'protocol.json'))
    dump(OUTPUT / "protocol.json", protocol)
    pending = [m for m in cohort[:args.count]
               if not (OUTPUT / "normalized" / m["market_id"] / "manifest.json").exists()
               or read(OUTPUT / 'normalized' / m['market_id'] / 'manifest.json').get('normalization_version') != NORMALIZATION_VERSION]
    begun = time.monotonic()
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(prepare_market, m, metadata[m["market_id"]], str(SOURCE), str(OUTPUT)): m for m in pending}
        for i, future in enumerate(as_completed(tasks), 1):
            try:
                future.result()
            except Exception as exc:
                errors[tasks[future]["market_id"]] = repr(exc)
            if i % 50 == 0 or i == len(tasks):
                print(f"Prepared {i}/{len(tasks)}; errors {len(errors)}; {time.monotonic()-begun:.0f}s", flush=True)
                dump(OUTPUT / "preparation_progress.json", dict(count=args.count, batch_done=i, batch_total=len(tasks), errors=errors))
    dump(OUTPUT / "preparation_errors.json", errors)
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
