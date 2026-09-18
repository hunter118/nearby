"""Independent 10,000-market full-history collection; never edits frozen inputs.

The old metadata snapshot determines membership and volume rank. Both nested
cohorts use the same newly collected history window. Exhausting the public API
is not a claim of complete on-chain coverage or point-in-time universe selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import threading
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/universe-expansion-2026-09-17"
OLD_MARKETS = ROOT / ".cache/backtest/markets_d5251626a43e92b1.pkl"
OLD_VECTORS = ROOT / ".cache/backtest/embeddings/market_embeddings_68de3261e02d304a.npz"
START = int(datetime(2023, 9, 18, tzinfo=timezone.utc).timestamp())
END = 1786171350
PAGE = 10000
LOCAL = threading.local()
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")).encode()
    if path.suffix == ".gz":
        payload = gzip.compress(payload, compresslevel=1, mtime=0)
    temporary.write_bytes(payload)
    temporary.replace(path)


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return json.loads(payload)


def request(url, params):
    global NEXT_REQUEST
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(7):
        try:
            with REQUEST_LOCK:
                slot = max(time.monotonic(), NEXT_REQUEST)
                NEXT_REQUEST = slot + 0.065
            if slot > time.monotonic():
                time.sleep(slot - time.monotonic())
            response = LOCAL.session.get(url, params=params, timeout=(10, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 6:
                raise
            time.sleep(min(15, 0.5 * 2 ** attempt))


def prepare(output):
    path = output / "study_manifest.json"
    if path.exists():
        manifest = read(path)
        if manifest["start_epoch"] != START or manifest["end_epoch"] != END:
            raise ValueError("Existing study window does not match")
        if sha(output / "cohort.json") != manifest["cohort_sha256"]:
            raise ValueError("Cohort changed after preparation")
        return
    # These are trusted local caches produced by this repository, never remote pickle.
    with OLD_MARKETS.open("rb") as f:
        old = pickle.load(f)
    ranked = sorted((m for m in old.values() if m.get("market_id") and m.get("created_at")),
                    key=lambda m: float(m.get("volume") or 0), reverse=True)[:10000]
    cohort = [dict(rank=i + 1, market_id=m["market_id"], question=m["question"],
                   historical_volume=m["volume"], created_at=m["created_at"],
                   historical_close_time=m.get("close_time")) for i, m in enumerate(ranked)]
    assert len(cohort) == len({x["market_id"] for x in cohort}) == 10000
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "cohort.json", cohort)
    with np.load(OLD_VECTORS, allow_pickle=True) as archive:
        index = {mid: i for i, mid in enumerate(archive["market_ids"].tolist())}
        vectors = archive["vectors"][[index[m["market_id"]] for m in cohort]]
    assert vectors.shape == (10000, 1024) and np.isfinite(vectors).all()
    np.savez(output / "vectors.npz", market_ids=np.array([m["market_id"] for m in cohort]),
             vectors=vectors)
    frozen = [ROOT / "reports/polymarket_iclr2027/frozen/main.pdf",
              ROOT / "reports/polymarket_iclr2027/frozen/anonymous_code.zip"]
    dump(path, dict(
        created_utc=datetime.now(timezone.utc).isoformat(), candidate_count=10000,
        start_epoch=START, end_epoch=END,
        start_utc=datetime.fromtimestamp(START, timezone.utc).isoformat(),
        end_utc=datetime.fromtimestamp(END, timezone.utc).isoformat(),
        cohort_sha256=sha(output / "cohort.json"), source_metadata_sha256=sha(OLD_MARKETS),
        source_vectors_sha256=sha(OLD_VECTORS), vectors_sha256=sha(output / "vectors.npz"),
        ranking="Descending volume in the existing April 24 metadata archive; no outcome filtering",
        cohorts=[3000, 10000], early_history_policy="Fetch all available API rows in common window",
        metadata_policy="Reuse frozen August metadata when available; refresh other markets",
        source_metadata=str(OLD_MARKETS.relative_to(ROOT)), taker_only=True,
        limitations=[
            "Market-scoped Data API has an approximately three-year rolling floor. Common window starts September 18, 2023; no imputed pre-window trades.",
            "Exhaustive API pagination does not certify on-chain coverage.",
            "April volume-selected universe is retrospective before selection, not point-in-time out-of-sample.",
            "Original quantity-unconstrained ICLR execution and parameters are unchanged; this is not a capacity test.",
        ], frozen_sha256={str(p.relative_to(ROOT)): sha(p) for p in frozen},
        reference_config_sha256=sha(ROOT / "config/paper_experiments.json"),
        collection_code_sha256=sha(Path(__file__))))
    print("Prepared 10,000 fixed candidates and original BGE vectors", flush=True)


def fetch_metadata(output, workers):
    cohort = read(output / "cohort.json")
    path = output / "metadata.json.gz"
    if path.exists():
        return
    with (ROOT / ".cache/research/2026-08-08/gamma_markets_raw.pkl").open("rb") as f:
        old = pickle.load(f)
    ids = {m["market_id"] for m in cohort}
    found = {mid: row for mid, row in old.items() if mid in ids}
    missing = [m["market_id"] for m in cohort if m["market_id"] not in found]
    batches = [missing[i:i + 50] for i in range(0, len(missing), 50)]

    def batch(mids):
        received = {}
        for closed in (True, False):
            key = hashlib.sha256((str(closed) + "|".join(mids)).encode()).hexdigest()
            cached = output / "metadata_batches" / f"{key}.json.gz"
            if cached.exists():
                bundle = read(cached)
            else:
                response = request("https://gamma-api.polymarket.com/markets",
                                   {"condition_ids": mids, "closed": str(closed).lower(), "limit": 100})
                rows = response.json()
                if not isinstance(rows, list) or any(x.get("conditionId") not in mids for x in rows):
                    raise ValueError("Gamma ignored condition filter")
                bundle = dict(rows=rows, url=response.url, fetched_utc=datetime.now(timezone.utc).isoformat(),
                              response_sha256=hashlib.sha256(response.content).hexdigest())
                dump(cached, bundle)
            for row in bundle["rows"]:
                received[row["conditionId"]] = row
        return received

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(batch, mids): mids for mids in batches}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                found.update(future.result())
            except Exception as exc:
                errors.append(dict(markets=futures[future], error=repr(exc)))
            if i % 10 == 0:
                print(f"Metadata: {len(found)}/10000; batches {i}/{len(batches)}", flush=True)
    missing = sorted(ids - found.keys())
    dump(output / "metadata_errors.json", dict(errors=errors, missing=missing))
    if missing or errors:
        raise RuntimeError(f"Metadata incomplete: {len(missing)} missing markets; rerun resumes batches")
    dump(path, found)
    print("Metadata: all 10,000 markets available", flush=True)


def validate_page(rows, mid, start, end, page_size):
    if not isinstance(rows, list) or len(rows) > page_size:
        raise ValueError("Invalid page shape")
    times = []
    for row in rows:
        if row.get("conditionId") != mid:
            raise ValueError("Trade market filter violated")
        stamp = int(row["timestamp"])
        if not start <= stamp <= end:
            raise ValueError("Trade time filter violated")
        times.append(stamp)
    if times != sorted(times, reverse=True):
        raise ValueError("API no longer returns descending time order")


def history_pages(query, mid, start, end, page_size=PAGE):
    """Yield disjoint time pages, retaining every row at second-level boundaries.

    A full page's oldest second is re-requested instead of advancing past it.
    Extremely dense single seconds use offset pagination and fail closed if the
    documented offset budget is exhausted. Identical rows are not deduplicated:
    several legitimate fills may share all public response fields.
    """
    cursor = end
    while cursor >= start:
        rows, source = query(start, cursor, 0, page_size)
        validate_page(rows, mid, start, cursor, page_size)
        if len(rows) < page_size:
            yield rows, source, True
            return
        oldest = min(int(x["timestamp"]) for x in rows)
        accepted = [x for x in rows if int(x["timestamp"]) > oldest]
        if accepted:
            yield accepted, source, False
            cursor = oldest
            continue
        # The complete page occupies one second. Re-query the exact second so
        # offset pagination cannot silently spill into earlier timestamps.
        offset = 0
        while True:
            chunk, source = query(oldest, oldest, offset, page_size)
            validate_page(chunk, mid, oldest, oldest, page_size)
            yield chunk, source, False
            if len(chunk) < page_size:
                break
            offset += page_size
            if offset > 10000:
                raise RuntimeError("Single-second trade volume exceeds API offset budget")
        cursor = oldest - 1


def fetch_one_market(output, item, page_size=PAGE):
    mid = item["market_id"]
    folder = output / "trades" / mid
    completed_path = folder / "complete.json"
    if completed_path.exists():
        return read(completed_path)

    def query(start, end, offset, limit):
        path = folder / f"page_{start}_{end}_{offset}_{limit}.json.gz"
        if path.exists():
            bundle = read(path)
        else:
            params = dict(market=mid, start=start, end=end, offset=offset,
                          limit=limit, takerOnly="true")
            response = request("https://data-api.polymarket.com/trades", params)
            rows = response.json()
            validate_page(rows, mid, start, end, limit)
            bundle = dict(rows=rows, params=params, url=response.url,
                          fetched_utc=datetime.now(timezone.utc).isoformat(),
                          response_sha256=hashlib.sha256(response.content).hexdigest())
            dump(path, bundle)
        return bundle["rows"], path.name

    count, pages, first, last = 0, [], None, None
    for rows, source, terminal in history_pages(query, mid, START, END, page_size):
        if rows:
            times = [int(row["timestamp"]) for row in rows]
            first = min(first if first is not None else min(times), min(times))
            last = max(last if last is not None else max(times), max(times))
        # Store selection interval because full source pages may overlap at the
        # deliberately deferred boundary second. Rebuild must select only these.
        pages.append(dict(file=source, count=len(rows),
                          min_time=min((int(x["timestamp"]) for x in rows), default=None),
                          max_time=max((int(x["timestamp"]) for x in rows), default=None),
                          sha256=sha(folder / source), terminal=terminal))
        count += len(rows)
    result = dict(market_id=mid, rank=item["rank"], rows=count, pages=pages,
                  first_timestamp=first, last_timestamp=last, start=START, end=END,
                  status="api_window_exhausted", completed_utc=datetime.now(timezone.utc).isoformat())
    dump(completed_path, result)
    return result


def fetch_trades(output, workers, limit=None):
    cohort = read(output / "cohort.json")
    if limit:
        cohort = cohort[:limit]
    done, errors, counts = 0, [], 0
    began = time.monotonic()
    # Complete the nested 3,000-market control first. Cached pages survive any
    # interrupted fetch, so concurrency can change without changing membership.
    order = sorted(cohort, key=lambda x: x["rank"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one_market, output, item): item for item in order}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                counts += result["rows"]
            except Exception as exc:
                errors.append(dict(market_id=item["market_id"], rank=item["rank"], error=repr(exc)))
            done += 1
            if done % 20 == 0 or done == len(cohort):
                progress = dict(attempted=done, total=len(cohort), successful=done-len(errors),
                                rows=counts, errors=errors, elapsed_seconds=time.monotonic()-began)
                dump(output / "collection_progress.json", progress)
                print(json.dumps({k: v for k, v in progress.items() if k != "errors"}) +
                      f"; errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} markets failed; cached pages retained for resume")


def recover_failed(output, workers):
    """Retry completed failed requests with smaller pages, never a smaller total cap."""
    progress = output / "collection_progress.json"
    errors = read(progress).get("errors", []) if progress.exists() else []
    cohort = {m["market_id"]: m for m in read(output / "cohort.json")}
    failed = [cohort[e["market_id"]] for e in errors
              if not (output / "trades" / e["market_id"] / "complete.json").exists()]
    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one_market, output, m, 1000): m for m in failed}
        for future in as_completed(futures):
            m = futures[future]
            try:
                result = future.result()
                status = dict(rank=m["rank"], market_id=m["market_id"], status="recovered", rows=result["rows"])
            except Exception as exc:
                status = dict(rank=m["rank"], market_id=m["market_id"], status="failed", error=repr(exc))
            outcomes.append(status)
            print(json.dumps(status), flush=True)
    if outcomes:
        dump(output / "recovery" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+".json"), outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "metadata", "fetch", "recover", "all"])
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Pilot only; never creates a complete-study marker")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()) or output == ROOT / "artifacts":
        parser.error("Use a dedicated artifact subdirectory")
    prepare(output)
    if args.stage in ("metadata", "all"):
        fetch_metadata(output, args.workers)
    if args.stage in ("fetch", "all"):
        fetch_trades(output, args.workers, args.limit)
    if args.stage == "recover":
        recover_failed(output, args.workers)


if __name__ == "__main__":
    main()
