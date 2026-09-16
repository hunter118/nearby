"""Portable replay inputs: numeric arrays and JSON, never executable pickle.

Retains float precision, record order (including timestamp ties), and wallet
identity equality. Wallets and unused transaction IDs become anonymous indices.
Settlement sufficient statistics replace raw histories outside the entry window.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from data.build_dataset import build_markets
from models import Side, TradeEvent, TraderMarketSettlement

EPOCH = datetime(1970, 1, 1)
SIDES = tuple(Side)


def _us(value: datetime) -> int:
    return (value - EPOCH) // timedelta(microseconds=1)


def _time(value: int) -> datetime:
    return EPOCH + timedelta(microseconds=int(value))


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_snapshot(root, markets, trades, settlements, vectors, study_manifest):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    market_index = {mid: i for i, mid in enumerate(markets)}
    wallet_index = {}
    # Number wallets by first occurrence; identity equality is all the engine uses.
    for rows in (trades, settlements):
        for row in rows:
            if row.trader_id not in wallet_index:
                wallet_index[row.trader_id] = len(wallet_index)
    side_index = {side: i for i, side in enumerate(SIDES)}
    arrays = {"vectors": np.asarray(vectors)}
    for prefix, rows in (("trade", trades), ("history", settlements)):
        arrays[prefix + "_market"] = np.fromiter(
            (market_index[x.market_id] for x in rows), dtype=np.uint32, count=len(rows))
        arrays[prefix + "_wallet"] = np.fromiter(
            (wallet_index[x.trader_id] for x in rows), dtype=np.uint32, count=len(rows))
        arrays[prefix + "_time"] = np.fromiter(
            (_us(x.timestamp if prefix == "trade" else x.settled_at) for x in rows),
            dtype=np.int64, count=len(rows))
    for name, values, dtype in (
        ("trade_side", (side_index[x.side] for x in trades), np.uint8),
        ("trade_price", (x.price_yes for x in trades), np.float64),
        ("trade_size", (x.size for x in trades), np.float64),
        ("history_score", (x.score for x in settlements), np.float64),
        ("history_notional", (x.notional for x in settlements), np.float64),
    ):
        arrays[name] = np.fromiter(values, dtype=dtype)
    archive_path = root / "replay.npz"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_LZMA
            archive.writestr(info, buffer.getvalue())
    metadata = {
        "schema_version": 1,
        "markets": [asdict(x) for x in markets.values()],
        "wallet_count": len(wallet_index),
        "trade_count": len(trades), "settlement_count": len(settlements),
        "archive_sha256": file_sha256(archive_path),
        "study_manifest": study_manifest,
        "provenance": {
            "trades": "https://data-api.polymarket.com/trades",
            "metadata": "https://gamma-api.polymarket.com/markets",
            "embeddings": "https://huggingface.co/BAAI/bge-large-en-v1.5",
            "scope": "Frozen public-feed observations, not complete on-chain wallet histories.",
            "transformations": [
                "Same normalization, overlap removal and timestamp adjustment as research runner.",
                "Trades outside the entry window summarized as wallet-market settlements.",
                "Wallet addresses replaced by consistent integer identifiers; no identity linking.",
                "Trade IDs replaced by row indices; original stable event ordering retained.",
                "Prices, quantities and history statistics retained as float64; vectors unchanged.",
            ],
        },
    }
    (root / "snapshot.json").write_text(
        json.dumps(metadata, default=lambda x: x.isoformat(), indent=2) + "\n")
    print(f"Exported {archive_path}: {archive_path.stat().st_size:,} bytes", flush=True)


def load_snapshot(root, start, end):
    root = Path(root)
    meta = json.loads((root / "snapshot.json").read_text())
    if meta["schema_version"] != 1:
        raise ValueError("Unsupported replay schema")
    archive = root / "replay.npz"
    if file_sha256(archive) != meta["archive_sha256"]:
        raise ValueError("Replay archive checksum mismatch")
    bounds = meta["study_manifest"]
    for value, key, is_lower in ((start, "requested_start_utc", True),
                                  (end, "effective_end_utc", False)):
        bound = datetime.fromisoformat(bounds[key].removesuffix("Z"))
        if (is_lower and value < bound) or (not is_lower and value > bound):
            raise ValueError("Requested window falls outside frozen replay snapshot")
    records = meta["markets"]
    for record in records:
        for key in ("created_at", "close_time", "resolved_at"):
            if record[key] is not None:
                record[key] = datetime.fromisoformat(record[key])
    markets = build_markets(records)
    mids = list(markets)
    wallets = [f"wallet_{i}" for i in range(meta["wallet_count"])]
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if len(arrays["trade_time"]) != meta["trade_count"]:
        raise ValueError("Trade count mismatch")
    if len(arrays["history_time"]) != meta["settlement_count"]:
        raise ValueError("Settlement count mismatch")
    if arrays["vectors"].shape[0] != len(mids):
        raise ValueError("Embedding alignment mismatch")
    eligible = np.flatnonzero((arrays["trade_time"] >= _us(start)) &
                             (arrays["trade_time"] <= _us(end)))
    trades = [TradeEvent(
        str(i), mids[arrays["trade_market"][i]], wallets[arrays["trade_wallet"][i]],
        SIDES[arrays["trade_side"][i]], float(arrays["trade_price"][i]),
        float(arrays["trade_size"][i]), _time(arrays["trade_time"][i])) for i in eligible]
    history = [TraderMarketSettlement(
        wallets[w], mids[m], float(s), float(n), _time(t))
        for w, m, s, n, t in zip(arrays["history_wallet"], arrays["history_market"],
                                arrays["history_score"], arrays["history_notional"],
                                arrays["history_time"])]
    return markets, trades, history, arrays["vectors"], meta
