from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from models import Direction, Market, ResolutionEvent, Side, TimelineEvent, TradeEvent


def _as_direction(value: Any) -> Direction | None:
    if value is None:
        return None
    text = str(value).upper()
    if text in {"YES", "1", "TRUE"}:
        return Direction.YES
    if text in {"NO", "0", "FALSE"}:
        return Direction.NO
    return None


def build_markets(records: Iterable[dict[str, Any]]) -> dict[str, Market]:
    markets: dict[str, Market] = {}
    for raw in records:
        if not raw.get("market_id") or not raw.get("created_at"):
            continue
        markets[raw["market_id"]] = Market(
            market_id=raw["market_id"],
            question=raw.get("question", ""),
            category=raw.get("category", "unknown"),
            created_at=raw["created_at"],
            close_time=raw.get("close_time"),
            resolved_at=raw.get("resolved_at"),
            resolution=_as_direction(raw.get("resolution")),
            volume=float(raw.get("volume", 0.0)),
            active=bool(raw.get("active", True)),
        )
    return markets


def build_trade_events(records: Iterable[dict[str, Any]]) -> list[TradeEvent]:
    events: list[TradeEvent] = []
    for raw in records:
        ts = raw.get("timestamp")
        side = raw.get("side")
        if not ts or side not in {x.value for x in Side}:
            continue
        events.append(
            TradeEvent(
                trade_id=str(raw.get("trade_id", "")),
                market_id=str(raw.get("market_id")),
                trader_id=str(raw.get("trader_id")),
                side=Side(str(side)),
                price_yes=float(raw.get("price_yes", 0.5)),
                size=float(raw.get("size", 0.0)),
                timestamp=ts,
            )
        )
    events.sort(key=lambda x: x.timestamp)
    return events


def build_resolution_events(markets: dict[str, Market]) -> list[ResolutionEvent]:
    result: list[ResolutionEvent] = []
    for market in markets.values():
        if market.resolved_at and market.resolution:
            result.append(
                ResolutionEvent(
                    market_id=market.market_id,
                    resolved_at=market.resolved_at,
                    resolution=market.resolution,
                )
            )
    result.sort(key=lambda x: x.resolved_at)
    return result


def build_timeline(
    trade_events: list[TradeEvent],
    resolution_events: list[ResolutionEvent],
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
) -> list[TimelineEvent]:
    timeline: list[TimelineEvent] = []
    for trade in trade_events:
        if start_ts and trade.timestamp < start_ts:
            continue
        if end_ts and trade.timestamp > end_ts:
            continue
        timeline.append(TimelineEvent(event_type="trade", ts=trade.timestamp, payload=trade))

    for resolution in resolution_events:
        if start_ts and resolution.resolved_at < start_ts:
            continue
        if end_ts and resolution.resolved_at > end_ts:
            continue
        timeline.append(TimelineEvent(event_type="resolution", ts=resolution.resolved_at, payload=resolution))

    timeline.sort(key=lambda x: x.ts)
    return timeline
