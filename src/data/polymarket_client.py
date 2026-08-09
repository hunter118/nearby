from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
from typing import Any, Iterable, Optional

import requests


def _parse_ts(value: Any) -> Optional[datetime]:
    def _to_utc_naive(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _to_utc_naive(datetime.fromtimestamp(float(value), tz=timezone.utc))
    if isinstance(value, str):
        raw = value.replace("Z", "+00:00")
        try:
            return _to_utc_naive(datetime.fromisoformat(raw))
        except ValueError:
            # Normalize fractional seconds to 6 digits for older Python parsers.
            normalized = re.sub(r"\.(\d+)([+-]\d\d:\d\d)$", lambda m: "." + m.group(1)[:6].ljust(6, "0") + m.group(2), raw)
            try:
                return _to_utc_naive(datetime.fromisoformat(normalized))
            except ValueError:
                return None
    return None


class PolymarketClient:
    """Thin HTTP client for Polymarket public endpoints."""

    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        gamma_host: str = "https://gamma-api.polymarket.com",
        data_api_host: str = "https://data-api.polymarket.com",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.gamma_host = gamma_host.rstrip("/")
        self.data_api_host = data_api_host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def fetch_markets(
        self,
        limit: int = 500,
        offset: int = 0,
        closed: bool | None = None,
        order: str | None = None,
        ascending: bool | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if closed is not None:
            params["closed"] = str(closed).lower()
        if order:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()
        response = self.session.get(
            f"{self.gamma_host}/markets",
            params=params,
            timeout=self.timeout_seconds,
        )
        if response.status_code in (400, 422):
            return []
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else data.get("data", [])

    def fetch_trades(
        self,
        market_id: str | None = None,
        limit: int = 5000,
        offset: int = 0,
        start: int | None = None,
        end: int | None = None,
        taker_only: bool = True,
    ) -> list[dict[str, Any]]:
        batch_size = min(limit, 10000)
        current_offset = offset
        result: list[dict[str, Any]] = []
        while len(result) < limit:
            params: dict[str, Any] = {"limit": batch_size, "offset": current_offset}
            if market_id:
                params["market"] = market_id
            if start is not None:
                params["start"] = int(start)
            if end is not None:
                params["end"] = int(end)
            params["takerOnly"] = str(taker_only).lower()
            response = self.session.get(
                f"{self.data_api_host}/trades",
                params=params,
                timeout=self.timeout_seconds,
            )
            retry_count = 0
            while response.status_code == 429 and retry_count < 5:
                retry_count += 1
                time.sleep(0.6 * retry_count)
                response = self.session.get(
                    f"{self.data_api_host}/trades",
                    params=params,
                    timeout=self.timeout_seconds,
                )
            if response.status_code == 400 and "offset" in response.text.lower():
                break
            response.raise_for_status()
            chunk = response.json()
            if not isinstance(chunk, list) or not chunk:
                break
            result.extend(chunk)
            if len(chunk) < batch_size:
                break
            current_offset += batch_size
            remaining = limit - len(result)
            batch_size = min(10000, remaining)
        return result[:limit]

    def fetch_market_book(self, token_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.host}/book",
            params={"token_id": token_id},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fetch_market_by_condition(self, condition_id: str) -> dict[str, Any] | None:
        # Gamma defaults to open markets when ``closed`` is omitted, so a
        # condition-id lookup must explicitly try both states.
        for closed in (False, True):
            response = self.session.get(
                f"{self.gamma_host}/markets",
                params={
                    "condition_ids": condition_id,
                    "closed": str(closed).lower(),
                    "limit": 1,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            rows = data if isinstance(data, list) else data.get("data", [])
            if rows:
                return rows[0]
        return None

    @staticmethod
    def normalize_market(raw: dict[str, Any]) -> dict[str, Any]:
        outcomes_raw = raw.get("outcomes")
        prices_raw = raw.get("outcomePrices")
        token_ids_raw = raw.get("clobTokenIds")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
        resolved_outcome: str | None = None
        if (
            isinstance(outcomes, list)
            and isinstance(prices, list)
            and len(outcomes) == 2
            and len(prices) == 2
            and raw.get("closed") is True
        ):
            try:
                p0 = float(prices[0])
                p1 = float(prices[1])
                if p0 >= 0.99 and p1 <= 0.01:
                    resolved_outcome = "YES"
                elif p1 >= 0.99 and p0 <= 0.01:
                    resolved_outcome = "NO"
            except (TypeError, ValueError):
                pass

        # ``endDate`` is the scheduled market deadline, not the time at which the
        # outcome became public.  Using it as ``resolved_at`` can make a backtest
        # learn a label before Polymarket actually closed/resolved the market.
        # Gamma currently exposes ``umaEndDate`` and/or ``closedTime`` for that
        # purpose.  Keep the scheduled deadline separately as ``close_time``.
        resolution_candidates = [
            _parse_ts(raw.get("resolvedAt")),
            _parse_ts(raw.get("umaEndDate")),
            _parse_ts(raw.get("closedTime")),
            _parse_ts(raw.get("resolveDate")),
        ]
        valid_resolution_candidates = [x for x in resolution_candidates if x is not None]
        resolved_at = max(valid_resolution_candidates) if valid_resolution_candidates else None

        return {
            "market_id": str(raw.get("conditionId") or raw.get("id") or raw.get("questionID")),
            "question": raw.get("question") or raw.get("title") or "",
            "category": raw.get("category") or "unknown",
            "created_at": _parse_ts(raw.get("createdAt") or raw.get("created_at")),
            "close_time": _parse_ts(raw.get("endDate") or raw.get("closeTime") or raw.get("end_date_iso")),
            "resolved_at": resolved_at,
            "resolution": resolved_outcome,
            "volume": float(raw.get("volume") or raw.get("volumeNum") or 0.0),
            "active": bool(raw.get("active", True)),
            # Preserve the outcome-token mapping because the Data API may emit
            # ``outcomeIndex=999``.  The asset id remains authoritative.
            "outcome_labels": [str(x) for x in outcomes] if isinstance(outcomes, list) else [],
            "clob_token_ids": [str(x) for x in token_ids] if isinstance(token_ids, list) else [],
        }

    @staticmethod
    def normalize_trade(
        raw: dict[str, Any],
        asset_outcome_index: dict[str, int] | None = None,
        outcome_labels_by_market: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        side = str(raw.get("side") or "").upper()
        outcome_index = raw.get("outcomeIndex")
        token_outcome = str(raw.get("outcome") or raw.get("token") or "").upper()
        market_id = str(raw.get("conditionId") or raw.get("market") or raw.get("marketId"))
        asset_id = str(raw.get("asset") or raw.get("asset_id") or "")

        resolved_index: int | None = None
        if outcome_index in {0, "0"}:
            resolved_index = 0
        elif outcome_index in {1, "1"}:
            resolved_index = 1
        elif asset_outcome_index and asset_id in asset_outcome_index:
            resolved_index = asset_outcome_index[asset_id]
        elif outcome_labels_by_market:
            labels = outcome_labels_by_market.get(market_id, [])
            normalized_labels = [str(label).strip().upper() for label in labels]
            if token_outcome in normalized_labels:
                resolved_index = normalized_labels.index(token_outcome)

        if resolved_index == 0 and side in {"BUY", "BID"}:
            normalized_side = "BUY_YES"
        elif resolved_index == 1 and side in {"BUY", "BID"}:
            normalized_side = "BUY_NO"
        elif resolved_index == 0 and side in {"SELL", "ASK"}:
            normalized_side = "SELL_YES"
        elif resolved_index == 1 and side in {"SELL", "ASK"}:
            normalized_side = "SELL_NO"
        elif side in {"BUY", "BID"} and token_outcome == "YES":
            normalized_side = "BUY_YES"
        elif side in {"BUY", "BID"} and token_outcome == "NO":
            normalized_side = "BUY_NO"
        elif side in {"SELL", "ASK"} and token_outcome == "YES":
            normalized_side = "SELL_YES"
        elif side in {"SELL", "ASK"} and token_outcome == "NO":
            normalized_side = "SELL_NO"
        else:
            # Silently mapping an unknown outcome to BUY_YES creates a strong,
            # systematic direction bias.  Dataset construction drops these rows
            # and reports the count instead.
            normalized_side = None

        raw_price = raw.get("price") or raw.get("priceYes") or raw.get("yesPrice") or 0.5
        price_yes = float(raw_price)
        if normalized_side and normalized_side.endswith("NO") and raw.get("priceYes") is None:
            # Some endpoints report NO token price. Convert to YES probability price.
            price_yes = 1.0 - float(raw_price)

        return {
            "trade_id": str(
                raw.get("id")
                or raw.get("tradeID")
                or raw.get("transactionHash")
                or raw.get("transaction_hash")
                or ""
            ),
            "market_id": market_id,
            "trader_id": str(
                raw.get("proxyWallet")
                or raw.get("makerAddress")
                or raw.get("takerAddress")
                or raw.get("user")
                or "unknown"
            ),
            "side": normalized_side,
            "price_yes": max(0.0, min(1.0, price_yes)),
            "size": float(raw.get("size") or raw.get("amount") or raw.get("shares") or 0.0),
            "timestamp": _parse_ts(
                raw.get("timestamp")
                or raw.get("match_time")
                or raw.get("last_update")
                or raw.get("createdAt")
                or raw.get("time")
            ),
        }

    def iter_markets(
        self,
        page_size: int = 500,
        max_pages: int = 20,
        closed: bool | None = None,
    ) -> Iterable[dict[str, Any]]:
        for page_idx in range(max_pages):
            offset = page_idx * page_size
            records = self.fetch_markets(limit=page_size, offset=offset, closed=closed)
            if not records:
                break
            for raw in records:
                yield self.normalize_market(raw)
            if len(records) < page_size:
                break
