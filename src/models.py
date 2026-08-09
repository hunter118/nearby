from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Optional


class Side(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SELL_YES = "SELL_YES"
    SELL_NO = "SELL_NO"


class Direction(str, Enum):
    YES = "YES"
    NO = "NO"


@dataclass(frozen=True)
class Market:
    market_id: str
    question: str
    category: str
    created_at: datetime
    close_time: Optional[datetime]
    resolved_at: Optional[datetime]
    resolution: Optional[Direction]
    volume: float
    active: bool


@dataclass(frozen=True)
class TradeEvent:
    trade_id: str
    market_id: str
    trader_id: str
    side: Side
    price_yes: float
    size: float
    timestamp: datetime


@dataclass(frozen=True)
class ResolutionEvent:
    market_id: str
    resolved_at: datetime
    resolution: Direction


@dataclass
class Position:
    market_id: str
    direction: Direction
    quantity: float
    avg_entry_price: float
    opened_at: datetime


@dataclass(frozen=True)
class FilledOrder:
    market_id: str
    direction: Direction
    quantity: float
    fill_price: float
    notional: float
    fee: float
    signal_time: datetime
    filled_at: datetime
    signal_confidence: float = 0.0
    skilled_trader_count: int = 0
    directional_trader_count: int = 0
    effective_directional_traders: float = 0.0
    signal_concentration: float = 0.0
    directional_concentration: float = 0.0
    mean_expert_history_markets: float = 0.0
    mean_expert_similarity: float = 0.0
    mean_positive_history_fraction: float = 0.0
    mean_skill_score_std: float = 0.0
    semantic_risk_class: str = "standard"
    signal_token_price: float = 0.0
    order_requested_notional: float = 0.0
    source_trade_size: float = 0.0
    source_trade_side: str = ""
    child_fill_index: int = 1


@dataclass(frozen=True)
class ClosedPosition:
    market_id: str
    direction: Direction
    quantity: float
    avg_entry_price: float
    notional: float
    payout: float
    pnl: float
    opened_at: datetime
    resolved_at: datetime


@dataclass(frozen=True)
class PendingOrder:
    market_id: str
    direction: Direction
    signal_time: datetime
    execute_after: datetime
    max_notional: float
    signal_confidence: float
    allowed_entry_price: float
    signal_token_price: float = 0.0
    original_notional: float = 0.0
    fixed_quantity: float | None = None
    price_bucket: str = "stable"
    confirmation_trades_seen: int = 0
    remaining_slices: int = 1
    skilled_trader_count: int = 0
    directional_trader_count: int = 0
    effective_directional_traders: float = 0.0
    signal_concentration: float = 0.0
    directional_concentration: float = 0.0
    mean_expert_history_markets: float = 0.0
    mean_expert_similarity: float = 0.0
    mean_positive_history_fraction: float = 0.0
    mean_skill_score_std: float = 0.0
    semantic_risk_class: str = "standard"


@dataclass(frozen=True)
class TraderMarketSettlement:
    trader_id: str
    market_id: str
    score: float
    notional: float
    settled_at: datetime


EventType = Literal["trade", "resolution"]


@dataclass(frozen=True)
class TimelineEvent:
    event_type: EventType
    ts: datetime
    payload: TradeEvent | ResolutionEvent
