from __future__ import annotations

from dataclasses import dataclass

from models import Direction


@dataclass(frozen=True)
class FlowObservation:
    trader_id: str
    market_id: str
    direction: Direction
    volume: float
    skill: float


@dataclass(frozen=True)
class SignalDecision:
    should_trade: bool
    direction: Direction | None
    confidence: float
    total_weight: float
    skilled_trader_count: int
    max_single_trader_weight: float


def decide_direction_from_flow(
    flow: list[FlowObservation],
    skill_threshold: float,
    consensus_threshold: float,
    min_skilled_traders: int = 1,
    max_single_trader_weight: float = 1.0,
) -> SignalDecision:
    skilled = [x for x in flow if x.skill >= skill_threshold and x.volume > 0]
    if not skilled:
        return SignalDecision(False, None, 0.0, 0.0, 0, 0.0)

    unique_traders = {x.trader_id for x in skilled}
    skilled_count = len(unique_traders)
    if skilled_count < min_skilled_traders:
        return SignalDecision(False, None, 0.0, 0.0, skilled_count, 0.0)

    yes_weight = sum(x.skill * x.volume for x in skilled if x.direction == Direction.YES)
    no_weight = sum(x.skill * x.volume for x in skilled if x.direction == Direction.NO)
    total = yes_weight + no_weight
    if total <= 0:
        return SignalDecision(False, None, 0.0, 0.0, skilled_count, 0.0)

    by_trader_weight: dict[str, float] = {}
    for row in skilled:
        by_trader_weight[row.trader_id] = by_trader_weight.get(row.trader_id, 0.0) + row.skill * row.volume
    concentration = max(by_trader_weight.values()) / total if by_trader_weight else 0.0
    if concentration > max_single_trader_weight:
        return SignalDecision(False, None, 0.0, total, skilled_count, concentration)

    yes_ratio = yes_weight / total
    no_ratio = no_weight / total
    if yes_ratio >= consensus_threshold:
        return SignalDecision(True, Direction.YES, yes_ratio, total, skilled_count, concentration)
    if no_ratio >= consensus_threshold:
        return SignalDecision(True, Direction.NO, no_ratio, total, skilled_count, concentration)
    return SignalDecision(False, None, max(yes_ratio, no_ratio), total, skilled_count, concentration)
