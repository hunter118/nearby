from __future__ import annotations

from dataclasses import dataclass
import heapq

from models import Direction


@dataclass(frozen=True)
class FlowObservation:
    trader_id: str
    market_id: str
    direction: Direction
    volume: float
    skill: float
    effective_history_markets: float = 0.0
    mean_similarity: float = 0.0
    positive_history_weight_fraction: float = 0.0
    skill_score_std: float = 0.0


@dataclass(frozen=True)
class SignalDecision:
    should_trade: bool
    direction: Direction | None
    confidence: float
    total_weight: float
    skilled_trader_count: int
    max_single_trader_weight: float
    directional_trader_count: int = 0
    effective_directional_traders: float = 0.0
    directional_concentration: float = 0.0
    mean_expert_history_markets: float = 0.0
    mean_expert_similarity: float = 0.0
    mean_positive_history_fraction: float = 0.0
    mean_skill_score_std: float = 0.0


def _observation_weight(row: FlowObservation, weighting: str) -> float:
    if weighting == "skill_volume":
        return row.skill * row.volume
    if weighting == "volume":
        return row.volume
    if weighting == "skill":
        return row.skill
    if weighting == "equal":
        return 1.0
    raise ValueError(f"Unsupported signal weighting: {weighting}")


class FlowAccumulator:
    """Incremental equivalent of ``decide_direction_from_flow``.

    The legacy implementation rescanned every prior observation after each new
    trade, which becomes quadratic for active markets.  This accumulator keeps
    the same sufficient statistics and supports removal for rolling windows.
    """

    def __init__(
        self,
        skill_threshold: float,
        consensus_threshold: float,
        min_skilled_traders: int = 1,
        max_single_trader_weight: float = 1.0,
        weighting: str = "skill_volume",
        min_directional_traders: int = 1,
        min_effective_directional_traders: float = 1.0,
        max_directional_trader_weight: float = 1.0,
        min_expert_effective_history_markets: float = 0.0,
        min_expert_mean_similarity: float = 0.0,
        min_expert_positive_history_fraction: float = 0.0,
        max_expert_score_std: float = float("inf"),
    ) -> None:
        self.skill_threshold = skill_threshold
        self.consensus_threshold = consensus_threshold
        self.min_skilled_traders = min_skilled_traders
        self.max_single_trader_weight = max_single_trader_weight
        self.weighting = weighting
        self.min_directional_traders = min_directional_traders
        self.min_effective_directional_traders = min_effective_directional_traders
        self.max_directional_trader_weight = max_directional_trader_weight
        self.min_expert_effective_history_markets = min_expert_effective_history_markets
        self.min_expert_mean_similarity = min_expert_mean_similarity
        self.min_expert_positive_history_fraction = min_expert_positive_history_fraction
        self.max_expert_score_std = max_expert_score_std
        self.yes_weight = 0.0
        self.no_weight = 0.0
        self.by_trader_weight: dict[str, float] = {}
        self.by_trader_count: dict[str, int] = {}
        self.by_direction_trader_weight: dict[Direction, dict[str, float]] = {
            Direction.YES: {},
            Direction.NO: {},
        }
        self.direction_diagnostic_sums: dict[Direction, list[float]] = {
            Direction.YES: [0.0, 0.0, 0.0, 0.0],
            Direction.NO: [0.0, 0.0, 0.0, 0.0],
        }
        self._max_heap: list[tuple[float, str]] = []

    def _is_skilled(self, row: FlowObservation) -> bool:
        return (
            row.skill >= self.skill_threshold
            and row.volume > 0.0
            and row.effective_history_markets >= self.min_expert_effective_history_markets
            and row.mean_similarity >= self.min_expert_mean_similarity
            and row.positive_history_weight_fraction
            >= self.min_expert_positive_history_fraction
            and row.skill_score_std <= self.max_expert_score_std
        )

    def add(self, row: FlowObservation) -> None:
        if not self._is_skilled(row):
            return
        weight = _observation_weight(row, self.weighting)
        if row.direction == Direction.YES:
            self.yes_weight += weight
        else:
            self.no_weight += weight
        updated = self.by_trader_weight.get(row.trader_id, 0.0) + weight
        self.by_trader_weight[row.trader_id] = updated
        self.by_trader_count[row.trader_id] = self.by_trader_count.get(row.trader_id, 0) + 1
        direction_weights = self.by_direction_trader_weight[row.direction]
        direction_weights[row.trader_id] = direction_weights.get(row.trader_id, 0.0) + weight
        diagnostics = self.direction_diagnostic_sums[row.direction]
        diagnostics[0] += weight * row.effective_history_markets
        diagnostics[1] += weight * row.mean_similarity
        diagnostics[2] += weight * row.positive_history_weight_fraction
        diagnostics[3] += weight * row.skill_score_std
        heapq.heappush(self._max_heap, (-updated, row.trader_id))

    def remove(self, row: FlowObservation) -> None:
        if not self._is_skilled(row):
            return
        weight = _observation_weight(row, self.weighting)
        if row.direction == Direction.YES:
            self.yes_weight -= weight
        else:
            self.no_weight -= weight
        remaining_count = self.by_trader_count[row.trader_id] - 1
        updated = self.by_trader_weight[row.trader_id] - weight
        direction_weights = self.by_direction_trader_weight[row.direction]
        direction_updated = direction_weights[row.trader_id] - weight
        if abs(direction_updated) <= 1e-12:
            direction_weights.pop(row.trader_id, None)
        else:
            direction_weights[row.trader_id] = direction_updated
        diagnostics = self.direction_diagnostic_sums[row.direction]
        diagnostics[0] -= weight * row.effective_history_markets
        diagnostics[1] -= weight * row.mean_similarity
        diagnostics[2] -= weight * row.positive_history_weight_fraction
        diagnostics[3] -= weight * row.skill_score_std
        if remaining_count <= 0:
            self.by_trader_count.pop(row.trader_id, None)
            self.by_trader_weight.pop(row.trader_id, None)
        else:
            self.by_trader_count[row.trader_id] = remaining_count
            self.by_trader_weight[row.trader_id] = updated
            heapq.heappush(self._max_heap, (-updated, row.trader_id))

    def _largest_trader_weight(self) -> float:
        while self._max_heap:
            negative_weight, trader_id = self._max_heap[0]
            current = self.by_trader_weight.get(trader_id)
            if current is not None and abs(current + negative_weight) <= 1e-9 * max(1.0, current):
                return current
            heapq.heappop(self._max_heap)
        return 0.0

    def decide(self) -> SignalDecision:
        skilled_count = len(self.by_trader_count)
        if skilled_count == 0:
            return SignalDecision(False, None, 0.0, 0.0, 0, 0.0)
        if skilled_count < self.min_skilled_traders:
            return SignalDecision(False, None, 0.0, 0.0, skilled_count, 0.0)
        yes_weight = max(0.0, self.yes_weight)
        no_weight = max(0.0, self.no_weight)
        total = yes_weight + no_weight
        if total <= 0.0:
            return SignalDecision(False, None, 0.0, 0.0, skilled_count, 0.0)
        concentration = self._largest_trader_weight() / total
        if concentration > self.max_single_trader_weight:
            return SignalDecision(False, None, 0.0, total, skilled_count, concentration)
        yes_ratio = yes_weight / total
        no_ratio = no_weight / total
        direction: Direction | None = None
        confidence = max(yes_ratio, no_ratio)
        if yes_ratio >= self.consensus_threshold:
            direction = Direction.YES
            confidence = yes_ratio
        elif no_ratio >= self.consensus_threshold:
            direction = Direction.NO
            confidence = no_ratio
        if direction is not None:
            direction_weights = self.by_direction_trader_weight[direction]
            direction_total = yes_weight if direction == Direction.YES else no_weight
            directional_count = len(direction_weights)
            squared_weight_sum = sum(weight**2 for weight in direction_weights.values())
            effective_directional = (
                direction_total**2 / squared_weight_sum if squared_weight_sum > 0.0 else 0.0
            )
            directional_concentration = (
                max(direction_weights.values()) / direction_total
                if direction_weights and direction_total > 0.0
                else 0.0
            )
            diagnostics = self.direction_diagnostic_sums[direction]
            diagnostic_means = [
                value / direction_total if direction_total > 0.0 else 0.0
                for value in diagnostics
            ]
            should_trade = (
                directional_count >= self.min_directional_traders
                and effective_directional >= self.min_effective_directional_traders
                and directional_concentration <= self.max_directional_trader_weight
            )
            return SignalDecision(
                should_trade,
                direction,
                confidence,
                total,
                skilled_count,
                concentration,
                directional_count,
                effective_directional,
                directional_concentration,
                diagnostic_means[0],
                diagnostic_means[1],
                diagnostic_means[2],
                diagnostic_means[3],
            )
        return SignalDecision(False, None, max(yes_ratio, no_ratio), total, skilled_count, concentration)


def decide_direction_from_flow(
    flow: list[FlowObservation],
    skill_threshold: float,
    consensus_threshold: float,
    min_skilled_traders: int = 1,
    max_single_trader_weight: float = 1.0,
    weighting: str = "skill_volume",
    min_directional_traders: int = 1,
    min_effective_directional_traders: float = 1.0,
    max_directional_trader_weight: float = 1.0,
    min_expert_effective_history_markets: float = 0.0,
    min_expert_mean_similarity: float = 0.0,
    min_expert_positive_history_fraction: float = 0.0,
    max_expert_score_std: float = float("inf"),
) -> SignalDecision:
    accumulator = FlowAccumulator(
        skill_threshold=skill_threshold,
        consensus_threshold=consensus_threshold,
        min_skilled_traders=min_skilled_traders,
        max_single_trader_weight=max_single_trader_weight,
        weighting=weighting,
        min_directional_traders=min_directional_traders,
        min_effective_directional_traders=min_effective_directional_traders,
        max_directional_trader_weight=max_directional_trader_weight,
        min_expert_effective_history_markets=min_expert_effective_history_markets,
        min_expert_mean_similarity=min_expert_mean_similarity,
        min_expert_positive_history_fraction=min_expert_positive_history_fraction,
        max_expert_score_std=max_expert_score_std,
    )
    for row in flow:
        accumulator.add(row)
    return accumulator.decide()
