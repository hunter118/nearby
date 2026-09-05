from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from collections import deque

from alpha.signal import FlowAccumulator, FlowObservation, SignalDecision
from alpha.semantic_risk import semantic_risk_class
from alpha.trader_skill import TraderSkillEstimator
from models import (
    ClosedPosition,
    Direction,
    FilledOrder,
    Market,
    PendingOrder,
    Position,
    ResolutionEvent,
    Side,
    TimelineEvent,
    TradeEvent,
)


def _trade_to_direction(side: Side) -> Direction:
    if side in (Side.BUY_YES, Side.SELL_NO):
        return Direction.YES
    return Direction.NO


def _trade_matches_target_token(trade: TradeEvent, direction: Direction) -> bool:
    if direction == Direction.YES:
        return trade.side in (Side.BUY_YES, Side.SELL_YES)
    return trade.side in (Side.BUY_NO, Side.SELL_NO)


def _trade_is_target_token_buy(trade: TradeEvent, direction: Direction) -> bool:
    expected_side = Side.BUY_YES if direction == Direction.YES else Side.BUY_NO
    return trade.side == expected_side


def _trade_is_target_token_sell(trade: TradeEvent, direction: Direction) -> bool:
    expected_side = Side.SELL_YES if direction == Direction.YES else Side.SELL_NO
    return trade.side == expected_side


def _traded_token_price(trade: TradeEvent) -> float:
    if trade.side in (Side.BUY_YES, Side.SELL_YES):
        return trade.price_yes
    return 1.0 - trade.price_yes


def _token_price_from_yes(direction: Direction, price_yes: float) -> float:
    return price_yes if direction == Direction.YES else 1.0 - price_yes


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class BacktestConfig:
    delay_seconds: int
    skill_threshold: float
    consensus_threshold: float
    min_skilled_traders: int
    max_single_trader_weight: float
    min_edge: float
    min_user_volume: float
    max_trades_per_market: int
    stable_min_price: float
    lottery_min_price: float
    lottery_max_price: float
    stable_balance_fraction: float
    lottery_lot_size: float
    lottery_max_exposure_fraction: float
    min_days_to_resolution: float
    max_days_to_resolution: float
    trade_fee_bps: float
    slippage_bps: float
    min_entry_price: float
    max_entry_price: float
    dynamic_price_at_consensus: float
    dynamic_price_at_high_confidence: float
    dynamic_high_confidence: float
    max_market_fraction: float
    max_balance_fraction: float
    max_loss_per_trade_fraction: float
    min_ticket_size: float
    initial_balance: float
    position_sizing: str = "fixed_fraction"
    target_exposure_fraction: float = 0.0
    cash_buffer_fraction: float = 0.0
    min_target_order_fraction: float = 0.0
    max_target_order_fraction: float = 1.0
    annualized_edge_multiplier: float = 1.0
    signal_weighting: str = "skill_volume"
    flow_lookback_seconds: int | None = None
    max_fill_participation: float | None = None
    enforce_risk_caps: bool = False
    signal_mode: str = "expert_flow"
    equity_record_interval: int = 1
    min_directional_traders: int = 1
    min_effective_directional_traders: float = 1.0
    max_directional_trader_weight: float = 1.0
    min_expert_effective_history_markets: float = 0.0
    min_expert_mean_similarity: float = 0.0
    min_expert_positive_history_fraction: float = 0.0
    max_expert_score_std: float = float("inf")
    semantic_cluster_similarity_threshold: float | None = None
    max_semantic_cluster_exposure_fraction: float = 1.0
    apply_market_volume_cap: bool = True
    apply_balance_cap: bool = True
    apply_loss_cap: bool = True
    entry_start_ts: datetime | None = None
    entry_end_ts: datetime | None = None
    min_signal_mean_expert_history_markets: float = 0.0
    max_competitive_event_exposure_fraction: float | None = None
    max_position_exposure_fraction: float | None = None
    execution_trade_filter: str = "any"
    execution_confirmation_trades: int = 1
    max_price_deterioration_bps: float | None = None
    pending_order_expiry_seconds: int | None = None
    execution_slices: int = 1
    execution_recheck_signal: bool = False
    execution_partial_fill: bool = False
    execution_max_child_fills: int = 1
    execution_one_order_per_market: bool = False
    execution_wait_for_price: bool = False
    execution_reserve_parent_cash: bool = False


class EventDrivenBacktester:
    def __init__(
        self,
        markets: dict[str, Market],
        timeline: list[TimelineEvent],
        skill_estimator: TraderSkillEstimator,
        config: BacktestConfig,
    ) -> None:
        self.markets = markets
        # Dataset construction already returns a chronological timeline.  Avoid
        # copying and re-sorting multi-million-event holdouts for every ablation.
        self.timeline = timeline
        self.skill_estimator = skill_estimator
        self.config = config
        if config.execution_trade_filter not in {
            "any",
            "signal_direction",
            "target_token",
            "target_token_buy",
            "target_token_sell",
        }:
            raise ValueError(
                "execution_trade_filter must be one of 'any', 'signal_direction', "
                "'target_token', 'target_token_buy', or 'target_token_sell'."
            )
        if config.execution_confirmation_trades < 1:
            raise ValueError("execution_confirmation_trades must be at least one.")
        if config.execution_slices < 1:
            raise ValueError("execution_slices must be at least one.")
        if config.execution_max_child_fills < 1:
            raise ValueError("execution_max_child_fills must be at least one.")
        if config.max_fill_participation is not None and not (
            0.0 < config.max_fill_participation <= 1.0
        ):
            raise ValueError("max_fill_participation must be in (0, 1].")

        self.balance = config.initial_balance
        self.positions: dict[str, Position] = {}
        self.pending_orders: list[PendingOrder] = []
        self.fills: list[FilledOrder] = []
        self.closed: list[ClosedPosition] = []
        self.market_flow: dict[str, list[FlowObservation]] = {}
        self.flow_accumulators: dict[str, FlowAccumulator] = {}
        self.timestamped_flow: dict[str, deque[tuple[object, FlowObservation]]] = {}
        self.market_trade_count: dict[str, int] = {}
        self.execution_rejected_markets: set[str] = set()
        self.execution_submitted_markets: set[str] = set()
        self.execution_order_requests: dict[str, float] = {}
        self.market_observed_notional: dict[str, float] = {}
        self.last_yes_price: dict[str, float] = {}
        self.equity_curve: list[dict[str, float | str]] = []

    def _max_allowed_entry_price(self, signal_confidence: float) -> float:
        low = self.config.dynamic_price_at_consensus
        high = self.config.dynamic_price_at_high_confidence
        high_conf = self.config.dynamic_high_confidence
        consensus = self.config.consensus_threshold
        if high_conf <= consensus:
            return min(self.config.max_entry_price, low)
        clipped_conf = _clamp(signal_confidence, consensus, high_conf)
        ratio = (clipped_conf - consensus) / (high_conf - consensus)
        dynamic_cap = low + ratio * (high - low)
        return min(self.config.max_entry_price, dynamic_cap)

    def _classify_price_bucket(self, token_price: float) -> str | None:
        if token_price > self.config.stable_min_price:
            return "stable"
        if self.config.lottery_min_price <= token_price <= self.config.lottery_max_price:
            return "lottery"
        return None

    def _lottery_open_exposure(self) -> float:
        exposure = 0.0
        for order in self.pending_orders:
            if order.price_bucket == "lottery":
                exposure += order.max_notional
        for position in self.positions.values():
            token_price = position.avg_entry_price
            if self.config.lottery_min_price <= token_price <= self.config.lottery_max_price:
                exposure += position.quantity * position.avg_entry_price
        return exposure

    def _pending_stable_exposure(self) -> float:
        return sum(order.max_notional for order in self.pending_orders if order.price_bucket == "stable")

    def _semantic_cluster_exposure(self, target_market_id: str) -> float:
        threshold = self.config.semantic_cluster_similarity_threshold
        if threshold is None:
            return 0.0
        exposure = 0.0
        for order in self.pending_orders:
            if (
                order.price_bucket == "stable"
                and self.skill_estimator.market_similarity(target_market_id, order.market_id)
                >= threshold
            ):
                exposure += order.max_notional
        for position in self.positions.values():
            if (
                self.skill_estimator.market_similarity(target_market_id, position.market_id)
                >= threshold
            ):
                exposure += position.quantity * position.avg_entry_price
        return exposure

    def _stable_order_notional(self, token_price: float, confidence: float, wait_days: float) -> float:
        if self.config.position_sizing != "target_exposure_annualized":
            return self.balance * self.config.stable_balance_fraction

        open_notional, _, _, total_equity = self._portfolio_snapshot()
        target_deployed = total_equity * self.config.target_exposure_fraction
        deployed_or_committed = open_notional + self._pending_stable_exposure()
        remaining_to_target = max(0.0, target_deployed - deployed_or_committed)
        cash_buffer = total_equity * self.config.cash_buffer_fraction
        available_cash = max(0.0, self.balance - cash_buffer)
        if remaining_to_target <= 0.0 or available_cash <= 0.0:
            return 0.0

        signal_edge = max(0.0, confidence - token_price)
        annualized_edge = signal_edge * 365.0 / max(wait_days, 1e-9)
        allocation_fraction = annualized_edge * self.config.annualized_edge_multiplier
        allocation_fraction = _clamp(
            allocation_fraction,
            self.config.min_target_order_fraction,
            self.config.max_target_order_fraction,
        )
        return min(remaining_to_target * allocation_fraction, available_cash)

    def _execution_available_notional(
        self,
        exclude_order: PendingOrder | None = None,
    ) -> float:
        """Cash available to one parent order after honoring other reservations."""
        if not self.config.execution_reserve_parent_cash:
            return self.balance
        fee_multiplier = 1.0 + self.config.trade_fee_bps / 10000.0
        reserved_gross = sum(
            order.max_notional * fee_multiplier
            for order in self.pending_orders
            if order is not exclude_order
        )
        return max(0.0, self.balance - reserved_gross) / fee_multiplier

    def run(self) -> dict[str, object]:
        last_event_ts = None
        for event_index, event in enumerate(self.timeline):
            if (
                self.config.equity_record_interval <= 0
                and last_event_ts is not None
                and event.ts.date() != last_event_ts.date()
            ):
                # In daily mode, preserve the state after the final event of
                # the preceding UTC day before processing the new day's event.
                self._record_equity(last_event_ts)
            if event.event_type == "trade":
                trade = event.payload
                assert isinstance(trade, TradeEvent)
                self.last_yes_price[trade.market_id] = trade.price_yes
                self._process_due_orders(trade.timestamp, trade)
                self._on_trade(trade)
            else:
                resolution = event.payload
                assert isinstance(resolution, ResolutionEvent)
                self._on_resolution(resolution)
            interval = self.config.equity_record_interval
            if interval > 0 and (
                event_index % interval == 0 or event.event_type == "resolution"
            ):
                self._record_equity(event.ts)
            last_event_ts = event.ts

        if last_event_ts is not None and (
            not self.equity_curve or self.equity_curve[-1]["ts"] != last_event_ts.isoformat(sep=" ")
        ):
            self._record_equity(last_event_ts)

        open_notional, open_market_value, open_unrealized_pnl, total_equity = self._portfolio_snapshot()

        return {
            "balance": self.balance,
            "fills": self.fills,
            "closed_positions": self.closed,
            "open_positions": list(self.positions.values()),
            "open_notional": open_notional,
            "open_market_value": open_market_value,
            "open_unrealized_pnl": open_unrealized_pnl,
            "total_equity": total_equity,
            "equity_curve": self.equity_curve,
            "execution_order_requests": dict(self.execution_order_requests),
            "execution_rejected_markets": sorted(self.execution_rejected_markets),
            "execution_pending_orders": len(self.pending_orders),
        }

    def _record_equity(self, timestamp) -> None:
        open_notional, open_market_value, open_unrealized_pnl, total_equity = self._portfolio_snapshot()
        self.equity_curve.append(
            {
                "ts": timestamp.isoformat(sep=" "),
                "cash_balance": self.balance,
                "open_notional": open_notional,
                "open_market_value": open_market_value,
                "open_unrealized_pnl": open_unrealized_pnl,
                "total_equity": total_equity,
            }
        )

    def _portfolio_snapshot(self) -> tuple[float, float, float, float]:
        open_notional = 0.0
        open_market_value = 0.0
        for position in self.positions.values():
            open_notional += position.quantity * position.avg_entry_price
            last_yes = self.last_yes_price.get(position.market_id, position.avg_entry_price)
            token_price = _token_price_from_yes(position.direction, last_yes)
            open_market_value += position.quantity * token_price
        open_unrealized_pnl = open_market_value - open_notional
        total_equity = self.balance + open_market_value
        return open_notional, open_market_value, open_unrealized_pnl, total_equity

    def _process_due_orders(self, now, trade: TradeEvent) -> None:
        if not self.pending_orders:
            return
        remaining: list[PendingOrder] = []
        for order in self.pending_orders:
            if (
                self.config.pending_order_expiry_seconds is not None
                and now
                > order.execute_after
                + timedelta(seconds=self.config.pending_order_expiry_seconds)
            ):
                continue
            if order.execute_after > now:
                remaining.append(order)
                continue
            if trade.market_id != order.market_id:
                remaining.append(order)
                continue
            trade_count = self.market_trade_count.get(order.market_id, 0)
            child_fill_limit = (
                self.config.execution_max_child_fills
                if self.config.execution_partial_fill
                else self.config.execution_slices
            )
            if trade_count >= child_fill_limit:
                continue
            if (
                self.config.execution_trade_filter == "signal_direction"
                and _trade_to_direction(trade.side) != order.direction
            ):
                remaining.append(order)
                continue
            if (
                self.config.execution_trade_filter == "target_token"
                and not _trade_matches_target_token(trade, order.direction)
            ):
                remaining.append(order)
                continue
            if (
                self.config.execution_trade_filter == "target_token_buy"
                and not _trade_is_target_token_buy(trade, order.direction)
            ):
                remaining.append(order)
                continue
            if (
                self.config.execution_trade_filter == "target_token_sell"
                and not _trade_is_target_token_sell(trade, order.direction)
            ):
                remaining.append(order)
                continue
            if self.config.execution_recheck_signal:
                accumulator = self.flow_accumulators.get(order.market_id)
                current_decision = accumulator.decide() if accumulator is not None else None
                if (
                    current_decision is None
                    or not current_decision.should_trade
                    or current_decision.direction != order.direction
                ):
                    self.execution_rejected_markets.add(order.market_id)
                    continue

            token_price = _token_price_from_yes(order.direction, trade.price_yes)
            slippage = self.config.slippage_bps / 10000.0
            token_price = min(1.0, max(0.0, token_price * (1.0 + slippage)))
            if token_price <= 0:
                continue
            if token_price < self.config.min_entry_price:
                continue
            if (
                token_price > self.config.max_entry_price
                or token_price > order.allowed_entry_price
            ):
                if self.config.execution_wait_for_price:
                    remaining.append(order)
                continue
            if self.config.max_price_deterioration_bps is not None:
                deterioration_multiplier = (
                    1.0 + self.config.max_price_deterioration_bps / 10000.0
                )
                if token_price > order.signal_token_price * deterioration_multiplier:
                    if self.config.execution_wait_for_price:
                        remaining.append(order)
                    else:
                        # A fill-or-kill no-chase policy rejects the market at
                        # the first otherwise executable post-delay print.
                        self.execution_rejected_markets.add(order.market_id)
                    continue
            if order.price_bucket == "stable" and token_price <= self.config.stable_min_price:
                continue
            if order.price_bucket == "lottery" and token_price >= self.config.lottery_max_price:
                continue

            confirmations = order.confirmation_trades_seen + 1
            if confirmations < self.config.execution_confirmation_trades:
                remaining.append(
                    replace(order, confirmation_trades_seen=confirmations)
                )
                continue

            if order.fixed_quantity is not None:
                qty = order.fixed_quantity / max(order.remaining_slices, 1)
                max_notional = qty * token_price
            else:
                slice_notional = (
                    order.max_notional
                    if self.config.execution_partial_fill
                    else order.max_notional / max(order.remaining_slices, 1)
                )
                max_notional = min(
                    slice_notional,
                    self._execution_available_notional(exclude_order=order),
                )
                if self.config.max_fill_participation is not None:
                    observed_notional = trade.size * token_price
                    max_notional = min(
                        max_notional,
                        observed_notional * self.config.max_fill_participation,
                    )
                qty = max_notional / token_price
                if qty <= 0:
                    continue
            fee = max_notional * (self.config.trade_fee_bps / 10000.0)
            total_cost = max_notional + fee
            if total_cost > self.balance:
                continue
            self.balance -= total_cost
            existing_position = self.positions.get(order.market_id)
            if existing_position is None:
                self.positions[order.market_id] = Position(
                    market_id=order.market_id,
                    direction=order.direction,
                    quantity=qty,
                    avg_entry_price=token_price,
                    opened_at=trade.timestamp,
                )
            else:
                combined_quantity = existing_position.quantity + qty
                combined_notional = (
                    existing_position.quantity * existing_position.avg_entry_price
                    + max_notional
                )
                self.positions[order.market_id] = Position(
                    market_id=order.market_id,
                    direction=order.direction,
                    quantity=combined_quantity,
                    avg_entry_price=combined_notional / combined_quantity,
                    opened_at=existing_position.opened_at,
                )
            self.fills.append(
                FilledOrder(
                    market_id=order.market_id,
                    direction=order.direction,
                    quantity=qty,
                    fill_price=token_price,
                    notional=max_notional,
                    fee=fee,
                    signal_time=order.signal_time,
                    filled_at=trade.timestamp,
                    signal_confidence=order.signal_confidence,
                    skilled_trader_count=order.skilled_trader_count,
                    directional_trader_count=order.directional_trader_count,
                    effective_directional_traders=order.effective_directional_traders,
                    signal_concentration=order.signal_concentration,
                    directional_concentration=order.directional_concentration,
                    mean_expert_history_markets=order.mean_expert_history_markets,
                    mean_expert_similarity=order.mean_expert_similarity,
                    mean_positive_history_fraction=order.mean_positive_history_fraction,
                    mean_skill_score_std=order.mean_skill_score_std,
                    semantic_risk_class=order.semantic_risk_class,
                    signal_token_price=order.signal_token_price,
                    order_requested_notional=order.original_notional,
                    source_trade_size=trade.size,
                    source_trade_side=trade.side.value,
                    child_fill_index=trade_count + 1,
                )
            )
            self.market_trade_count[order.market_id] = trade_count + 1
            remaining_slices = order.remaining_slices - 1
            remaining_notional = max(0.0, order.max_notional - max_notional)
            if (
                (
                    self.config.execution_partial_fill
                    or remaining_slices > 0
                )
                and remaining_notional >= self.config.min_ticket_size
                and trade_count + 1 < child_fill_limit
            ):
                remaining.append(
                    replace(
                        order,
                        max_notional=remaining_notional,
                        fixed_quantity=(
                            None
                            if order.fixed_quantity is None
                            else max(0.0, order.fixed_quantity - qty)
                        ),
                        confirmation_trades_seen=(
                            self.config.execution_confirmation_trades
                            if self.config.execution_partial_fill
                            else 0
                        ),
                        remaining_slices=(
                            order.remaining_slices
                            if self.config.execution_partial_fill
                            else remaining_slices
                        ),
                    )
                )
        self.pending_orders = remaining

    def _on_trade(self, trade: TradeEvent) -> None:
        if trade.size <= 0:
            return
        # Capacity controls must use volume observed by this point in the
        # timeline, never Gamma's final snapshot volume.
        self.market_observed_notional[trade.market_id] = (
            self.market_observed_notional.get(trade.market_id, 0.0)
            + trade.size * _traded_token_price(trade)
        )
        skill = None
        if self.config.signal_mode == "favorite":
            direction = Direction.YES if trade.price_yes >= 0.5 else Direction.NO
            confidence = _token_price_from_yes(direction, trade.price_yes)
            decision = SignalDecision(True, direction, confidence, 1.0, 0, 0.0)
        elif self.config.signal_mode == "expert_flow":
            skill = self.skill_estimator.estimate(
                trader_id=trade.trader_id,
                target_market_id=trade.market_id,
                as_of=trade.timestamp,
            )
            direction = _trade_to_direction(trade.side)
            obs = FlowObservation(
                trader_id=trade.trader_id,
                market_id=trade.market_id,
                direction=direction,
                volume=trade.size,
                skill=skill.weighted_score,
                weighted_history_notional=skill.weighted_history_notional,
                effective_history_markets=skill.effective_history_markets,
                mean_similarity=skill.mean_similarity,
                positive_history_weight_fraction=skill.positive_history_weight_fraction,
                skill_score_std=skill.weighted_score_std,
            )
            accumulator = self.flow_accumulators.get(trade.market_id)
            if accumulator is None:
                accumulator = FlowAccumulator(
                    skill_threshold=self.config.skill_threshold,
                    consensus_threshold=self.config.consensus_threshold,
                    min_skilled_traders=self.config.min_skilled_traders,
                    max_single_trader_weight=self.config.max_single_trader_weight,
                    weighting=self.config.signal_weighting,
                    min_directional_traders=self.config.min_directional_traders,
                    min_effective_directional_traders=self.config.min_effective_directional_traders,
                    max_directional_trader_weight=self.config.max_directional_trader_weight,
                    min_weighted_history_notional=self.config.min_user_volume,
                    min_expert_effective_history_markets=(
                        self.config.min_expert_effective_history_markets
                    ),
                    min_expert_mean_similarity=self.config.min_expert_mean_similarity,
                    min_expert_positive_history_fraction=(
                        self.config.min_expert_positive_history_fraction
                    ),
                    max_expert_score_std=self.config.max_expert_score_std,
                )
                self.flow_accumulators[trade.market_id] = accumulator
            accumulator.add(obs)
            if self.config.flow_lookback_seconds is not None:
                cutoff = trade.timestamp - timedelta(seconds=self.config.flow_lookback_seconds)
                rows = self.timestamped_flow.setdefault(trade.market_id, deque())
                rows.append((trade.timestamp, obs))
                while rows and rows[0][0] < cutoff:
                    _, expired = rows.popleft()
                    accumulator.remove(expired)
            decision = accumulator.decide()
        else:
            raise ValueError(f"Unsupported signal mode: {self.config.signal_mode}")
        if not decision.should_trade or decision.direction is None:
            return
        if self.config.entry_start_ts is not None and trade.timestamp < self.config.entry_start_ts:
            return
        if self.config.entry_end_ts is not None and trade.timestamp > self.config.entry_end_ts:
            return
        if (
            decision.mean_expert_history_markets
            < self.config.min_signal_mean_expert_history_markets
        ):
            return
        token_price = _token_price_from_yes(decision.direction, trade.price_yes)
        if token_price < self.config.min_entry_price or token_price > self.config.max_entry_price:
            return
        if decision.confidence - token_price < self.config.min_edge:
            return
        allowed_entry_price = self._max_allowed_entry_price(decision.confidence)
        if token_price > allowed_entry_price:
            return
        bucket = self._classify_price_bucket(token_price)
        if bucket is None:
            return
        if trade.market_id in self.execution_rejected_markets:
            return
        if (
            self.config.execution_one_order_per_market
            and trade.market_id in self.execution_submitted_markets
        ):
            return
        if trade.market_id in self.positions:
            return
        if any(x.market_id == trade.market_id for x in self.pending_orders):
            return

        market = self.markets.get(trade.market_id)
        if market is None:
            return
        # Entry gating must use the scheduled close known to traders, not the
        # ex-post realization time of the outcome.
        settle_ts = market.close_time
        if settle_ts is None:
            return
        min_wait_seconds = self.config.min_days_to_resolution * 86400.0
        max_wait_seconds = self.config.max_days_to_resolution * 86400.0
        wait_seconds = (settle_ts - trade.timestamp).total_seconds()
        if wait_seconds <= min_wait_seconds or wait_seconds >= max_wait_seconds:
            return
        wait_days = wait_seconds / 86400.0
        risk_class = semantic_risk_class(market.question)

        if bucket == "stable":
            order_notional = self._stable_order_notional(
                token_price=token_price,
                confidence=decision.confidence,
                wait_days=wait_days,
            )
            if order_notional < self.config.min_ticket_size:
                return
            if self.config.max_position_exposure_fraction is not None:
                _, _, _, total_equity = self._portfolio_snapshot()
                order_notional = min(
                    order_notional,
                    total_equity * self.config.max_position_exposure_fraction,
                )
                if order_notional < self.config.min_ticket_size:
                    return
            if (
                risk_class == "competitive_event"
                and self.config.max_competitive_event_exposure_fraction is not None
            ):
                _, _, _, total_equity = self._portfolio_snapshot()
                order_notional = min(
                    order_notional,
                    total_equity * self.config.max_competitive_event_exposure_fraction,
                )
                if order_notional < self.config.min_ticket_size:
                    return
            if self.config.enforce_risk_caps:
                risk_caps = [order_notional]
                if self.config.apply_market_volume_cap:
                    risk_caps.append(
                        self.market_observed_notional.get(trade.market_id, 0.0)
                        * self.config.max_market_fraction
                    )
                if self.config.apply_balance_cap:
                    risk_caps.append(self.balance * self.config.max_balance_fraction)
                if self.config.apply_loss_cap:
                    risk_caps.append(self.balance * self.config.max_loss_per_trade_fraction)
                order_notional = min(risk_caps)
                if order_notional < self.config.min_ticket_size:
                    return
            if self.config.semantic_cluster_similarity_threshold is not None:
                _, _, _, total_equity = self._portfolio_snapshot()
                cluster_cap = total_equity * self.config.max_semantic_cluster_exposure_fraction
                remaining_cluster_capacity = max(
                    0.0,
                    cluster_cap - self._semantic_cluster_exposure(trade.market_id),
                )
                order_notional = min(order_notional, remaining_cluster_capacity)
                if order_notional < self.config.min_ticket_size:
                    return
            fixed_quantity = None
        else:
            if self.config.lottery_max_exposure_fraction <= 0:
                return
            fixed_quantity = self.config.lottery_lot_size
            order_notional = fixed_quantity * token_price
            equity = self.balance
            lottery_cap = equity * self.config.lottery_max_exposure_fraction
            if self._lottery_open_exposure() + order_notional > lottery_cap:
                return

        if self.config.execution_reserve_parent_cash:
            available_notional = self._execution_available_notional()
            if fixed_quantity is not None and order_notional > available_notional:
                return
            order_notional = min(order_notional, available_notional)
            if order_notional < self.config.min_ticket_size:
                return

        self.pending_orders.append(
            PendingOrder(
                market_id=trade.market_id,
                direction=decision.direction,
                signal_time=trade.timestamp,
                execute_after=trade.timestamp + timedelta(seconds=self.config.delay_seconds),
                max_notional=order_notional,
                signal_confidence=decision.confidence,
                allowed_entry_price=allowed_entry_price,
                signal_token_price=min(
                    1.0,
                    token_price * (1.0 + self.config.slippage_bps / 10000.0),
                ),
                original_notional=order_notional,
                fixed_quantity=fixed_quantity,
                price_bucket=bucket,
                remaining_slices=self.config.execution_slices,
                skilled_trader_count=decision.skilled_trader_count,
                directional_trader_count=decision.directional_trader_count,
                effective_directional_traders=decision.effective_directional_traders,
                signal_concentration=decision.max_single_trader_weight,
                directional_concentration=decision.directional_concentration,
                mean_expert_history_markets=decision.mean_expert_history_markets,
                mean_expert_similarity=decision.mean_expert_similarity,
                mean_positive_history_fraction=decision.mean_positive_history_fraction,
                mean_skill_score_std=decision.mean_skill_score_std,
                semantic_risk_class=risk_class,
            )
        )
        self.execution_submitted_markets.add(trade.market_id)
        self.execution_order_requests[trade.market_id] = order_notional

    def _on_resolution(self, resolution: ResolutionEvent) -> None:
        self.pending_orders = [
            order
            for order in self.pending_orders
            if order.market_id != resolution.market_id
        ]
        position = self.positions.pop(resolution.market_id, None)
        if position is None:
            return
        payout_per_share = 1.0 if resolution.resolution == position.direction else 0.0
        payout = position.quantity * payout_per_share
        notional = position.quantity * position.avg_entry_price
        pnl = payout - notional
        self.balance += payout
        self.closed.append(
            ClosedPosition(
                market_id=position.market_id,
                direction=position.direction,
                quantity=position.quantity,
                avg_entry_price=position.avg_entry_price,
                notional=notional,
                payout=payout,
                pnl=pnl,
                opened_at=position.opened_at,
                resolved_at=resolution.resolved_at,
            )
        )
