from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from alpha.signal import FlowObservation, decide_direction_from_flow
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


class EventDrivenBacktester:
    def __init__(
        self,
        markets: dict[str, Market],
        timeline: list[TimelineEvent],
        skill_estimator: TraderSkillEstimator,
        config: BacktestConfig,
    ) -> None:
        self.markets = markets
        self.timeline = sorted(timeline, key=lambda x: x.ts)
        self.skill_estimator = skill_estimator
        self.config = config

        self.balance = config.initial_balance
        self.positions: dict[str, Position] = {}
        self.pending_orders: list[PendingOrder] = []
        self.fills: list[FilledOrder] = []
        self.closed: list[ClosedPosition] = []
        self.market_flow: dict[str, list[FlowObservation]] = {}
        self.market_trade_count: dict[str, int] = {}
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

    def run(self) -> dict[str, object]:
        for event in self.timeline:
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
            open_notional, open_market_value, open_unrealized_pnl, total_equity = self._portfolio_snapshot()
            self.equity_curve.append(
                {
                    "ts": event.ts.isoformat(sep=" "),
                    "cash_balance": self.balance,
                    "open_notional": open_notional,
                    "open_market_value": open_market_value,
                    "open_unrealized_pnl": open_unrealized_pnl,
                    "total_equity": total_equity,
                }
            )

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
        }

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
            if order.execute_after > now:
                remaining.append(order)
                continue
            if trade.market_id != order.market_id:
                remaining.append(order)
                continue
            if order.market_id in self.positions:
                continue
            trade_count = self.market_trade_count.get(order.market_id, 0)
            if trade_count >= self.config.max_trades_per_market:
                continue

            token_price = _token_price_from_yes(order.direction, trade.price_yes)
            slippage = self.config.slippage_bps / 10000.0
            token_price = min(1.0, max(0.0, token_price * (1.0 + slippage)))
            if token_price <= 0:
                continue
            if order.price_bucket == "stable" and token_price <= self.config.stable_min_price:
                continue
            if order.price_bucket == "lottery" and token_price >= self.config.lottery_max_price:
                continue

            if order.fixed_quantity is not None:
                qty = order.fixed_quantity
                max_notional = qty * token_price
            else:
                max_notional = min(order.max_notional, self.balance)
                qty = max_notional / token_price
                if qty <= 0:
                    continue
            fee = max_notional * (self.config.trade_fee_bps / 10000.0)
            total_cost = max_notional + fee
            if total_cost > self.balance:
                continue
            self.balance -= total_cost
            self.positions[order.market_id] = Position(
                market_id=order.market_id,
                direction=order.direction,
                quantity=qty,
                avg_entry_price=token_price,
                opened_at=trade.timestamp,
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
                )
            )
            self.market_trade_count[order.market_id] = trade_count + 1
        self.pending_orders = remaining

    def _on_trade(self, trade: TradeEvent) -> None:
        if trade.size <= 0:
            return
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
        )
        self.market_flow.setdefault(trade.market_id, []).append(obs)
        decision = decide_direction_from_flow(
            flow=self.market_flow[trade.market_id],
            skill_threshold=self.config.skill_threshold,
            consensus_threshold=self.config.consensus_threshold,
            min_skilled_traders=self.config.min_skilled_traders,
            max_single_trader_weight=self.config.max_single_trader_weight,
        )
        if not decision.should_trade or decision.direction is None:
            return
        # Filter out traders without enough relevant historical volume.
        if skill.weighted_history_notional < self.config.min_user_volume:
            return
        token_price = _token_price_from_yes(decision.direction, trade.price_yes)
        bucket = self._classify_price_bucket(token_price)
        if bucket is None:
            return
        if trade.market_id in self.positions:
            return
        if any(x.market_id == trade.market_id for x in self.pending_orders):
            return

        market = self.markets.get(trade.market_id)
        if market is None:
            return
        settle_ts = market.resolved_at or market.close_time
        if settle_ts is None:
            return
        min_wait_seconds = self.config.min_days_to_resolution * 86400.0
        max_wait_seconds = self.config.max_days_to_resolution * 86400.0
        wait_seconds = (settle_ts - trade.timestamp).total_seconds()
        if wait_seconds <= min_wait_seconds or wait_seconds >= max_wait_seconds:
            return
        wait_days = wait_seconds / 86400.0

        if bucket == "stable":
            order_notional = self._stable_order_notional(
                token_price=token_price,
                confidence=decision.confidence,
                wait_days=wait_days,
            )
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

        self.pending_orders.append(
            PendingOrder(
                market_id=trade.market_id,
                direction=decision.direction,
                signal_time=trade.timestamp,
                execute_after=trade.timestamp + timedelta(seconds=self.config.delay_seconds),
                max_notional=order_notional,
                signal_confidence=decision.confidence,
                allowed_entry_price=1.0,
                fixed_quantity=fixed_quantity,
                price_bucket=bucket,
            )
        )

    def _on_resolution(self, resolution: ResolutionEvent) -> None:
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
