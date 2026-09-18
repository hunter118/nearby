"""Research-only, prefix-measurable market activation and payout accounting.

No final market volume, future price or realized resolution time enters a gate.
Hourly bars become observable only at the END of the hour. Activation is a
one-time event; both wallet labels and current consensus may use the suffix.
The original core and frozen ICLR package are deliberately left unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np

from backtest.engine import EventDrivenBacktester
from models import ClosedPosition, Direction, ResolutionEvent


@dataclass(frozen=True)
class ActivationSpec:
    min_recent_notional: float = 100_000.0
    min_conviction: float = .90
    window_hours: int = 24
    min_recent_prints: int = 100
    min_observed_hours: int = 24
    max_days_to_scheduled_close: float = 40.0

    def __post_init__(self):
        if not .5 <= self.min_conviction <= 1:
            raise ValueError("Conviction must be in [0.5, 1].")
        if self.window_hours < 1 or self.min_observed_hours < 0:
            raise ValueError("Invalid lookback/warmup.")
        if self.min_recent_notional < 0 or self.min_recent_prints < 1:
            raise ValueError("Invalid volume/print threshold.")


# Small initial grid, selected before seeing this study's returns. Choosing a
# winner after this grid is explicitly exploratory, not an out-of-sample test.
GATES = {
    f"p{int(p * 100)}_v{int(v / 1000)}k": ActivationSpec(v, p)
    for p in (.80, .90, .95)
    for v in (50_000., 250_000.)
}


@dataclass(frozen=True)
class CumulativeActivationSpec:
    """Change only the dollar-volume horizon; price/print windows stay 24h."""
    min_cumulative_notional: float = 250_000.0
    min_conviction: float = .90
    window_hours: int = 24
    min_recent_prints: int = 100
    min_observed_hours: int = 24
    max_days_to_scheduled_close: float = 40.0
    volume_mode: str = 'cumulative'

    def __post_init__(self):
        ActivationSpec(self.min_cumulative_notional,self.min_conviction,self.window_hours,
            self.min_recent_prints,self.min_observed_hours,self.max_days_to_scheduled_close)
        if self.volume_mode != 'cumulative':
            raise ValueError('Use ActivationSpec for rolling-volume gates.')


def rolling_sum(values, width):
    cumulative = np.concatenate(([0.], np.cumsum(values, dtype=np.float64)))
    end = np.arange(1, len(cumulative))
    return cumulative[end] - cumulative[np.maximum(0, end - width)]


def first_activation(times, sizes, yes_prices, token_prices, scheduled_close,
                     spec: ActivationSpec, observed_until=None):
    """Return UTC epoch activation time and its observable diagnostics.

    Input is chronological trade tape, not the future market's final volume.
    No row from the trigger bar enters post-activation scores/consensus. Prices
    are quantity-weighted YES-equivalent prices (NO prints are complemented).
    Windows are wall-clock hours, including hours with zero prints.
    """
    if len(times) == 0 or scheduled_close is None:
        return None
    times = np.asarray(times, dtype=np.int64)
    if np.any(np.diff(times) < 0):
        raise ValueError("Activation tape must be chronological.")
    hours = times // 3600
    first_hour = hours[0]
    bins = hours - first_hour
    n = int(bins[-1] + 1)
    quantities = rolling_sum(np.bincount(bins, weights=sizes, minlength=n), spec.window_hours)
    price_quantity = rolling_sum(np.bincount(bins, weights=sizes * yes_prices,
                                             minlength=n), spec.window_hours)
    dollar_bars = np.bincount(bins, weights=sizes * token_prices, minlength=n)
    notional = rolling_sum(dollar_bars, spec.window_hours)
    cumulative_mode = isinstance(spec, CumulativeActivationSpec)
    trigger_notional = np.cumsum(dollar_bars) if cumulative_mode else notional
    volume_threshold = spec.min_cumulative_notional if cumulative_mode else spec.min_recent_notional
    prints = rolling_sum(np.bincount(bins, minlength=n), spec.window_hours)
    prices = np.divide(price_quantity, quantities, out=np.full(n, .5), where=quantities > 0)
    end_times = (first_hour + np.arange(n) + 1) * 3600
    # A live prefix must not schedule a gate using a partially observed hour.
    # An exhausted historical query can supply its verified observation bound,
    # which may be later than its last print (including silent hours).
    observed_until = times[-1] if observed_until is None else observed_until
    conviction = np.maximum(prices, 1 - prices)
    eligible = ((end_times - times[0] >= spec.min_observed_hours * 3600)
                & (end_times <= observed_until)
                & (scheduled_close - end_times > 0)
                & (scheduled_close - end_times < spec.max_days_to_scheduled_close * 86400)
                & (trigger_notional >= volume_threshold)
                & (prints >= spec.min_recent_prints)
                & (conviction >= spec.min_conviction))
    matches = np.flatnonzero(eligible)
    if not len(matches):
        return None
    i = int(matches[0])
    result = dict(time=int(end_times[i]), recent_notional=float(notional[i]),
                recent_prints=int(prints[i]), recent_yes_vwap=float(prices[i]),
                observed_hours=float((end_times[i] - times[0]) / 3600))
    if cumulative_mode:
        result.update(cumulative_notional=float(trigger_notional[i]),volume_mode='cumulative')
    return result


def settlement_arrays(wallet, side, size, yes_price, yes_payout, wallet_count):
    """Post-cutoff incremental trading P&L, NOT full wallet account profit.

    Sells remain signed exposure, as in the original estimator. An opening
    inventory/cost basis is not inferred after a cut. Fractional payout applies
    equally to the portfolio and the historical wallet labels.
    """
    if yes_payout is None or not len(wallet):
        return dict(wallet=np.array([], dtype=np.uint32), score=np.array([], dtype=float),
                    notional=np.array([], dtype=float))
    if not math.isfinite(yes_payout) or not 0 <= yes_payout <= 1:
        raise ValueError("Invalid payout.")
    # SIDES in models.py: BUY_YES, BUY_NO, SELL_YES, SELL_NO.
    yes = (side == 0) | (side == 2)
    buy = side < 2
    price = np.where(yes, yes_price, 1 - yes_price)
    notional = np.bincount(wallet, weights=size * price, minlength=wallet_count)
    # Match the original aggregation order: total cashflows, then net shares.
    sign = np.where(buy, 1., -1.)
    cash = np.bincount(wallet, weights=-sign * size * price, minlength=wallet_count)
    yes_shares = np.bincount(wallet, weights=sign * size * yes, minlength=wallet_count)
    no_shares = np.bincount(wallet, weights=sign * size * ~yes, minlength=wallet_count)
    pnl = cash + yes_shares * yes_payout + no_shares * (1 - yes_payout)
    valid = np.flatnonzero(notional > 0)
    return dict(wallet=valid.astype(np.uint32), score=np.clip(pnl[valid] / notional[valid], -1., 1.),
                notional=notional[valid])


@dataclass(frozen=True)
class FractionalResolution(ResolutionEvent):
    resolution: Direction | None
    yes_payout: float


class ActivationBacktester(EventDrivenBacktester):
    """Keep the execution/position rules; change only the admitted signal tape."""
    def __init__(self, *args, activation_times=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_times = activation_times
        self.processed = 0
        self.admitted = 0

    def _on_trade(self, trade):
        self.processed += 1
        active_at = None if self.activation_times is None else self.activation_times.get(trade.market_id)
        if self.activation_times is None or (active_at is not None and trade.timestamp >= active_at):
            self.admitted += 1
            super()._on_trade(trade)
        # The inherited run() still sees every print for mark-to-market and
        # execution. Only _on_trade's consensus accumulation is gated.
        if self.processed % 2_000_000 == 0:
            print(f"Tape {self.processed:,}; admitted {self.admitted:,}; {trade.timestamp}", flush=True)

    def _on_resolution(self, resolution):
        if not isinstance(resolution, FractionalResolution):
            return super()._on_resolution(resolution)
        self.pending_orders = [o for o in self.pending_orders if o.market_id != resolution.market_id]
        position = self.positions.pop(resolution.market_id, None)
        if position is None:
            return
        rate = resolution.yes_payout if position.direction == Direction.YES else 1 - resolution.yes_payout
        payout = position.quantity * rate
        notional = position.quantity * position.avg_entry_price
        self.balance += payout
        self.closed.append(ClosedPosition(position.market_id, position.direction, position.quantity,
            position.avg_entry_price, notional, payout, payout - notional,
            position.opened_at, resolution.resolved_at))
