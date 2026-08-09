"""Small, reproducible execution-only candidate grid.

The caller is responsible for passing the frozen signal/risk specification as
``base_config``.  Every preset below changes execution fields only; it never
alters expert selection, semantic controls, consensus, or position sizing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from backtest.engine import BacktestConfig


@dataclass(frozen=True)
class ExecutionPreset:
    name: str
    family: str
    description: str
    overrides: tuple[tuple[str, Any], ...]

    def apply(self, base_config: BacktestConfig) -> BacktestConfig:
        return replace(base_config, **dict(self.overrides))

    def override_dict(self) -> dict[str, Any]:
        return dict(self.overrides)


def _overrides(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(values.items())


def _partial_fill_overrides(
    participation: float,
    ttl_seconds: int,
    trade_filter: str = "target_token_buy",
    max_entry_price: float | None = None,
) -> tuple[tuple[str, Any], ...]:
    values: dict[str, Any] = dict(
        execution_trade_filter=trade_filter,
        max_fill_participation=participation,
        execution_partial_fill=True,
        execution_max_child_fills=1000,
        execution_one_order_per_market=True,
        execution_wait_for_price=True,
        execution_reserve_parent_cash=True,
        pending_order_expiry_seconds=ttl_seconds,
    )
    if max_entry_price is not None:
        values["max_entry_price"] = max_entry_price
    return _overrides(**values)


EXECUTION_PRESETS: tuple[ExecutionPreset, ...] = (
    # Delay sensitivity.  Even zero delay fills on the next tape print because
    # orders are created only after processing the signal-generating trade.
    ExecutionPreset("execution_delay_0s", "delay", "Next post-signal market print.", _overrides(delay_seconds=0)),
    ExecutionPreset("execution_delay_60s", "delay", "First market print at least 60 seconds later.", _overrides(delay_seconds=60)),
    ExecutionPreset("execution_delay_180s", "delay", "First market print at least three minutes later.", _overrides(delay_seconds=180)),
    ExecutionPreset("execution_delay_300s", "delay", "Five-minute reference execution delay.", _overrides(delay_seconds=300)),
    ExecutionPreset("execution_delay_600s", "delay", "First market print at least ten minutes later.", _overrides(delay_seconds=600)),
    ExecutionPreset("execution_delay_900s", "delay", "First market print at least fifteen minutes later.", _overrides(delay_seconds=900)),
    ExecutionPreset("execution_delay_1800s", "delay", "First market print at least thirty minutes later.", _overrides(delay_seconds=1800)),
    # Sequential confirmation on post-delay market prints.
    ExecutionPreset("execution_confirm_1", "confirmation", "Fill on the first eligible print.", _overrides(execution_confirmation_trades=1)),
    ExecutionPreset("execution_confirm_2", "confirmation", "Fill on the second eligible print.", _overrides(execution_confirmation_trades=2)),
    ExecutionPreset("execution_confirm_3", "confirmation", "Fill on the third eligible print.", _overrides(execution_confirmation_trades=3)),
    # Tape-side filter.  A finite TTL prevents a direction-filtered order from
    # reserving exposure indefinitely in an inactive market.
    ExecutionPreset("execution_trade_filter_any", "trade_filter", "Use any post-delay market print.", _overrides(execution_trade_filter="any")),
    ExecutionPreset(
        "execution_trade_filter_signal_direction_24h",
        "trade_filter",
        "Require a same-direction tape print and expire the order after 24 hours.",
        _overrides(
            execution_trade_filter="signal_direction",
            pending_order_expiry_seconds=86400,
        ),
    ),
    # Fill-or-kill limits relative to the signal-time executable token price.
    ExecutionPreset("execution_no_chase_0bps", "no_chase", "Reject any adverse price move.", _overrides(max_price_deterioration_bps=0.0)),
    ExecutionPreset("execution_no_chase_25bps", "no_chase", "Allow 25 bps adverse movement.", _overrides(max_price_deterioration_bps=25.0)),
    ExecutionPreset("execution_no_chase_50bps", "no_chase", "Allow 50 bps adverse movement.", _overrides(max_price_deterioration_bps=50.0)),
    ExecutionPreset("execution_no_chase_100bps", "no_chase", "Allow 100 bps adverse movement.", _overrides(max_price_deterioration_bps=100.0)),
    ExecutionPreset("execution_no_chase_250bps", "no_chase", "Allow 250 bps adverse movement.", _overrides(max_price_deterioration_bps=250.0)),
    ExecutionPreset("execution_no_chase_500bps", "no_chase", "Allow 500 bps adverse movement.", _overrides(max_price_deterioration_bps=500.0)),
    # Equal-notional time slicing across eligible market prints.
    ExecutionPreset("execution_slices_1", "slices", "Single-print execution.", _overrides(execution_slices=1)),
    ExecutionPreset("execution_slices_2", "slices", "Two equal-notional post-signal slices.", _overrides(execution_slices=2)),
    ExecutionPreset(
        "execution_recheck_signal",
        "signal_recheck",
        "Require the frozen signal direction to remain valid immediately before execution.",
        _overrides(execution_recheck_signal=True),
    ),
    # A deliberately small set of economically interpretable interactions.
    ExecutionPreset(
        "execution_delay_60s_confirm_2",
        "combination",
        "One-minute delay followed by a second-print confirmation.",
        _overrides(delay_seconds=60, execution_confirmation_trades=2),
    ),
    ExecutionPreset(
        "execution_delay_60s_no_chase_100bps",
        "combination",
        "One-minute delay with a 100 bps fill-or-kill chase limit.",
        _overrides(delay_seconds=60, max_price_deterioration_bps=100.0),
    ),
    ExecutionPreset(
        "execution_signal_direction_confirm_2_24h",
        "combination",
        "Two same-direction confirmations with a 24-hour order TTL.",
        _overrides(
            execution_trade_filter="signal_direction",
            execution_confirmation_trades=2,
            pending_order_expiry_seconds=86400,
        ),
    ),
    ExecutionPreset(
        "execution_delay_60s_slices_2",
        "combination",
        "One-minute delay with two equal-notional execution slices.",
        _overrides(delay_seconds=60, execution_slices=2),
    ),
    ExecutionPreset(
        "execution_delay_60s_recheck_signal",
        "combination",
        "One-minute delay with a pre-execution consensus recheck.",
        _overrides(delay_seconds=60, execution_recheck_signal=True),
    ),
    # Capacity-aware partial execution.  Every child fill is capped at a fixed
    # fraction of the contemporaneous public print; unfilled notional remains
    # live only until the causal TTL expires.
    ExecutionPreset(
        "execution_partial_any_25pct_2h",
        "capacity_sequence",
        "Partial fills at 25% of any market print for up to two hours.",
        _partial_fill_overrides(0.25, 7200, "any"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_2h",
        "capacity_sequence",
        "Partial fills at 25% of same-token prints for up to two hours.",
        _partial_fill_overrides(0.25, 7200, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_5pct_6h",
        "participation_token",
        "Same-token volume participation at 5% for up to six hours.",
        _partial_fill_overrides(0.05, 21600, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_10pct_6h",
        "participation_token",
        "Same-token volume participation at 10% for up to six hours.",
        _partial_fill_overrides(0.10, 21600, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_6h",
        "participation_token",
        "Same-token volume participation at 25% for up to six hours.",
        _partial_fill_overrides(0.25, 21600, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_50pct_6h",
        "participation_token",
        "Same-token volume participation at 50% for up to six hours.",
        _partial_fill_overrides(0.50, 21600, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_100pct_6h",
        "participation_token",
        "Same-token volume participation at 100% for up to six hours.",
        _partial_fill_overrides(1.00, 21600, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_30m",
        "ttl_token",
        "Same-token volume participation at 25% for 30 minutes.",
        _partial_fill_overrides(0.25, 1800, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_24h",
        "ttl_token",
        "Same-token volume participation at 25% for 24 hours.",
        _partial_fill_overrides(0.25, 86400, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_48h",
        "extended_ttl_token",
        "Same-token volume participation at 25% for up to 48 hours.",
        _partial_fill_overrides(0.25, 172800, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_72h",
        "extended_ttl_token",
        "Same-token volume participation at 25% for up to 72 hours.",
        _partial_fill_overrides(0.25, 259200, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_48h_recheck",
        "extended_ttl_recheck",
        "The 48-hour rule with consensus revalidation before every child fill.",
        _partial_fill_overrides(0.25, 172800, "target_token")
        + _overrides(execution_recheck_signal=True),
    ),
    ExecutionPreset(
        "execution_partial_target_token_10pct_24h",
        "participation_token_24h",
        "Same-token volume participation at 10% for up to 24 hours.",
        _partial_fill_overrides(0.10, 86400, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_50pct_24h",
        "participation_token_24h",
        "Same-token volume participation at 50% for up to 24 hours.",
        _partial_fill_overrides(0.50, 86400, "target_token"),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_24h_recheck",
        "stale_order_control",
        "25% same-token participation for 24 hours with consensus revalidation.",
        _partial_fill_overrides(0.25, 86400, "target_token")
        + _overrides(execution_recheck_signal=True),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_24h_entry_0998",
        "price_capacity_24h",
        "25% same-token participation for 24 hours with a 0.998 entry ceiling.",
        _partial_fill_overrides(0.25, 86400, "target_token", 0.998),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_24h_entry_0999",
        "price_capacity_24h",
        "25% same-token participation for 24 hours with a 0.999 entry ceiling.",
        _partial_fill_overrides(0.25, 86400, "target_token", 0.999),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_24h_cost_20_30bps",
        "cost_capacity_24h",
        "25% same-token participation for 24 hours with 20 bp fees and 30 bp slippage.",
        _partial_fill_overrides(0.25, 86400, "target_token")
        + _overrides(trade_fee_bps=20.0, slippage_bps=30.0),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_24h_entry_0998_cost_20_30bps",
        "cost_capacity_24h",
        "The 0.998 entry-ceiling rule with 20 bp fees and 30 bp slippage.",
        _partial_fill_overrides(0.25, 86400, "target_token", 0.998)
        + _overrides(trade_fee_bps=20.0, slippage_bps=30.0),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_6h_entry_0995",
        "price_capacity",
        "25% same-token participation for six hours with a 0.995 entry ceiling.",
        _partial_fill_overrides(0.25, 21600, "target_token", 0.995),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_6h_entry_0998",
        "price_capacity",
        "25% same-token participation for six hours with a 0.998 entry ceiling.",
        _partial_fill_overrides(0.25, 21600, "target_token", 0.998),
    ),
    ExecutionPreset(
        "execution_partial_target_token_25pct_6h_entry_0999",
        "price_capacity",
        "25% same-token participation for six hours with a 0.999 entry ceiling.",
        _partial_fill_overrides(0.25, 21600, "target_token", 0.999),
    ),
    ExecutionPreset(
        "execution_partial_target_token_50pct_6h_entry_0999",
        "price_capacity",
        "50% same-token participation for six hours with a 0.999 entry ceiling.",
        _partial_fill_overrides(0.50, 21600, "target_token", 0.999),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_5pct_2h",
        "participation",
        "Same-token buy prints, 5% participation, two-hour TTL.",
        _partial_fill_overrides(0.05, 7200),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_10pct_2h",
        "participation",
        "Same-token buy prints, 10% participation, two-hour TTL.",
        _partial_fill_overrides(0.10, 7200),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_25pct_2h",
        "participation",
        "Same-token buy prints, 25% participation, two-hour TTL.",
        _partial_fill_overrides(0.25, 7200),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_50pct_2h",
        "participation",
        "Same-token buy prints, 50% participation, two-hour TTL.",
        _partial_fill_overrides(0.50, 7200),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_100pct_2h",
        "participation",
        "Same-token buy prints, 100% participation, two-hour TTL.",
        _partial_fill_overrides(1.00, 7200),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_25pct_30m",
        "ttl",
        "Same-token buy prints, 25% participation, 30-minute TTL.",
        _partial_fill_overrides(0.25, 1800),
    ),
    ExecutionPreset(
        "execution_partial_target_buy_25pct_6h",
        "ttl",
        "Same-token buy prints, 25% participation, six-hour TTL.",
        _partial_fill_overrides(0.25, 21600),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_5pct_6h",
        "participation_passive",
        "Passive same-token sell prints, 5% participation, six-hour TTL.",
        _partial_fill_overrides(0.05, 21600, "target_token_sell"),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_10pct_6h",
        "participation_passive",
        "Passive same-token sell prints, 10% participation, six-hour TTL.",
        _partial_fill_overrides(0.10, 21600, "target_token_sell"),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_25pct_2h",
        "participation_passive",
        "Passive same-token sell prints, 25% participation, two-hour TTL.",
        _partial_fill_overrides(0.25, 7200, "target_token_sell"),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_25pct_6h",
        "participation_passive",
        "Passive same-token sell prints, 25% participation, six-hour TTL.",
        _partial_fill_overrides(0.25, 21600, "target_token_sell"),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_50pct_2h",
        "participation_passive",
        "Passive same-token sell prints, 50% participation, two-hour TTL.",
        _partial_fill_overrides(0.50, 7200, "target_token_sell"),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_50pct_6h",
        "participation_passive",
        "Passive same-token sell prints, 50% participation, six-hour TTL.",
        _partial_fill_overrides(0.50, 21600, "target_token_sell"),
    ),
    ExecutionPreset(
        "execution_partial_target_sell_100pct_6h",
        "participation_passive",
        "Passive same-token sell prints, 100% participation, six-hour TTL.",
        _partial_fill_overrides(1.00, 21600, "target_token_sell"),
    ),
)


_PRESET_BY_NAME = {preset.name: preset for preset in EXECUTION_PRESETS}


def get_execution_preset(name: str) -> ExecutionPreset:
    try:
        return _PRESET_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(_PRESET_BY_NAME)
        raise KeyError(f"Unknown execution preset {name!r}; available: {available}") from exc


def build_execution_candidate_grid(
    base_config: BacktestConfig,
    preset_names: Iterable[str] | None = None,
) -> list[tuple[str, BacktestConfig]]:
    """Return ordered fresh configs, optionally restricted to named presets."""

    selected = (
        EXECUTION_PRESETS
        if preset_names is None
        else tuple(get_execution_preset(name) for name in preset_names)
    )
    return [(preset.name, preset.apply(base_config)) for preset in selected]
