from dataclasses import asdict

import pytest

from alpha.execution_presets import (
    EXECUTION_PRESETS,
    build_execution_candidate_grid,
    get_execution_preset,
)
from backtest.engine import BacktestConfig


EXECUTION_FIELDS = {
    "delay_seconds",
    "execution_trade_filter",
    "execution_confirmation_trades",
    "max_price_deterioration_bps",
    "pending_order_expiry_seconds",
    "execution_slices",
    "execution_recheck_signal",
    "max_fill_participation",
    "execution_partial_fill",
    "execution_max_child_fills",
    "execution_one_order_per_market",
    "execution_wait_for_price",
    "execution_reserve_parent_cash",
    "max_entry_price",
    "trade_fee_bps",
    "slippage_bps",
}


def _primary_config() -> BacktestConfig:
    return BacktestConfig(
        delay_seconds=300,
        skill_threshold=0.03,
        consensus_threshold=0.70,
        min_skilled_traders=1,
        max_single_trader_weight=1.0,
        min_edge=0.0,
        min_user_volume=10.0,
        max_trades_per_market=2,
        stable_min_price=0.8,
        lottery_min_price=0.1,
        lottery_max_price=0.5,
        stable_balance_fraction=0.2,
        lottery_lot_size=5.0,
        lottery_max_exposure_fraction=0.0,
        min_days_to_resolution=5.0,
        max_days_to_resolution=40.0,
        trade_fee_bps=0.0,
        slippage_bps=10.0,
        min_entry_price=0.0,
        max_entry_price=1.0,
        dynamic_price_at_consensus=1.0,
        dynamic_price_at_high_confidence=1.0,
        dynamic_high_confidence=0.97,
        max_market_fraction=0.02,
        max_balance_fraction=0.10,
        max_loss_per_trade_fraction=0.25,
        min_ticket_size=10.0,
        initial_balance=10_000.0,
        position_sizing="target_exposure_annualized",
        target_exposure_fraction=0.97,
        min_directional_traders=2,
        min_effective_directional_traders=1.25,
        max_directional_trader_weight=0.75,
        min_signal_mean_expert_history_markets=1.5,
        max_competitive_event_exposure_fraction=0.15,
        max_position_exposure_fraction=0.15,
    )


def test_grid_is_ordered_unique_and_returns_fresh_configs():
    base = _primary_config()
    grid = build_execution_candidate_grid(base)

    assert [name for name, _ in grid] == [preset.name for preset in EXECUTION_PRESETS]
    assert len({name for name, _ in grid}) == len(grid)
    assert all(config is not base for _, config in grid)


def test_every_preset_preserves_all_non_execution_fields():
    base_values = asdict(_primary_config())

    for name, candidate in build_execution_candidate_grid(_primary_config()):
        candidate_values = asdict(candidate)
        changed = {
            field
            for field, value in candidate_values.items()
            if value != base_values[field]
        }
        assert changed <= EXECUTION_FIELDS, name


def test_requested_one_dimensional_values_are_complete():
    base = _primary_config()
    delay_names = [
        "execution_delay_0s",
        "execution_delay_60s",
        "execution_delay_180s",
        "execution_delay_300s",
        "execution_delay_600s",
        "execution_delay_900s",
        "execution_delay_1800s",
    ]
    confirmation_names = [f"execution_confirm_{value}" for value in [1, 2, 3]]
    no_chase_names = [
        "execution_no_chase_0bps",
        "execution_no_chase_25bps",
        "execution_no_chase_50bps",
        "execution_no_chase_100bps",
        "execution_no_chase_250bps",
        "execution_no_chase_500bps",
    ]

    assert [get_execution_preset(name).apply(base).delay_seconds for name in delay_names] == [
        0,
        60,
        180,
        300,
        600,
        900,
        1800,
    ]
    assert [
        get_execution_preset(name).apply(base).execution_confirmation_trades
        for name in confirmation_names
    ] == [1, 2, 3]
    assert [
        get_execution_preset(name).apply(base).max_price_deterioration_bps
        for name in no_chase_names
    ] == [0.0, 25.0, 50.0, 100.0, 250.0, 500.0]


def test_direction_filter_has_finite_ttl_and_slice_grid_is_complete():
    base = _primary_config()
    direction = get_execution_preset(
        "execution_trade_filter_signal_direction_24h"
    ).apply(base)
    slices = [
        get_execution_preset(name).apply(base).execution_slices
        for name in ["execution_slices_1", "execution_slices_2"]
    ]

    assert direction.execution_trade_filter == "signal_direction"
    assert direction.pending_order_expiry_seconds == 86400
    assert slices == [1, 2]


def test_subset_preserves_requested_order_and_unknown_name_is_clear():
    grid = build_execution_candidate_grid(
        _primary_config(),
        ["execution_slices_2", "execution_delay_60s", "execution_confirm_2"],
    )

    assert [name for name, _ in grid] == [
        "execution_slices_2",
        "execution_delay_60s",
        "execution_confirm_2",
    ]
    with pytest.raises(KeyError, match="Unknown execution preset.*execution_delay_60s"):
        get_execution_preset("missing")


def test_capacity_grid_has_requested_participation_and_ttl_values():
    base = _primary_config()
    participation_names = [
        f"execution_partial_target_buy_{value}pct_2h"
        for value in [5, 10, 25, 50, 100]
    ]
    participation = [
        get_execution_preset(name).apply(base).max_fill_participation
        for name in participation_names
    ]
    ttl = [
        get_execution_preset(name).apply(base).pending_order_expiry_seconds
        for name in [
            "execution_partial_target_buy_25pct_30m",
            "execution_partial_target_buy_25pct_2h",
            "execution_partial_target_buy_25pct_6h",
        ]
    ]

    assert participation == [0.05, 0.10, 0.25, 0.50, 1.00]
    assert ttl == [1800, 7200, 21600]
    core = get_execution_preset(
        "execution_partial_target_buy_25pct_2h"
    ).apply(base)
    assert core.execution_trade_filter == "target_token_buy"
    assert core.execution_partial_fill is True
    assert core.execution_one_order_per_market is True
    assert core.execution_reserve_parent_cash is True

    passive_names = [
        f"execution_partial_target_sell_{value}pct_6h"
        for value in [5, 10, 25, 50, 100]
    ]
    passive = [
        get_execution_preset(name).apply(base)
        for name in passive_names
    ]
    assert [config.max_fill_participation for config in passive] == [
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
    ]
    assert all(config.execution_trade_filter == "target_token_sell" for config in passive)
    assert all(config.pending_order_expiry_seconds == 21600 for config in passive)

    token_names = [
        f"execution_partial_target_token_{value}pct_6h"
        for value in [5, 10, 25, 50, 100]
    ]
    token_configs = [get_execution_preset(name).apply(base) for name in token_names]
    assert [config.max_fill_participation for config in token_configs] == [
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
    ]
    assert all(config.execution_trade_filter == "target_token" for config in token_configs)
    token_ttls = [
        get_execution_preset(name).apply(base).pending_order_expiry_seconds
        for name in [
            "execution_partial_target_token_25pct_30m",
            "execution_partial_target_token_25pct_2h",
            "execution_partial_target_token_25pct_6h",
            "execution_partial_target_token_25pct_24h",
        ]
    ]
    assert token_ttls == [1800, 7200, 21600, 86400]

    extended_ttls = [
        get_execution_preset(name).apply(base).pending_order_expiry_seconds
        for name in [
            "execution_partial_target_token_25pct_48h",
            "execution_partial_target_token_25pct_72h",
        ]
    ]
    assert extended_ttls == [172800, 259200]

    entry_caps = [
        get_execution_preset(name).apply(base).max_entry_price
        for name in [
            "execution_partial_target_token_25pct_6h_entry_0995",
            "execution_partial_target_token_25pct_6h_entry_0998",
            "execution_partial_target_token_25pct_6h_entry_0999",
        ]
    ]
    assert entry_caps == [0.995, 0.998, 0.999]

    rechecked = get_execution_preset(
        "execution_partial_target_token_25pct_24h_recheck"
    ).apply(base)
    assert rechecked.execution_recheck_signal is True
    assert rechecked.max_fill_participation == 0.25
    assert rechecked.pending_order_expiry_seconds == 86400

    stressed = get_execution_preset(
        "execution_partial_target_token_25pct_24h_cost_20_30bps"
    ).apply(base)
    assert stressed.trade_fee_bps == 20.0
    assert stressed.slippage_bps == 30.0
    stressed_capped = get_execution_preset(
        "execution_partial_target_token_25pct_24h_entry_0998_cost_20_30bps"
    ).apply(base)
    assert stressed_capped.max_entry_price == 0.998
    assert stressed_capped.trade_fee_bps == 20.0
    assert stressed_capped.slippage_bps == 30.0
