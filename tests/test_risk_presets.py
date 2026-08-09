from dataclasses import asdict

import pytest

from alpha.risk_presets import (
    RISK_PRESETS,
    apply_risk_preset,
    build_risk_candidate_grid,
    get_risk_preset,
)
from backtest.engine import BacktestConfig


def _base_config() -> BacktestConfig:
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
    )


def test_full_grid_is_ordered_unique_and_does_not_mutate_baseline():
    base = _base_config()
    before = asdict(base)

    grid = build_risk_candidate_grid(base)

    assert [name for name, _ in grid] == [preset.name for preset in RISK_PRESETS]
    assert len({name for name, _ in grid}) == len(grid)
    assert all(config is not base for _, config in grid)
    assert asdict(base) == before
    assert grid[0][0] == "baseline"
    assert asdict(grid[0][1]) == before


def test_loss_cap_presets_isolate_and_order_the_loss_limit():
    base = _base_config()
    names = ["loss_cap_2pct", "loss_cap_1p5pct", "loss_cap_1pct"]
    configs = [apply_risk_preset(base, name) for name in names]

    assert [config.max_loss_per_trade_fraction for config in configs] == [
        0.02,
        0.015,
        0.01,
    ]
    assert all(config.enforce_risk_caps for config in configs)
    assert all(config.apply_loss_cap for config in configs)
    assert all(not config.apply_market_volume_cap for config in configs)
    assert all(not config.apply_balance_cap for config in configs)


def test_consensus_and_history_families_tighten_monotonically():
    base = _base_config()
    consensus = [
        apply_risk_preset(base, f"consensus_{level}")
        for level in ["loose", "balanced", "strict"]
    ]
    history = [
        apply_risk_preset(base, f"history_quality_{level}")
        for level in ["loose", "balanced", "strict"]
    ]

    assert [config.min_directional_traders for config in consensus] == [2, 2, 3]
    assert [config.min_effective_directional_traders for config in consensus] == [
        1.25,
        1.6,
        2.0,
    ]
    assert [config.max_directional_trader_weight for config in consensus] == [
        0.75,
        0.65,
        0.55,
    ]
    assert [config.min_expert_effective_history_markets for config in history] == [
        1.25,
        1.75,
        2.5,
    ]
    assert [config.min_expert_mean_similarity for config in history] == [0.10, 0.20, 0.30]
    assert [config.min_expert_positive_history_fraction for config in history] == [
        0.52,
        0.55,
        0.60,
    ]
    assert [config.max_expert_score_std for config in history] == [0.80, 0.70, 0.60]


def test_robust_grid_combines_all_controls_and_has_local_neighbors():
    base = _base_config()
    robust = [
        apply_risk_preset(base, f"robust_{level}")
        for level in ["loose", "balanced", "strict"]
    ]

    assert all(config.enforce_risk_caps and config.apply_loss_cap for config in robust)
    assert all(config.min_directional_traders >= 2 for config in robust)
    assert all(config.min_expert_effective_history_markets > 1.0 for config in robust)
    assert all(config.semantic_cluster_similarity_threshold is not None for config in robust)
    assert [config.max_loss_per_trade_fraction for config in robust] == [0.02, 0.015, 0.01]
    assert [config.max_semantic_cluster_exposure_fraction for config in robust] == [
        0.20,
        0.12,
        0.08,
    ]


def test_runner_can_select_a_reproducible_subset_in_requested_order():
    grid = build_risk_candidate_grid(
        _base_config(),
        ["robust_balanced", "baseline", "loss_cap_1pct"],
    )

    assert [name for name, _ in grid] == [
        "robust_balanced",
        "baseline",
        "loss_cap_1pct",
    ]
    assert get_risk_preset("robust_balanced").family == "robust"
    assert get_risk_preset("robust_balanced").override_dict()[
        "max_loss_per_trade_fraction"
    ] == 0.015


def test_semantic_event_neighbors_share_consensus_and_vary_only_the_cap():
    base = _base_config()
    configs = [
        apply_risk_preset(base, name)
        for name in [
            "semantic_event_cap_15pct",
            "semantic_event_cap_10pct",
            "semantic_event_cap_5pct",
        ]
    ]

    assert [config.max_competitive_event_exposure_fraction for config in configs] == [
        0.15,
        0.10,
        0.05,
    ]
    assert all(config.min_directional_traders == 2 for config in configs)
    assert all(config.min_signal_mean_expert_history_markets == 1.5 for config in configs)


def test_tiered_position_neighbors_tighten_the_general_cap():
    base = _base_config()
    configs = [
        apply_risk_preset(base, name)
        for name in [
            "tiered_position_cap_25pct",
            "tiered_position_cap_20pct",
            "tiered_position_cap_15pct",
        ]
    ]

    assert [config.max_position_exposure_fraction for config in configs] == [
        0.25,
        0.20,
        0.15,
    ]
    assert all(config.max_competitive_event_exposure_fraction == 0.15 for config in configs)


def test_hyperparameter_robustness_presets_change_one_primary_parameter_at_a_time():
    base = _base_config()
    primary = apply_risk_preset(base, "tiered_position_cap_15pct")
    varied_fields = {
        "hp_directional_traders_3": "min_directional_traders",
        "hp_directional_traders_4": "min_directional_traders",
        "hp_effective_traders_1p0": "min_effective_directional_traders",
        "hp_effective_traders_1p5": "min_effective_directional_traders",
        "hp_max_concentration_0p65": "max_directional_trader_weight",
        "hp_max_concentration_0p85": "max_directional_trader_weight",
        "hp_mean_history_1p0": "min_signal_mean_expert_history_markets",
        "hp_mean_history_2p0": "min_signal_mean_expert_history_markets",
        "hp_position_cap_10pct": "max_position_exposure_fraction",
        "hp_position_cap_20pct": "max_position_exposure_fraction",
    }

    primary_values = asdict(primary)
    for preset_name, expected_field in varied_fields.items():
        candidate_values = asdict(apply_risk_preset(base, preset_name))
        changed = {
            field
            for field, value in candidate_values.items()
            if value != primary_values[field]
        }
        assert changed == {expected_field}


def test_unknown_preset_error_lists_available_names():
    with pytest.raises(KeyError, match="Unknown risk preset.*robust_balanced"):
        get_risk_preset("does_not_exist")
