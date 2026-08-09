"""Reproducible semantic-risk presets for long-window experiments.

The preset grid is deliberately small and ordered.  Single-mechanism presets
make it possible to attribute an improvement, while the three ``robust_*``
presets form a local loose/balanced/strict neighborhood for the final
robustness check.  Applying a preset always returns a fresh
:class:`BacktestConfig` via :func:`dataclasses.replace` and never mutates the
caller's baseline configuration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from backtest.engine import BacktestConfig


@dataclass(frozen=True)
class RiskPreset:
    """A named, documented set of ``BacktestConfig`` overrides."""

    name: str
    family: str
    strictness: str
    description: str
    overrides: tuple[tuple[str, Any], ...]

    def apply(self, base_config: BacktestConfig) -> BacktestConfig:
        """Return a fresh configuration with this preset applied."""

        return replace(base_config, **dict(self.overrides))

    def override_dict(self) -> dict[str, Any]:
        """Return a copy suitable for result manifests and tables."""

        return dict(self.overrides)


def _overrides(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(values.items())


RISK_PRESETS: tuple[RiskPreset, ...] = (
    RiskPreset(
        name="baseline",
        family="baseline",
        strictness="none",
        description="Unchanged strategy configuration used as the comparison point.",
        overrides=(),
    ),
    RiskPreset(
        name="loss_cap_2pct",
        family="loss_cap",
        strictness="loose",
        description="Cap worst-case notional loss per new position at 2% of cash.",
        overrides=_overrides(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
            max_loss_per_trade_fraction=0.02,
        ),
    ),
    RiskPreset(
        name="loss_cap_1p5pct",
        family="loss_cap",
        strictness="balanced",
        description="Cap worst-case notional loss per new position at 1.5% of cash.",
        overrides=_overrides(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
            max_loss_per_trade_fraction=0.015,
        ),
    ),
    RiskPreset(
        name="loss_cap_1pct",
        family="loss_cap",
        strictness="strict",
        description="Cap worst-case notional loss per new position at 1% of cash.",
        overrides=_overrides(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
            max_loss_per_trade_fraction=0.01,
        ),
    ),
    RiskPreset(
        name="consensus_loose",
        family="consensus",
        strictness="loose",
        description=(
            "Require two same-direction experts, 1.25 effective experts, and no "
            "expert above 75% of directional weight."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
        ),
    ),
    RiskPreset(
        name="consensus_balanced",
        family="consensus",
        strictness="balanced",
        description=(
            "Require two same-direction experts, 1.6 effective experts, and no "
            "expert above 65% of directional weight."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.6,
            max_directional_trader_weight=0.65,
        ),
    ),
    RiskPreset(
        name="consensus_strict",
        family="consensus",
        strictness="strict",
        description=(
            "Require three same-direction experts, two effective experts, and no "
            "expert above 55% of directional weight."
        ),
        overrides=_overrides(
            min_directional_traders=3,
            min_effective_directional_traders=2.0,
            max_directional_trader_weight=0.55,
        ),
    ),
    RiskPreset(
        name="consensus_history_1p5",
        family="consensus_history",
        strictness="balanced",
        description=(
            "Keep the loose independent-consensus rule and require the signal's "
            "weight-averaged expert history to span at least 1.5 effective related markets."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
        ),
    ),
    RiskPreset(
        name="semantic_event_cap_15pct",
        family="semantic_event",
        strictness="loose",
        description=(
            "Consensus/history filter plus a 15% equity cap for head-to-head or "
            "winner-take-all competitive sports events."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="semantic_event_cap_10pct",
        family="semantic_event",
        strictness="balanced",
        description=(
            "Consensus/history filter plus a 10% equity cap for concentrated "
            "competitive-event risk."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.10,
        ),
    ),
    RiskPreset(
        name="semantic_event_cap_5pct",
        family="semantic_event",
        strictness="strict",
        description=(
            "Consensus/history filter plus a conservative 5% equity cap for "
            "concentrated competitive-event risk."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.05,
        ),
    ),
    RiskPreset(
        name="tiered_position_cap_25pct",
        family="tiered_position",
        strictness="loose",
        description=(
            "Primary tiered candidate: consensus/history filter, 15% competitive-event "
            "cap, and a 25% equity ceiling for every individual position."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.25,
        ),
    ),
    RiskPreset(
        name="tiered_position_cap_20pct",
        family="tiered_position",
        strictness="balanced",
        description=(
            "Consensus/history filter with 15% competitive-event and 20% general "
            "single-position equity ceilings."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.20,
        ),
    ),
    RiskPreset(
        name="tiered_position_cap_15pct",
        family="tiered_position",
        strictness="strict",
        description=(
            "Consensus/history filter with a uniform 15% ceiling, retaining the "
            "competitive-event classification for diagnostics."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    # One-at-a-time local sensitivity checks around ``tiered_position_cap_15pct``.
    # Each row below repeats the primary specification and changes exactly one
    # hyperparameter.  Keeping these presets explicit makes the robustness table
    # reproducible without introducing a combinatorial search.
    RiskPreset(
        name="hp_directional_traders_3",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with at least three same-direction experts.",
        overrides=_overrides(
            min_directional_traders=3,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_directional_traders_4",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with at least four same-direction experts.",
        overrides=_overrides(
            min_directional_traders=4,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_effective_traders_1p0",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with a 1.0 effective-expert minimum.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.0,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_effective_traders_1p5",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with a 1.5 effective-expert minimum.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.5,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_max_concentration_0p65",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with a 65% maximum directional expert share.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.65,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_max_concentration_0p85",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with an 85% maximum directional expert share.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.85,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_mean_history_1p0",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with one mean effective related market required.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.0,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_mean_history_2p0",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with two mean effective related markets required.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=2.0,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
        ),
    ),
    RiskPreset(
        name="hp_position_cap_10pct",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with a 10% general single-position cap.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.10,
        ),
    ),
    RiskPreset(
        name="hp_position_cap_20pct",
        family="hyperparameter_robustness",
        strictness="local",
        description="Primary specification with a 20% general single-position cap.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.20,
        ),
    ),
    RiskPreset(
        name="tiered_15pct_cost_stress",
        family="robustness",
        strictness="stress",
        description=(
            "Primary 15% tiered specification with 50 bps entry fees and 50 bps slippage."
        ),
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
            trade_fee_bps=50.0,
            slippage_bps=50.0,
        ),
    ),
    RiskPreset(
        name="tiered_15pct_delay_30m",
        family="robustness",
        strictness="stress",
        description="Primary 15% tiered specification with a 30-minute execution delay.",
        overrides=_overrides(
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_signal_mean_expert_history_markets=1.5,
            max_competitive_event_exposure_fraction=0.15,
            max_position_exposure_fraction=0.15,
            delay_seconds=1800,
        ),
    ),
    RiskPreset(
        name="history_quality_loose",
        family="history_quality",
        strictness="loose",
        description=(
            "Keep experts with at least 1.25 effective related markets, mean semantic "
            "similarity 0.10, 52% positive historical weight, and score volatility at "
            "most 0.80."
        ),
        overrides=_overrides(
            min_expert_effective_history_markets=1.25,
            min_expert_mean_similarity=0.10,
            min_expert_positive_history_fraction=0.52,
            max_expert_score_std=0.80,
        ),
    ),
    RiskPreset(
        name="history_quality_balanced",
        family="history_quality",
        strictness="balanced",
        description=(
            "Keep experts with at least 1.75 effective related markets, mean semantic "
            "similarity 0.20, 55% positive historical weight, and score volatility at "
            "most 0.70."
        ),
        overrides=_overrides(
            min_expert_effective_history_markets=1.75,
            min_expert_mean_similarity=0.20,
            min_expert_positive_history_fraction=0.55,
            max_expert_score_std=0.70,
        ),
    ),
    RiskPreset(
        name="history_quality_strict",
        family="history_quality",
        strictness="strict",
        description=(
            "Keep experts with at least 2.5 effective related markets, mean semantic "
            "similarity 0.30, 60% positive historical weight, and score volatility at "
            "most 0.60."
        ),
        overrides=_overrides(
            min_expert_effective_history_markets=2.5,
            min_expert_mean_similarity=0.30,
            min_expert_positive_history_fraction=0.60,
            max_expert_score_std=0.60,
        ),
    ),
    RiskPreset(
        name="robust_loose",
        family="robust",
        strictness="loose",
        description=(
            "Combine the loose 2% loss, consensus, expert-history, and correlated "
            "semantic-exposure controls."
        ),
        overrides=_overrides(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
            max_loss_per_trade_fraction=0.02,
            min_directional_traders=2,
            min_effective_directional_traders=1.25,
            max_directional_trader_weight=0.75,
            min_expert_effective_history_markets=1.25,
            min_expert_mean_similarity=0.10,
            min_expert_positive_history_fraction=0.52,
            max_expert_score_std=0.80,
            semantic_cluster_similarity_threshold=0.85,
            max_semantic_cluster_exposure_fraction=0.20,
        ),
    ),
    RiskPreset(
        name="robust_balanced",
        family="robust",
        strictness="balanced",
        description=(
            "Balanced primary candidate: 1.5% loss cap plus moderate consensus, "
            "expert-history, and semantic-cluster controls."
        ),
        overrides=_overrides(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
            max_loss_per_trade_fraction=0.015,
            min_directional_traders=2,
            min_effective_directional_traders=1.6,
            max_directional_trader_weight=0.65,
            min_expert_effective_history_markets=1.75,
            min_expert_mean_similarity=0.20,
            min_expert_positive_history_fraction=0.55,
            max_expert_score_std=0.70,
            semantic_cluster_similarity_threshold=0.80,
            max_semantic_cluster_exposure_fraction=0.12,
        ),
    ),
    RiskPreset(
        name="robust_strict",
        family="robust",
        strictness="strict",
        description=(
            "Strict neighbor: 1% loss cap, three-expert support, stronger history "
            "quality, and an 8% semantic-cluster exposure ceiling."
        ),
        overrides=_overrides(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
            max_loss_per_trade_fraction=0.01,
            min_directional_traders=3,
            min_effective_directional_traders=2.0,
            max_directional_trader_weight=0.55,
            min_expert_effective_history_markets=2.5,
            min_expert_mean_similarity=0.30,
            min_expert_positive_history_fraction=0.60,
            max_expert_score_std=0.60,
            semantic_cluster_similarity_threshold=0.75,
            max_semantic_cluster_exposure_fraction=0.08,
        ),
    ),
)

_PRESET_BY_NAME = {preset.name: preset for preset in RISK_PRESETS}


def get_risk_preset(name: str) -> RiskPreset:
    """Look up one preset, raising a useful error for misspelled names."""

    try:
        return _PRESET_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(_PRESET_BY_NAME)
        raise KeyError(f"Unknown risk preset {name!r}; available: {available}") from exc


def apply_risk_preset(base_config: BacktestConfig, name: str) -> BacktestConfig:
    """Apply a named preset without mutating ``base_config``."""

    return get_risk_preset(name).apply(base_config)


def build_risk_candidate_grid(
    base_config: BacktestConfig,
    preset_names: Iterable[str] | None = None,
) -> list[tuple[str, BacktestConfig]]:
    """Build an ordered ``(experiment_name, config)`` grid for a runner.

    Passing ``preset_names`` selects a reproducible subset while preserving the
    caller's requested order.  With no selection, the full documented grid is
    returned in ``RISK_PRESETS`` order.
    """

    selected = (
        RISK_PRESETS
        if preset_names is None
        else tuple(get_risk_preset(name) for name in preset_names)
    )
    return [(preset.name, preset.apply(base_config)) for preset in selected]
