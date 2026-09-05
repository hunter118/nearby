from datetime import datetime, timedelta

import pytest

from alpha.trader_skill import SkillEstimate
from alpha.semantic_risk import semantic_risk_class
from backtest.engine import BacktestConfig, EventDrivenBacktester
from models import (
    Direction,
    Market,
    ResolutionEvent,
    Side,
    TimelineEvent,
    TradeEvent,
)


class _Estimator:
    def estimate(self, trader_id, target_market_id, as_of):
        return SkillEstimate(
            trader_id,
            target_market_id,
            as_of,
            weighted_score=0.2,
            weighted_history_notional=100.0,
            supporting_markets=4,
            effective_history_markets=3.0,
            mean_similarity=0.7,
            positive_history_weight_fraction=0.8,
            weighted_score_std=0.2,
        )

    def market_similarity(self, left_market_id, right_market_id):
        return 1.0


class _MixedHistoryEstimator(_Estimator):
    def estimate(self, trader_id, target_market_id, as_of):
        estimate = super().estimate(trader_id, target_market_id, as_of)
        return SkillEstimate(
            estimate.trader_id,
            estimate.market_id,
            estimate.as_of,
            weighted_score=estimate.weighted_score,
            weighted_history_notional=5.0 if trader_id == "thin" else 100.0,
            supporting_markets=estimate.supporting_markets,
            effective_history_markets=estimate.effective_history_markets,
            mean_similarity=estimate.mean_similarity,
            positive_history_weight_fraction=estimate.positive_history_weight_fraction,
            weighted_score_std=estimate.weighted_score_std,
        )


def _config(**overrides):
    values = dict(
        delay_seconds=0,
        skill_threshold=0.03,
        consensus_threshold=0.7,
        min_skilled_traders=1,
        max_single_trader_weight=1.0,
        min_edge=0.0,
        min_user_volume=10.0,
        max_trades_per_market=1,
        stable_min_price=0.8,
        lottery_min_price=0.1,
        lottery_max_price=0.5,
        stable_balance_fraction=0.2,
        lottery_lot_size=5.0,
        lottery_max_exposure_fraction=0.0,
        min_days_to_resolution=5.0,
        max_days_to_resolution=40.0,
        trade_fee_bps=0.0,
        slippage_bps=0.0,
        min_entry_price=0.0,
        max_entry_price=1.0,
        dynamic_price_at_consensus=1.0,
        dynamic_price_at_high_confidence=1.0,
        dynamic_high_confidence=0.97,
        max_market_fraction=1.0,
        max_balance_fraction=1.0,
        max_loss_per_trade_fraction=0.02,
        min_ticket_size=1.0,
        initial_balance=10_000.0,
    )
    values.update(overrides)
    return BacktestConfig(**values)


def _market(market_id, start, question=None):
    return Market(
        market_id=market_id,
        question=question or f"Question {market_id}",
        category="test",
        created_at=start - timedelta(days=1),
        close_time=start + timedelta(days=10),
        resolved_at=start + timedelta(days=11),
        resolution=Direction.YES,
        volume=1_000.0,
        active=False,
    )


def _trade(trade_id, market_id, trader_id, ts):
    return TimelineEvent(
        "trade",
        ts,
        TradeEvent(trade_id, market_id, trader_id, Side.BUY_YES, 0.9, 10.0, ts),
    )


def _priced_trade(trade_id, market_id, trader_id, ts, side, price_yes):
    return TimelineEvent(
        "trade",
        ts,
        TradeEvent(trade_id, market_id, trader_id, side, price_yes, 10.0, ts),
    )


def test_loss_cap_and_signal_diagnostics_reach_the_fill():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("1", "m1", "alice", start),
        _trade("2", "m1", "bob", start + timedelta(seconds=1)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            enforce_risk_caps=True,
            apply_market_volume_cap=False,
            apply_balance_cap=False,
            apply_loss_cap=True,
        ),
    ).run()

    assert len(result["fills"]) == 1
    fill = result["fills"][0]
    assert fill.notional == pytest.approx(200.0)
    assert fill.directional_trader_count == 1
    assert fill.effective_directional_traders == pytest.approx(1.0)
    assert fill.mean_expert_history_markets == pytest.approx(3.0)
    assert fill.mean_positive_history_fraction == pytest.approx(0.8)


def test_every_consensus_observation_meets_weighted_history_minimum():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("thin", "m1", "thin", start, Side.BUY_YES, 0.9),
        _priced_trade(
            "qualified",
            "m1",
            "qualified",
            start + timedelta(seconds=1),
            Side.BUY_NO,
            0.1,
        ),
        _priced_trade(
            "fill",
            "m1",
            "liquidity",
            start + timedelta(seconds=2),
            Side.BUY_NO,
            0.1,
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _MixedHistoryEstimator(),
        _config(),
    ).run()

    assert len(result["fills"]) == 1
    assert result["fills"][0].direction == Direction.NO


def test_daily_equity_mode_records_the_last_event_of_each_utc_day():
    start = datetime(2026, 1, 1, 12)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("signal", "m1", "alice", start),
        _trade("fill", "m1", "bob", start + timedelta(hours=1)),
        _trade("mark", "m1", "carol", start + timedelta(days=1)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(equity_record_interval=0),
    ).run()

    assert [row["ts"] for row in result["equity_curve"]] == [
        (start + timedelta(hours=1)).isoformat(sep=" "),
        (start + timedelta(days=1)).isoformat(sep=" "),
    ]


def test_partial_parent_orders_reserve_cash_at_submission():
    start = datetime(2026, 1, 1)
    markets = {mid: _market(mid, start) for mid in ["m1", "m2"]}
    timeline = [
        _trade("m1-signal", "m1", "alice", start),
        _trade("m2-signal", "m2", "bob", start + timedelta(seconds=1)),
        _trade("m1-fill", "m1", "carol", start + timedelta(seconds=10)),
        _trade("m2-fill", "m2", "dave", start + timedelta(seconds=11)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            delay_seconds=5,
            stable_balance_fraction=0.8,
            execution_partial_fill=True,
            execution_max_child_fills=10,
            execution_one_order_per_market=True,
            execution_reserve_parent_cash=True,
        ),
    ).run()

    assert result["execution_order_requests"]["m1"] == pytest.approx(8_000.0)
    assert result["execution_order_requests"]["m2"] == pytest.approx(2_000.0)
    assert sum(fill.notional for fill in result["fills"]) == pytest.approx(10_000.0)


def test_semantic_cluster_cap_limits_correlated_market_exposure():
    start = datetime(2026, 1, 1)
    markets = {mid: _market(mid, start) for mid in ["m1", "m2"]}
    timeline = [
        _trade("1", "m1", "alice", start),
        _trade("2", "m1", "bob", start + timedelta(seconds=1)),
        _trade("3", "m2", "carol", start + timedelta(seconds=2)),
        _trade("4", "m2", "dave", start + timedelta(seconds=3)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            semantic_cluster_similarity_threshold=0.7,
            max_semantic_cluster_exposure_fraction=0.25,
        ),
    ).run()

    assert [fill.notional for fill in result["fills"]] == pytest.approx([2_000.0, 500.0])


def test_competitive_event_semantics_and_exposure_cap():
    assert semantic_risk_class("Ravens vs. Bills") == "competitive_event"
    assert (
        semantic_risk_class("Will the New York Knicks win the 2026 NBA Finals?")
        == "competitive_event"
    )
    assert semantic_risk_class("Will inflation fall below 3%?") == "standard"
    assert semantic_risk_class("2 Trump vs. Harris debates before election?") == "standard"

    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start, "Ravens vs. Bills")}
    timeline = [
        _trade("1", "m1", "alice", start),
        _trade("2", "m1", "bob", start + timedelta(seconds=1)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(max_competitive_event_exposure_fraction=0.05),
    ).run()

    assert len(result["fills"]) == 1
    assert result["fills"][0].notional == pytest.approx(500.0)
    assert result["fills"][0].semantic_risk_class == "competitive_event"


def test_signal_level_effective_history_gate_can_reject_thin_consensus():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("1", "m1", "alice", start),
        _trade("2", "m1", "bob", start + timedelta(seconds=1)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(min_signal_mean_expert_history_markets=4.0),
    ).run()

    assert result["fills"] == []


def test_general_position_cap_uses_total_equity_not_remaining_cash():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("1", "m1", "alice", start),
        _trade("2", "m1", "bob", start + timedelta(seconds=1)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(max_position_exposure_fraction=0.10),
    ).run()

    assert result["fills"][0].notional == pytest.approx(1_000.0)


def test_execution_confirmation_uses_only_later_trade_prints():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("1", "m1", "alice", start),
        _trade("2", "m1", "bob", start + timedelta(seconds=1)),
        _trade("3", "m1", "carol", start + timedelta(seconds=2)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(execution_confirmation_trades=2),
    ).run()

    assert len(result["fills"]) == 1
    assert result["fills"][0].filled_at == start + timedelta(seconds=2)


def test_execution_direction_filter_waits_for_confirming_flow():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("1", "m1", "alice", start),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.BUY_NO, 0.1
        ),
        _trade("3", "m1", "carol", start + timedelta(seconds=2)),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(execution_trade_filter="signal_direction"),
    ).run()

    assert len(result["fills"]) == 1
    assert result["fills"][0].filled_at == start + timedelta(seconds=2)


def test_no_chase_policy_is_fill_or_kill_at_first_due_print():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.BUY_YES, 0.91
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=2), Side.BUY_YES, 0.89
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(max_price_deterioration_bps=0.0),
    ).run()

    assert result["fills"] == []


def test_two_slice_execution_averages_two_post_signal_prices():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.BUY_YES, 0.90
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=2), Side.BUY_YES, 0.85
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(execution_slices=2, max_trades_per_market=2),
    ).run()

    assert [fill.notional for fill in result["fills"]] == pytest.approx([1_000.0, 1_000.0])
    assert len(result["open_positions"]) == 1
    expected_price = 2_000.0 / (1_000.0 / 0.90 + 1_000.0 / 0.85)
    assert result["open_positions"][0].avg_entry_price == pytest.approx(expected_price)


def test_execution_recheck_rejects_consensus_that_reverses_before_fill():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=5), Side.BUY_NO, 0.10
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=10), Side.BUY_YES, 0.90
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(delay_seconds=10, execution_recheck_signal=True),
    ).run()

    assert result["fills"] == []


def test_resolution_removes_unfilled_execution_order():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _trade("1", "m1", "alice", start),
        TimelineEvent(
            "resolution",
            start + timedelta(seconds=2),
            ResolutionEvent("m1", start + timedelta(seconds=2), Direction.YES),
        ),
    ]
    backtester = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            delay_seconds=10,
            execution_trade_filter="signal_direction",
            pending_order_expiry_seconds=60,
        ),
    )
    backtester.run()

    assert backtester.pending_orders == []


def test_partial_execution_respects_each_print_participation_limit():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.BUY_YES, 0.90
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=2), Side.BUY_YES, 0.90
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            execution_partial_fill=True,
            execution_max_child_fills=2,
            max_fill_participation=0.10,
        ),
    ).run()

    assert len(result["fills"]) == 2
    assert [fill.quantity for fill in result["fills"]] == pytest.approx([1.0, 1.0])
    assert all(
        fill.quantity <= 0.10 * fill.source_trade_size
        for fill in result["fills"]
    )
    assert result["open_positions"][0].quantity == pytest.approx(2.0)


def test_target_token_buy_filter_skips_bid_like_print():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.SELL_YES, 0.90
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=2), Side.BUY_YES, 0.90
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(execution_trade_filter="target_token_buy"),
    ).run()

    assert len(result["fills"]) == 1
    assert result["fills"][0].filled_at == start + timedelta(seconds=2)
    assert result["fills"][0].source_trade_side == Side.BUY_YES.value


def test_passive_target_token_sell_waits_for_limit_eligible_print():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.SELL_YES, 0.91
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=2), Side.SELL_YES, 0.89
        ),
    ]
    result = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            execution_trade_filter="target_token_sell",
            max_price_deterioration_bps=0.0,
            execution_wait_for_price=True,
            pending_order_expiry_seconds=60,
            execution_one_order_per_market=True,
        ),
    ).run()

    assert len(result["fills"]) == 1
    assert result["fills"][0].fill_price == pytest.approx(0.89)
    assert result["fills"][0].source_trade_side == Side.SELL_YES.value


def test_one_order_lifecycle_does_not_revive_expired_stale_signal():
    start = datetime(2026, 1, 1)
    markets = {"m1": _market("m1", start)}
    timeline = [
        _priced_trade("1", "m1", "alice", start, Side.BUY_YES, 0.90),
        _priced_trade(
            "2", "m1", "bob", start + timedelta(seconds=1), Side.SELL_YES, 0.90
        ),
        _priced_trade(
            "3", "m1", "carol", start + timedelta(seconds=2), Side.BUY_YES, 0.90
        ),
    ]
    backtester = EventDrivenBacktester(
        markets,
        timeline,
        _Estimator(),
        _config(
            execution_trade_filter="target_token_buy",
            pending_order_expiry_seconds=1,
            execution_one_order_per_market=True,
        ),
    )
    result = backtester.run()

    assert result["fills"] == []
    assert backtester.pending_orders == []
