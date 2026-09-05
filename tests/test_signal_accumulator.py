from alpha.signal import FlowAccumulator, FlowObservation, decide_direction_from_flow
from models import Direction


def _assert_same(left, right):
    assert left.should_trade == right.should_trade
    assert left.direction == right.direction
    assert left.confidence == right.confidence
    assert left.total_weight == right.total_weight
    assert left.skilled_trader_count == right.skilled_trader_count
    assert left.max_single_trader_weight == right.max_single_trader_weight
    assert left.directional_trader_count == right.directional_trader_count
    assert left.effective_directional_traders == right.effective_directional_traders
    assert left.directional_concentration == right.directional_concentration


def test_incremental_flow_matches_full_rescan_after_each_observation():
    rows = [
        FlowObservation("a", "m", Direction.YES, 10.0, 0.2),
        FlowObservation("b", "m", Direction.NO, 4.0, 0.1),
        FlowObservation("a", "m", Direction.YES, 2.0, 0.2),
        FlowObservation("c", "m", Direction.NO, 100.0, 0.01),
    ]
    accumulator = FlowAccumulator(0.03, 0.7, 1, 1.0, "skill_volume")
    visible = []
    for row in rows:
        visible.append(row)
        accumulator.add(row)
        expected = decide_direction_from_flow(visible, 0.03, 0.7, 1, 1.0, "skill_volume")
        _assert_same(accumulator.decide(), expected)


def test_incremental_flow_removal_matches_rolling_window_rescan():
    rows = [
        FlowObservation("a", "m", Direction.YES, 10.0, 0.2),
        FlowObservation("b", "m", Direction.NO, 4.0, 0.1),
        FlowObservation("a", "m", Direction.NO, 3.0, 0.2),
    ]
    accumulator = FlowAccumulator(0.03, 0.55, 1, 1.0, "volume")
    for row in rows:
        accumulator.add(row)
    accumulator.remove(rows[0])
    expected = decide_direction_from_flow(rows[1:], 0.03, 0.55, 1, 1.0, "volume")
    _assert_same(accumulator.decide(), expected)


def test_directional_expert_count_and_concentration_gate_signal():
    one_dominant = [
        FlowObservation("a", "m", Direction.YES, 90.0, 0.2, 3.0, 0.7, 0.8, 0.2),
        FlowObservation("b", "m", Direction.YES, 10.0, 0.2, 3.0, 0.7, 0.8, 0.2),
    ]
    rejected = decide_direction_from_flow(
        one_dominant,
        0.03,
        0.7,
        min_directional_traders=2,
        min_effective_directional_traders=1.2,
        max_directional_trader_weight=0.8,
    )
    assert rejected.direction == Direction.YES
    assert rejected.directional_trader_count == 2
    assert rejected.directional_concentration == 0.9
    assert not rejected.should_trade

    balanced = [
        FlowObservation("a", "m", Direction.YES, 50.0, 0.2, 3.0, 0.7, 0.8, 0.2),
        FlowObservation("b", "m", Direction.YES, 50.0, 0.2, 3.0, 0.7, 0.8, 0.2),
    ]
    accepted = decide_direction_from_flow(
        balanced,
        0.03,
        0.7,
        min_directional_traders=2,
        min_effective_directional_traders=1.8,
        max_directional_trader_weight=0.6,
    )
    assert accepted.should_trade
    assert accepted.effective_directional_traders == 2.0


def test_semantic_expert_history_quality_gates_observations():
    rows = [
        FlowObservation("thin", "m", Direction.YES, 100.0, 0.2, 1.0, 0.7, 1.0, 0.0),
        FlowObservation("broad", "m", Direction.YES, 10.0, 0.2, 4.0, 0.7, 0.75, 0.2),
    ]
    decision = decide_direction_from_flow(
        rows,
        0.03,
        0.7,
        min_expert_effective_history_markets=2.0,
        min_expert_mean_similarity=0.5,
        min_expert_positive_history_fraction=0.6,
        max_expert_score_std=0.4,
    )
    assert decision.should_trade
    assert decision.skilled_trader_count == 1
    assert decision.directional_trader_count == 1
    assert decision.mean_expert_history_markets == 4.0


def test_weighted_history_notional_gate_applies_to_every_observation():
    rows = [
        FlowObservation(
            "thin",
            "m",
            Direction.YES,
            volume=100.0,
            skill=0.2,
            weighted_history_notional=5.0,
        ),
        FlowObservation(
            "qualified",
            "m",
            Direction.NO,
            volume=10.0,
            skill=0.2,
            weighted_history_notional=20.0,
        ),
    ]
    decision = decide_direction_from_flow(
        rows,
        0.03,
        0.7,
        min_weighted_history_notional=10.0,
    )
    assert decision.should_trade
    assert decision.direction == Direction.NO
    assert decision.skilled_trader_count == 1
