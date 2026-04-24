from alpha.signal import FlowObservation, decide_direction_from_flow
from models import Direction


def test_consensus_uses_skill_times_volume_weights():
    flow = [
        FlowObservation("u1", "m1", Direction.YES, volume=100.0, skill=0.6),
        FlowObservation("u2", "m1", Direction.YES, volume=20.0, skill=0.9),
        FlowObservation("u3", "m1", Direction.NO, volume=50.0, skill=0.2),
    ]
    decision = decide_direction_from_flow(
        flow=flow,
        skill_threshold=0.3,
        consensus_threshold=0.6,
    )
    assert decision.should_trade is True
    assert decision.direction == Direction.YES
    assert decision.confidence >= 0.6


def test_no_signal_when_high_skill_traders_do_not_reach_threshold():
    flow = [
        FlowObservation("u1", "m1", Direction.YES, volume=100.0, skill=0.31),
        FlowObservation("u2", "m1", Direction.NO, volume=100.0, skill=0.3),
    ]
    decision = decide_direction_from_flow(
        flow=flow,
        skill_threshold=0.3,
        consensus_threshold=0.7,
    )
    assert decision.should_trade is False
    assert decision.direction is None
