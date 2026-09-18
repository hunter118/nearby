from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from alpha.trader_skill import TraderSkillEstimator
from models import Direction, Market, TraderMarketSettlement
from run_similarity_kernel_study import KERNELS, Kernel, KernelSkillEstimator


def fixture_data():
    now = datetime(2026, 1, 1)
    m = Market("target", "Target", "x", now, now + timedelta(days=10),
               now + timedelta(days=11), Direction.YES, 0, False)
    markets = {key: replace(m, market_id=key) for key in ("target", "a", "b")}
    history = [TraderMarketSettlement("w", "a", .5, 100, now - timedelta(days=1)),
               TraderMarketSettlement("w", "b", -.5, 100, now - timedelta(days=1))]
    vectors = np.array([[1., 0.], [.8, .6], [.4, np.sqrt(.84)]])
    return now, markets, history, vectors


def test_kernel_definitions_are_nonnegative_monotone_and_thresholded():
    grid = np.linspace(-1, 1, 201)
    for k in KERNELS.values():
        if k is None:
            continue
        weights = k.apply(grid)
        assert (weights >= 0).all() and (weights <= 1).all()
        assert (np.diff(weights) >= 0).all()
        assert (weights[grid <= k.threshold] == 0).all()
    np.testing.assert_allclose(Kernel(.5).apply([.4, .5, .6, .8, 1.]), [0, 0, .2, .6, 1])
    np.testing.assert_allclose(Kernel(.5, 2).apply([.4, .5, .6, .8, 1.]), [0, 0, .04, .36, 1])
    with pytest.raises(ValueError):
        Kernel(1)
    with pytest.raises(ValueError):
        Kernel(power=float("nan"))


def test_identity_exactly_preserves_baseline_estimate():
    now, markets, history, vectors = fixture_data()
    original = TraderSkillEstimator(markets, history, precomputed_market_vectors=vectors)
    research = KernelSkillEstimator(markets, history, kernel=Kernel(), precomputed_market_vectors=vectors)
    assert original.estimate("w", "target", now) == research.estimate("w", "target", now)


def test_locality_and_constant_scale_have_distinct_effects():
    now, markets, history, vectors = fixture_data()
    estimates = []
    for norm in (True, False):
        e = KernelSkillEstimator(markets, history, kernel=Kernel(.5, normalize=norm),
                                precomputed_market_vectors=vectors)
        estimates.append(e.estimate("w", "target", now))
        assert e.market_similarity("target", "a") == pytest.approx(.8)
        assert e.market_similarity("target", "b") == pytest.approx(.4)
    normalized, unscaled = estimates
    assert normalized.weighted_score == unscaled.weighted_score == .5
    assert normalized.effective_history_markets == unscaled.effective_history_markets == 1
    assert normalized.weighted_history_notional == pytest.approx(60)
    assert unscaled.weighted_history_notional == pytest.approx(30)


def test_excludes_future_and_target_and_abstains_without_relevant_history():
    now, markets, history, vectors = fixture_data()
    history += [TraderMarketSettlement("w", "target", 1, 100000, now - timedelta(days=1)),
                TraderMarketSettlement("w", "a", 1, 100000, now + timedelta(days=1))]
    e = KernelSkillEstimator(markets, history, kernel=Kernel(.9), precomputed_market_vectors=vectors)
    skill = e.estimate("w", "target", now)
    assert skill.weighted_history_notional == skill.weighted_score == skill.supporting_markets == 0
    e = KernelSkillEstimator(markets, history, kernel=Kernel(.5), precomputed_market_vectors=vectors)
    assert e.estimate("w", "target", now).weighted_score == .5
