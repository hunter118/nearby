"""Independent boundary checks; does not modify or run an archived replay."""
from dataclasses import asdict, replace

import numpy as np

from alpha.trader_skill import TraderSkillEstimator
from build_expanded_replay import utc
from causal_rule_engine import MaturityRule, ObservedFinalityBacktester, prefix_activation
from expand_market_universe import START
from models import TraderMarketSettlement
from run_causal_rule_search import timeline
from test_dynamic_universe_fast import FixedSkills, fixture


def economic_result(result):
    return {
        key: [asdict(row) for row in result[key]]
        for key in ('fills', 'closed_positions', 'open_positions')
    } | {'equity': result['equity_curve'], 'cash': result['balance']}


def test_uniform_scores_only_release_labels_at_availability_and_exclude_target():
    t, markets, *_ = fixture()
    available = utc(t + 100)
    rows = [TraderMarketSettlement('a', mid, score, 1e6, available)
            for mid, score in [('inactive', .3), ('fractional', .5), ('active', -1.)]]
    estimator = TraderSkillEstimator(markets, rows, similarity_mode='uniform')
    assert estimator.estimate('a', 'active', utc(t + 99)).weighted_score == 0
    estimate = estimator.estimate('a', 'active', available)
    assert estimate.supporting_markets == 2
    assert estimate.weighted_score == .4
    # Appending a spectacular future result cannot change an earlier score.
    extended = TraderSkillEstimator(markets, rows + [
        TraderMarketSettlement('a', 'fractional', 1., 1e15, utc(t + 101))
    ], similarity_mode='uniform')
    assert extended.estimate('a', 'active', available) == estimate


def test_uniform_no_dates_portfolio_is_independent_of_archive_metadata_and_vectors():
    t, markets, manifests, arrays, addresses, _, config = fixture()
    manifests = [m | {'label_usable': True} for m in manifests]
    histories = []
    for number in (1, 2):
        mid = f'history{number}'
        markets[mid] = replace(markets['inactive'], market_id=mid, resolved_at=utc(t - 10))
        for wallet in addresses:
            histories.append(TraderMarketSettlement(wallet, mid, .5, 1e6, utc(t - 10)))

    def replay(mutated):
        current = {mid: replace(m,
            question='Ravens vs. Bills' if mutated else m.question,
            category='sports' if mutated else m.category,
            created_at=utc(t + 10**9) if mutated else m.created_at,
            close_time=utc(t - 1) if mutated else m.close_time,
            active=False if mutated else m.active,
            volume=1e99 if mutated else m.volume) for mid, m in markets.items()}
        estimator = TraderSkillEstimator(current, histories, similarity_mode='uniform',
            precomputed_market_vectors=np.full((len(current), 7), -123. if mutated else 1.))
        engine = ObservedFinalityBacktester(current,
            timeline(current, manifests, arrays, addresses, np.arange(len(arrays['time']))),
            estimator, config, activation_times={'active': utc(t + 100)},
            planning_horizon_days=14, ignore_risk_text=True, observation_start=utc(START))
        return engine.run()

    original, modified = replay(False), replay(True)
    assert original['fills']
    assert economic_result(original) == economic_result(modified)
    assert all(row.semantic_risk_class == 'standard' for row in modified['fills'])


def test_pending_order_is_cancelled_before_same_second_executable_print():
    t, markets, _, _, addresses, _, config = fixture()
    markets = {'active': replace(markets['active'], resolved_at=utc(t + 410))}
    manifests = [dict(market_id='active', label_usable=True, payout_yes=1.)]
    arrays = dict(time=np.array([t + 100, t + 110, t + 410]),
        market=np.zeros(3, dtype=int), wallet=np.array([0, 1, 2]),
        side=np.zeros(3, dtype=int), price=np.full(3, .9), size=np.full(3, 100_000.))
    engine = ObservedFinalityBacktester(markets,
        timeline(markets, manifests, arrays, addresses, np.arange(3)),
        FixedSkills(), config, activation_times={'active': utc(t + 100)},
        observation_start=utc(START))
    result = engine.run()
    assert result['execution_order_requests']
    assert not result['fills'] and not engine.pending_orders
    assert result['balance'] == config.initial_balance
    assert engine.post_resolution_prints_skipped == 1


def test_pre_start_finality_blocks_orders_not_just_post_run_assertion():
    _, markets, _, _, addresses, _, config = fixture()
    markets = {'active': replace(markets['active'], resolved_at=utc(START - 1),
        close_time=utc(START + 20 * 86400))}
    manifests = [dict(market_id='active', label_usable=True, payout_yes=1.)]
    arrays = dict(time=START + np.array([1, 11, 101, 1001]),
        market=np.zeros(4, dtype=int), wallet=np.array([0, 1, 2, 2]),
        side=np.zeros(4, dtype=int), price=np.full(4, .9), size=np.full(4, 100_000.))
    engine = ObservedFinalityBacktester(markets,
        timeline(markets, manifests, arrays, addresses, np.arange(4)),
        FixedSkills(), config, activation_times={'active': utc(START)},
        observation_start=utc(START))
    result = engine.run()
    assert not result['fills'] and not result['execution_order_requests']
    assert not result['open_positions']
    assert engine.post_resolution_prints_skipped == 4


def test_maturity_uses_actual_no_token_dollars_and_completed_hour_boundary():
    rule = MaturityRule(min_observed_hours=0, min_recent_prints=1)
    times = np.array([10, 3600, 7200])
    size = np.full(3, 10_000_000.)
    yes = np.full(3, .95)
    token = np.full(3, .05)
    assert prefix_activation(times, size, yes, token, rule=rule, observed_until=7199) is None
    gate = prefix_activation(times, size, yes, token, rule=rule, observed_until=7200)
    assert gate['time'] == 7200
    assert gate['cumulative_notional'] == 1_000_000.
    assert gate['recent_prints'] == 2
