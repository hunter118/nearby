from datetime import datetime, timedelta, timezone

from alpha.trader_skill import TraderSkillEstimator
from features.embeddings import SimilarityConfig
from models import Direction, Market, TraderMarketSettlement


def test_skill_estimator_only_uses_settled_history_before_as_of():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)
    t3 = t0 + timedelta(hours=3)

    markets = {
        "m1": Market(
            market_id="m1",
            question="Will BTC be above 100k by year end?",
            category="crypto",
            created_at=t0,
            close_time=None,
            resolved_at=t2,
            resolution=Direction.YES,
            volume=1000.0,
            active=False,
        ),
        "m2": Market(
            market_id="m2",
            question="Will ETH ETF approval happen in Q4?",
            category="crypto",
            created_at=t1,
            close_time=None,
            resolved_at=None,
            resolution=None,
            volume=1000.0,
            active=True,
        ),
    }
    settlements = [
        TraderMarketSettlement(
            trader_id="alice",
            market_id="m1",
            score=0.5,
            notional=200.0,
            settled_at=t2,
        )
    ]
    estimator = TraderSkillEstimator(
        markets=markets,
        settlements=settlements,
        min_weighted_history=0.0,
        similarity_config=SimilarityConfig(positive_similarity_only=True, similarity_floor=0.0),
    )

    before_settlement = estimator.estimate("alice", "m2", as_of=t1)
    assert before_settlement.weighted_score == 0.0
    assert before_settlement.weighted_history_notional == 0.0

    after_settlement = estimator.estimate("alice", "m2", as_of=t3)
    assert after_settlement.weighted_history_notional > 0.0
