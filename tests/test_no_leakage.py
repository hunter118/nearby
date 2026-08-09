from datetime import datetime, timedelta, timezone

import pytest

from alpha.trader_skill import TraderSkillEstimator
from features.embeddings import SimilarityConfig
from alpha.trader_skill import (
    build_trader_market_settlements,
    build_trader_market_settlements_from_records,
)
from models import Direction, Market, Side, TradeEvent, TraderMarketSettlement


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
    assert after_settlement.supporting_markets == 1
    assert after_settlement.effective_history_markets == pytest.approx(1.0)
    assert after_settlement.positive_history_weight_fraction == pytest.approx(1.0)


def test_settlement_accounting_handles_both_binary_tokens():
    settled_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    traded_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        TradeEvent("1", "yes_market", "buyer", Side.BUY_YES, 0.7, 10.0, traded_at),
        TradeEvent("2", "yes_market", "seller", Side.SELL_NO, 0.7, 10.0, traded_at),
        TradeEvent("3", "no_market", "buyer", Side.BUY_NO, 0.3, 10.0, traded_at),
        TradeEvent("4", "no_market", "seller", Side.SELL_YES, 0.3, 10.0, traded_at),
    ]
    rows = build_trader_market_settlements(
        trades,
        {"yes_market": Direction.YES, "no_market": Direction.NO},
        {"yes_market": settled_at, "no_market": settled_at},
    )
    scores = {(row.market_id, row.trader_id): row.score for row in rows}

    # Buying the winning token earns 0.3 / 0.7; selling the losing token
    # earns the full sale proceeds.  Both are positive directional calls.
    assert scores[("yes_market", "buyer")] == pytest.approx(3.0 / 7.0)
    assert scores[("yes_market", "seller")] == pytest.approx(1.0)
    assert scores[("no_market", "buyer")] == pytest.approx(3.0 / 7.0)
    assert scores[("no_market", "seller")] == pytest.approx(1.0)


def test_record_settlement_builder_matches_event_builder():
    settled_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    traded_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trade = TradeEvent("1", "m1", "alice", Side.BUY_NO, 0.25, 8.0, traded_at)
    event_rows = build_trader_market_settlements(
        [trade], {"m1": Direction.NO}, {"m1": settled_at}
    )
    record_rows = build_trader_market_settlements_from_records(
        [
            {
                "market_id": "m1",
                "trader_id": "alice",
                "side": "BUY_NO",
                "price_yes": 0.25,
                "size": 8.0,
                "timestamp": traded_at,
            },
            # Unresolved markets are intentionally ignored before aggregation.
            {
                "market_id": "open_market",
                "trader_id": "alice",
                "side": "BUY_YES",
                "price_yes": 0.5,
                "size": 8.0,
                "timestamp": traded_at,
            },
        ],
        {"m1": Direction.NO},
        {"m1": settled_at},
    )
    assert record_rows == event_rows
