from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from alpha.trader_skill import TraderSkillEstimator
from data.replay_snapshot import load_snapshot, save_snapshot
from models import Direction, Market, Side, TradeEvent, TraderMarketSettlement
from run_frozen_replay import verify_metrics


def inputs():
    now = datetime(2026, 1, 1, 0, 0, 0, 123456)
    market = Market("a", "Question A?", "test", now, now + timedelta(days=10),
                    now + timedelta(days=10), Direction.YES, 1234.56789, False)
    markets = {"a": market, "b": replace(market, market_id="b", question="Question B?")}
    trades = [TradeEvent("id2", "b", "public_address_1", Side.SELL_NO, .923456789,
                         15.123456789, now),
              TradeEvent("id1", "a", "public_address_2", Side.BUY_YES, .934567891,
                         20.234567891, now)]
    history = [TraderMarketSettlement("public_address_1", "a", .01234567891,
                                      300.123456789, now - timedelta(days=1))]
    vectors = np.array([[.6, .8], [.8, .6]], dtype=np.float32)
    manifest = {"requested_start_utc": "2025-01-01T00:00:00Z",
                "effective_end_utc": "2026-02-01T00:00:00Z"}
    return now, markets, trades, history, vectors, manifest


def test_snapshot_roundtrip_lossless_and_tie_order(tmp_path):
    now, markets, trades, history, vectors, manifest = inputs()
    save_snapshot(tmp_path, markets, trades, history, vectors, manifest)
    m, t, h, v, meta = load_snapshot(tmp_path, now, now + timedelta(days=20))
    assert m == markets
    assert [x.market_id for x in t] == ["b", "a"]  # no re-sort by transaction ID
    assert [x.timestamp for x in t] == [now, now]
    for original, decoded in zip(trades, t):
        assert original == replace(decoded, trade_id=original.trade_id, trader_id=original.trader_id)
    assert h[0] == replace(history[0], trader_id=t[0].trader_id)
    assert t[0].trader_id != t[1].trader_id
    assert v.dtype == vectors.dtype
    assert np.array_equal(v, vectors)
    assert "public_address" not in (tmp_path / "snapshot.json").read_text()


def test_snapshot_rejects_corruption_and_out_of_bounds(tmp_path):
    now, markets, trades, history, vectors, manifest = inputs()
    save_snapshot(tmp_path, markets, trades, history, vectors, manifest)
    with pytest.raises(ValueError, match="outside"):
        load_snapshot(tmp_path, datetime(2024, 1, 1), now)
    path = tmp_path / "replay.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(tmp_path, now, now)


def test_precomputed_vectors_skip_text_model_and_preserve_scores():
    now, markets, trades, history, vectors, manifest = inputs()
    estimator = TraderSkillEstimator(markets, history, precomputed_market_vectors=vectors)
    assert estimator.embedder is None
    estimate = estimator.estimate("public_address_1", "b", now)
    assert estimate.weighted_score == pytest.approx(history[0].score)
    with pytest.raises(ValueError, match="aligned"):
        TraderSkillEstimator(markets, history, precomputed_market_vectors=vectors[:1])


def test_uniform_history_is_global_not_semantic_and_still_as_of():
    now, markets, _, _, vectors, _ = inputs()
    markets["c"] = replace(markets["a"], market_id="c")
    history = [TraderMarketSettlement("u", "a", .5, 100, now-timedelta(days=2)),
               TraderMarketSettlement("u", "b", -.5, 100, now-timedelta(days=1)),
               TraderMarketSettlement("u", "c", 1, 100, now-timedelta(days=1)),
               TraderMarketSettlement("u", "a", 1, 1000, now+timedelta(days=1))]
    # c is aligned with a, orthogonal to b. Own-target and future records excluded.
    estimator = TraderSkillEstimator(markets, history,
        precomputed_market_vectors=np.array([[1., 0.], [0., 1.], [1., 0.]]))
    uniform = TraderSkillEstimator(markets, history, similarity_mode="uniform")
    assert estimator.estimate("u", "c", now).weighted_score == pytest.approx(.5)
    assert uniform.estimate("u", "c", now).weighted_score == pytest.approx(0)
    assert uniform.estimate("u", "c", now).weighted_history_notional == 200
    assert uniform.estimate("u", "c", now).effective_history_markets == 2


def test_reference_verification_detects_changed_metrics():
    assert all(x["passed"] for x in verify_metrics({"return": 1.}, {"return": 1.}))
    assert not verify_metrics({"return": 1.}, {"return": 2.})[0]["passed"]
