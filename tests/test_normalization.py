from data.polymarket_client import PolymarketClient


def test_market_uses_actual_resolution_time_and_preserves_token_mapping():
    row = PolymarketClient.normalize_market(
        {
            "conditionId": "m1",
            "question": "Up or down?",
            "createdAt": "2026-08-01T00:00:00Z",
            "endDate": "2026-08-08T12:00:00Z",
            "closedTime": "2026-08-08T12:34:56Z",
            "closed": True,
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["1", "0"]',
            "clobTokenIds": '["asset-up", "asset-down"]',
        }
    )
    assert row["resolved_at"].isoformat() == "2026-08-08T12:34:56"
    assert row["close_time"].isoformat() == "2026-08-08T12:00:00"
    assert row["outcome_labels"] == ["Up", "Down"]
    assert row["clob_token_ids"] == ["asset-up", "asset-down"]


def test_trade_maps_outcome_index_999_from_asset_id():
    row = PolymarketClient.normalize_trade(
        {
            "proxyWallet": "wallet",
            "side": "BUY",
            "asset": "asset-down",
            "conditionId": "m1",
            "size": 5,
            "price": 0.2,
            "timestamp": 1786170174,
            "outcome": "Down",
            "outcomeIndex": 999,
            "transactionHash": "0xabc",
        },
        asset_outcome_index={"asset-up": 0, "asset-down": 1},
    )
    assert row["side"] == "BUY_NO"
    assert row["price_yes"] == 0.8
    assert row["trade_id"] == "0xabc"


def test_trade_with_unknown_outcome_is_not_forced_to_yes():
    row = PolymarketClient.normalize_trade(
        {
            "side": "BUY",
            "conditionId": "m1",
            "size": 5,
            "price": 0.2,
            "timestamp": 1786170174,
            "outcome": "Unknown",
            "outcomeIndex": 999,
        }
    )
    assert row["side"] is None
