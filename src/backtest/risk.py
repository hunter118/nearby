from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_order_notional(
    balance: float,
    market_volume: float,
    max_market_fraction: float,
    max_balance_fraction: float,
    max_loss_per_trade_fraction: float,
    min_ticket_size: float,
) -> float:
    cap_market = market_volume * max_market_fraction
    cap_balance = balance * max_balance_fraction
    cap_loss = balance * max_loss_per_trade_fraction
    notional = min(cap_market, cap_balance, cap_loss)
    if notional < min_ticket_size:
        return 0.0
    return notional
