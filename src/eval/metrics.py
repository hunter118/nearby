from __future__ import annotations

import math

from models import ClosedPosition


def evaluate_closed_positions(
    closed_positions: list[ClosedPosition],
    initial_balance: float,
    final_balance: float,
) -> dict[str, float]:
    trades = len(closed_positions)
    total_pnl = sum(x.pnl for x in closed_positions)
    wins = sum(1 for x in closed_positions if x.pnl > 0)
    avg_pnl = total_pnl / trades if trades else 0.0
    win_rate = wins / trades if trades else 0.0
    roi = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0
    pnl_values = [x.pnl for x in closed_positions]
    volatility = 0.0
    if len(pnl_values) > 1:
        mean = sum(pnl_values) / len(pnl_values)
        variance = sum((x - mean) ** 2 for x in pnl_values) / (len(pnl_values) - 1)
        volatility = math.sqrt(variance)

    return {
        "num_trades": float(trades),
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "win_rate": win_rate,
        "roi": roi,
        "pnl_volatility": volatility,
        "final_balance": final_balance,
    }
