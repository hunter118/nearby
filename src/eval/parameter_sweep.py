from __future__ import annotations

from itertools import product

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import BacktestConfig, EventDrivenBacktester
from eval.metrics import evaluate_closed_positions
from models import Market, TimelineEvent, TraderMarketSettlement


def run_parameter_sweep(
    markets: dict[str, Market],
    timeline: list[TimelineEvent],
    settlements: list[TraderMarketSettlement],
    base_config: BacktestConfig,
    delay_seconds_grid: list[int],
    skill_threshold_grid: list[float],
    consensus_threshold_grid: list[float],
    min_user_volume_grid: list[float],
    make_skill_estimator,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for delay_seconds, skill_threshold, consensus_threshold, min_user_volume in product(
        delay_seconds_grid,
        skill_threshold_grid,
        consensus_threshold_grid,
        min_user_volume_grid,
    ):
        skill_estimator: TraderSkillEstimator = make_skill_estimator(markets, settlements)
        cfg = BacktestConfig(
            delay_seconds=delay_seconds,
            skill_threshold=skill_threshold,
            consensus_threshold=consensus_threshold,
            min_skilled_traders=base_config.min_skilled_traders,
            max_single_trader_weight=base_config.max_single_trader_weight,
            min_edge=base_config.min_edge,
            min_user_volume=min_user_volume,
            max_trades_per_market=base_config.max_trades_per_market,
            stable_min_price=base_config.stable_min_price,
            lottery_min_price=base_config.lottery_min_price,
            lottery_max_price=base_config.lottery_max_price,
            stable_balance_fraction=base_config.stable_balance_fraction,
            lottery_lot_size=base_config.lottery_lot_size,
            lottery_max_exposure_fraction=base_config.lottery_max_exposure_fraction,
            min_days_to_resolution=base_config.min_days_to_resolution,
            max_days_to_resolution=base_config.max_days_to_resolution,
            trade_fee_bps=base_config.trade_fee_bps,
            slippage_bps=base_config.slippage_bps,
            min_entry_price=base_config.min_entry_price,
            max_entry_price=base_config.max_entry_price,
            dynamic_price_at_consensus=base_config.dynamic_price_at_consensus,
            dynamic_price_at_high_confidence=base_config.dynamic_price_at_high_confidence,
            dynamic_high_confidence=base_config.dynamic_high_confidence,
            max_market_fraction=base_config.max_market_fraction,
            max_balance_fraction=base_config.max_balance_fraction,
            max_loss_per_trade_fraction=base_config.max_loss_per_trade_fraction,
            min_ticket_size=base_config.min_ticket_size,
            initial_balance=base_config.initial_balance,
        )
        engine = EventDrivenBacktester(
            markets=markets,
            timeline=timeline,
            skill_estimator=skill_estimator,
            config=cfg,
        )
        result = engine.run()
        metric = evaluate_closed_positions(
            closed_positions=result["closed_positions"],
            initial_balance=cfg.initial_balance,
            final_balance=result["balance"],
        )
        rows.append(
            {
                "delay_seconds": float(delay_seconds),
                "skill_threshold": skill_threshold,
                "consensus_threshold": consensus_threshold,
                "min_user_volume": min_user_volume,
                "roi": metric["roi"],
                "total_pnl": metric["total_pnl"],
                "num_trades": metric["num_trades"],
            }
        )
    return rows
