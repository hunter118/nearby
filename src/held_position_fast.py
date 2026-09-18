"""Portfolio-exact skip of dead signal updates after a position has opened.

This optimization is intentionally confined to ObservedFinalityBacktester:
positions are held until resolution and a resolved condition cannot reopen.
The inherited run loop still processes EVERY supplied print for pending fills,
expiry, marks, and equity. No timeline thinning or statistical rule is changed.

Signal accumulators and estimator cache contents for already-held markets are
not preserved. They are dead state only when execution_recheck_signal=False
and the estimator is one of the audited deterministic offline implementations.
Do not use this class with online learning, early exits/re-entry, flow-based
position management, or a mutated lifecycle/configuration.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from alpha.trader_skill import TraderSkillEstimator
from backtest.engine import _traded_token_price
from causal_rule_engine import ObservedFinalityBacktester
from run_similarity_kernel_study import KernelSkillEstimator


class HeldPositionFastBacktester(ObservedFinalityBacktester):
    """Drop expensive skill/flow work only after first fill in that condition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._require_no_signal_recheck()
        if self.config.signal_mode == 'expert_flow':
            estimator = self.skill_estimator
            if type(estimator) not in (TraderSkillEstimator, KernelSkillEstimator):
                raise TypeError('Held-position skip requires an audited deterministic offline estimator.')
            if any(name in estimator.__dict__ for name in
                   ('estimate', '_pairwise_similarity', 'market_similarity')):
                raise TypeError('Instance estimator method overrides are outside the audited contract.')
        elif self.config.signal_mode != 'favorite':
            raise ValueError('Unsupported signal mode for held-position skip.')
        self.held_position_signal_updates_skipped = 0

    def _require_no_signal_recheck(self):
        if self.config.execution_recheck_signal:
            raise ValueError('Held-position skip requires execution_recheck_signal=False.')

    def _process_due_orders(self, now, trade):
        # Guard before any pending child could read a deliberately stale flow.
        self._require_no_signal_recheck()
        return super()._process_due_orders(now, trade)

    def _on_trade(self, trade):
        self._require_no_signal_recheck()
        mid = trade.market_id
        if mid not in self.positions or mid in self.observed_resolved_markets:
            return super()._on_trade(trade)

        # Preserve the cheap metadata/volume/counter side effects exactly. No
        # other market uses this market's signal accumulator or observed volume.
        if self.planning_horizon_days is not None or self.ignore_risk_text:
            changes = {}
            if self.planning_horizon_days is not None:
                changes['close_time'] = trade.timestamp + timedelta(days=self.planning_horizon_days)
            if self.ignore_risk_text:
                changes.update(question='', category='unknown')
            self.markets[mid] = replace(self.markets[mid], **changes)

        self.processed += 1
        active_at = None if self.activation_times is None else self.activation_times.get(mid)
        if self.activation_times is None or (active_at is not None and trade.timestamp >= active_at):
            self.admitted += 1
            if trade.size > 0:
                self.market_observed_notional[mid] = (
                    self.market_observed_notional.get(mid, 0.0)
                    + trade.size * _traded_token_price(trade)
                )
            self.held_position_signal_updates_skipped += 1
        if self.processed % 2_000_000 == 0:
            print(f'Tape {self.processed:,}; admitted {self.admitted:,}; {trade.timestamp}', flush=True)
