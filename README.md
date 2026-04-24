## Nearby: Polymarket Backtesting Framework

Nearby is an event-driven backtesting framework for Polymarket strategies, with an emphasis on:

- public historical data ingestion (Gamma + Data API),
- strict anti-leakage simulation logic,
- settlement-based PnL accounting,
- scalable local caching for repeated experiments,
- equity curve export and plotting.

This repository is tailored for research on expert-following strategies (follow high-skill traders under configurable execution/risk constraints).

---

## 1) High-Level Architecture

### Data Layer
- Fetch market metadata from `gamma-api.polymarket.com`.
- Fetch historical trades from `data-api.polymarket.com`.
- Normalize everything into internal event objects (`trade` / `resolution`).

### Feature Layer
- Convert market question text into embeddings (`hashing` or `sentence_transformers`).
- Compute semantic similarity between current market and historical settled markets.

### Skill Layer
- For each trader and each settled market, compute a normalized settlement score.
- For a target market at time `t`, compute trader skill as a weighted average of historical scores:
  - weight = `similarity * historical_notional`.

### Signal + Execution Layer
- Build direction consensus from skilled trader flow.
- Trigger delayed execution.
- Enforce configurable price buckets, time-to-resolution constraints, and position sizing rules.

### Evaluation Layer
- Settlement metrics: `num_trades`, `win_rate`, `total_pnl`, `avg_pnl_per_trade`.
- Portfolio snapshot: `cash`, `open_notional`, `open_market_value`, `total_equity`.
- Full equity curve time series for plotting.

---

## 2) Anti-Leakage Guarantees

- Skill estimation only uses settled historical outcomes available at decision time.
- No future resolution labels are used for generating current signals.
- The event stream is processed chronologically.
- Orders are filled only after the configured delay.

---

## 3) PnL Interpretation

Two values matter:

- `final_balance`: cash on hand (can be low if much capital is still in open positions).
- `total_equity`: `cash + open_market_value` (preferred portfolio value indicator).

Use `total_equity` for strategy-level performance comparison.

---

## 4) Repository Structure

- `src/data/polymarket_client.py`: API clients + normalization helpers.
- `src/data/build_dataset.py`: market/trade/resolution dataset and timeline assembly.
- `src/features/embeddings.py`: embedding backends and similarity utilities.
- `src/alpha/trader_skill.py`: trader settlement scoring and skill estimation.
- `src/alpha/signal.py`: direction consensus signal logic.
- `src/backtest/engine.py`: event-driven execution and portfolio accounting.
- `src/eval/metrics.py`: summary metric computation.
- `src/run_backtest.py`: main backtest entrypoint.
- `src/plot_equity_curve.py`: export and plot equity curve.
- `config/default.yaml`: all experiment controls in one place.

---

## 5) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If using `sentence_transformers`, first run may download model weights.

---

## 6) Usage

### Run a Backtest

```bash
python src/run_backtest.py --config config/default.yaml
```

### Generate Equity Curve (CSV + PNG)

```bash
python src/plot_equity_curve.py \
  --config config/default.yaml \
  --csv-out artifacts/equity_curve.csv \
  --png-out artifacts/equity_curve.png
```

Outputs:
- `artifacts/equity_curve.csv`: timestamped equity snapshots.
- `artifacts/equity_curve.png`: equity chart.

---

## 7) Key Parameters (What Each One Means)

All parameters live in `config/default.yaml`.

### `data.*` (data scope and sampling)
- `markets_limit`: page size for Gamma market pagination.
- `open_market_pages`: number of pages fetched for open markets.
- `closed_market_pages`: number of pages fetched for closed markets.
- `trades_limit`: max number of trade rows fetched in total.
- `trade_fetch_mode`: trade fetch strategy (`by_markets` or `global`).
- `trade_market_count`: how many top-volume markets are sampled for trade fetch.
- `per_market_trades_limit`: per-market trade cap in `by_markets` mode.
- `lookback_days`: reserved lookback knob (useful for future filtering extensions).

### `cache.*` (performance)
- `enabled`: enable/disable local caching.
- `dir`: cache directory for markets, normalized trades, and embeddings.

### `embeddings.*` (text representation)
- `backend`: embedding backend (`hashing` or `sentence_transformers`).
- `hashing_n_features`: vector size for hashing backend.
- `st_model_name`: sentence-transformers model name (when backend is ST).
- `st_device`: runtime device (`auto`, `cpu`, `mps`, `cuda`).
- `st_batch_size`: embedding batch size.
- `st_normalize_embeddings`: whether to L2-normalize embeddings before similarity.

### `strategy.*` (signal logic)
- `min_user_volume`: minimum weighted historical notional required for a trader to be considered.
- `min_weighted_history`: minimum denominator (`sum(weights)`) for valid skill estimate.
- `skill_threshold`: minimum weighted skill to classify flow as skilled.
- `consensus_threshold`: minimum directional ratio to trigger a signal.
- `min_skilled_traders`: minimum number of skilled traders participating.
- `max_single_trader_weight`: concentration guardrail for one trader dominating signal weight.
- `min_edge`: minimum confidence-minus-price edge required.
- `positive_similarity_only`: clamp negative similarity values if true.
- `similarity_floor`: lower bound used in similarity clamping.
- `max_trades_per_market`: max entries allowed per market.

### `execution.*` (entry gating)
- `delay_seconds`: delay between signal and earliest allowed fill.
- `trade_fee_bps`: fee assumption in basis points.
- `slippage_bps`: slippage assumption in basis points.
- `min_entry_price`: global minimum allowed token entry price.
- `max_entry_price`: global maximum allowed token entry price.
- `stable_min_price`: lower bound for “stable” bucket.
- `lottery_min_price`: lower bound for “lottery” bucket.
- `lottery_max_price`: upper bound for “lottery” bucket.
- `dynamic_price_at_consensus`: dynamic cap at consensus threshold.
- `dynamic_price_at_high_confidence`: dynamic cap at high-confidence anchor.
- `dynamic_high_confidence`: confidence anchor for dynamic cap interpolation.

### `risk.*` (position sizing and holding constraints)
- `initial_balance`: starting cash.
- `max_market_fraction`: market-volume-based cap for order notional.
- `max_balance_fraction`: cash-based cap for order notional.
- `max_loss_per_trade_fraction`: per-trade maximum-loss budget as balance fraction.
- `min_ticket_size`: minimum notional to place any trade.
- `stable_balance_fraction`: notional size for stable bucket as fraction of current balance.
- `lottery_lot_size`: fixed quantity for lottery bucket trades.
- `lottery_max_exposure_fraction`: max portfolio exposure allocated to lottery bucket.
- `min_days_to_resolution`: minimum allowed days-to-resolution at entry time.
- `max_days_to_resolution`: maximum allowed days-to-resolution at entry time.

### `backtest.*` (time split placeholders)
- `start_ts`, `end_ts`: optional simulation window bounds.
- `train_end_ts`, `validation_end_ts`: optional split markers for train/validation/testing workflows.

---

## 8) Performance Notes

- First run at large scale can be slow (API pulls + embedding build).
- Re-runs with same configuration are much faster due to cache hits:
  - `markets_*.pkl`
  - `normalized_trades_*.pkl`
  - `embeddings/market_embeddings_*.npz`

---

## 9) Test

```bash
pytest
```
