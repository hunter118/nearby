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

---

## 10) 2026-08-08 Formal Research Snapshot

The formal study fixes the market cohort at
2026-04-27 06:45:40 UTC and evaluates subsequent events through
2026-08-08 06:42:30 UTC:

```bash
python src/run_research.py \
  --config config/research_2026_08_08.yaml
```

To update only the snapshot caches and manifest:

```bash
python src/run_research.py \
  --config config/research_2026_08_08.yaml \
  --fetch-only
```

The evidence is deliberately split into two samples.  The March 2023--August
2026 replay is used to study the signal and semantic risk structure, while only
the complete April--August 2026 trade tape is used for quantity-constrained
execution and capacity claims.  Signal ablations do not establish an independent
semantic alpha, so the paper's contribution is the expert-following and risk-control
framework together with an explicit public-print execution model.

### Exploratory semantic-risk study

The failure audit shows that the dominant loss was a window-boundary cold
start: one same-direction wallet was treated as unanimous consensus.  The
exploratory risk overlay therefore requires at least two same-direction
wallets, 1.25 effective wallets, no more than 75% directional weight from one
wallet, and at least 1.5 effective related markets in the signal's aggregate
expert history.  A transparent question classifier then caps competitive-event
exposure, while a 15% general position cap limits all other one-market losses.

Run the frozen offline long-window study with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python src/run_semantic_risk_study.py
```

The long development replay covers March 16, 2023--August 8, 2026.  Its source
cache is incomplete before the 2026 incremental segment, so it is used for
threshold development and stability checks rather than as a clean out-of-sample
estimate.  The selected 15% specification closes 312 positions, wins 310,
returns +294.32%, and limits maximum drawdown to 21.08%.  On the complete
April--August 2026 incremental segment, an unconstrained single-print fill
closes 48 positions, all profitable, and returns +8.21%, but its median order is
245% of the print used to price it.  The capacity-aware execution rule below
returns +4.30% with a 3.58% maximum drawdown and fills 87.1% of requested
notional.  Because the controls were chosen after inspecting losses, both
results remain exploratory; the next credible test is a frozen prospective
replay on post-snapshot data.

### Capacity-aware execution study

The execution runner can keep the signal and risk specification frozen while
changing only delay, token/side eligibility, participation, parent-order life,
price protection, and partial-fill behavior.  The primary recent rule waits
five minutes, participates in at most 25% of every subsequent same-token public
print, retains residual notional for 24 hours, permits only one parent order per
market, and reserves its cash until fill or expiry.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python src/run_semantic_risk_study.py \
  --start 2026-04-27T06:21:14 \
  --end 2026-08-08T06:42:30 \
  --base-preset tiered_position_cap_15pct \
  --preset-family execution \
  --presets execution_partial_target_token_25pct_24h
```

Use `--capacity-initial-balances 10000,25000,50000,100000` to replay the same
rule as a capacity curve.  Machine-readable results, monthly performance,
fixed-path cost stresses, and the paper figures are regenerated with:

```bash
python src/make_execution_study_artifacts.py
```

Code-only release contents:

- `src/alpha/`, `src/backtest/`, and `src/data/`: strategy, execution, and data
  normalization logic.
- `src/run_semantic_risk_study.py`: frozen offline long-window risk runner.
- `src/run_research.py`: point-in-time snapshot builder and research entry point.
- `src/alpha/risk_presets.py`: ordered threshold and robustness specifications.
- `src/make_semantic_risk_artifacts.py`: long-window tables and figures.
- `src/make_execution_study_artifacts.py`: recent execution, capacity, and cost
  artifacts.
- `src/make_paper_figures.py`: supporting deterministic paper figures.
- `src/check_paper_result_consistency.py`: checks reported TeX values against
  machine-readable artifacts.
- `config/` and `tests/`: frozen configuration and regression coverage.

The public repository intentionally excludes raw and normalized API data,
embeddings, local caches, generated CSV/JSON results, compiled papers, and paper
working files.  Artifact builders expect those local research inputs when used;
they are included so the transformation from experiment outputs to reported
tables and figures remains inspectable.
