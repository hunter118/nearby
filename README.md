# Semantic Expert Following in Prediction Markets

This is the code-only research release for the September 2026 manuscript. The
strategy uses public wallet histories, question-text similarity, consensus
controls and cash-funded allocation. It does not train a predictive model on
trading outcomes and does not submit live orders.

## What this repository provides

- The event-driven portfolio engine, semantic expert scoring and observable-prefix
  market-activation rule.
- Exact, fully resolved configurations for the four main portfolios and the
  reported sensitivity and exploratory comparisons.
- A portable, checksum-verified prepared-input format and replay command.
- Unit tests and a synthetic end-to-end example that require neither an API key
  nor a model download.

This public repository does **not** include the historical transaction archive,
wallet-history arrays, embedding vectors, empirical result files, manuscript,
or anonymous review attachment. A successful synthetic test is not a
reproduction of the paper's historical returns. Complete empirical replay
requires the separate prepared inputs described in [REPRODUCING.md](REPRODUCING.md).
Current API responses cannot be assumed to recover an identical past archive.

## Strategy

1. Observe actual-token dollar turnover from the declared observation boundary.
   Historical wallet evidence starts after a market crosses one million USDC.
2. Activate a current market only after sufficient observed turnover, at least
   24 hours of observation, at least 100 prints in the preceding 24 hours,
   a quantity-weighted YES-equivalent price at or above 0.90 (or at or below 0.10),
   and a scheduled close less than 28 days away.
3. Weight settled-wallet performance by the fixed question-vector kernel
   `(max(cosine - 0.5, 0) / 0.5) ** 2`. Estimate skill from observed incremental
   settlement PnL, not total wallet wealth or raw winning percentages.
4. Aggregate qualifying wallet flow and require directional consensus,
   distinct-address breadth, limited single-wallet concentration and relevant
   historical support.
5. Size orders from available cash and the remaining deployment target, subject
   to general and text-classified competitive-event position caps. Apply the
   declared execution delay and adverse price adjustment; hold filled positions
   to the recorded settlement event.

The main configurations differ in allocation speed, position cap and price
ceiling. Their exact definitions are in `config/paper_v2/`; do not reconstruct
them from the older `config/default.yaml` demonstration settings.

## Interpretation

The historical experiment uses an archived, retrospectively assembled candidate
inventory and an explicit metadata-proxy label schedule. Chronological event
processing and past-prefix activation do not certify contemporaneous availability
of every archived title, deadline or resolution label.

Main fills are delayed **quantity-unconstrained historical-price simulations**.
They do not reconstruct an order book, spread, queue priority or market impact.
Thresholds were inspected on the same archive, and local sensitivity is not
independent out-of-sample validation. These distinctions apply to all headline
returns and drawdowns.

## Installation and tests

Use Python 3.10 or newer; the release is tested with Python 3.12. Frozen-vector
replay needs only the core numerical dependencies, not PyTorch or a transformer
checkpoint.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

For exact numerical comparisons, use `requirements-reference.txt` to match the
recorded research environment; the broad dependency bounds above also support
clean-environment software tests but do not promise bitwise identity across
all future library releases.

Optional question-embedding regeneration uses the `embeddings` extra. The paper
uses `BAAI/bge-large-en-v1.5`, revision
`d4aa6901d3a41ba39fb536a557fa166f842b0e09`; simply downloading the current default
revision is not a substitute for the fixed vectors.

```sh
python -m pip install -e ".[embeddings]"
```

## Reproduction

Follow [REPRODUCING.md](REPRODUCING.md) for exact commands, the synthetic smoke
test, configuration groups, input verification and empirical replay. Read
[DATA_FORMAT.md](DATA_FORMAT.md) before preparing or transferring inputs.

The portable entry points are:

```sh
PYTHONPATH=src python -m paper_v2_inputs --help
PYTHONPATH=src python -m paper_v2_replay --help
```

For a small synthetic end-to-end check:

```sh
PYTHONPATH=src python -m paper_v2_inputs demo --output demo_inputs
PYTHONPATH=src python -m paper_v2_replay \
  --inputs demo_inputs/inputs --spec demo_inputs/spec.json \
  --output demo_run --engine fast-checked
```

Use new output directories for each run. This example creates a synthetic fill
and settlement and compares both engines; it is not a historical paper replay.

They are separate from the original workspace-bound research drivers, whose
integrity checks intentionally still require their original artifacts. The
portable format has its own declared content and implementation bindings; it
does not silently disable the original checks.

## Earlier release

The earlier implementation and documentation remain available under the
[`iclr2027-v1`](https://github.com/hunter118/nearby/tree/iclr2027-v1) tag.
The older `run_frozen_replay.py`/`config/paper_experiments.json` workflow is for
that release, not the current main experiment. Its small input attachment must
not be mistaken for the expanded archive required here.

Only core code, fixed configurations, tests and usage documentation are
published. No credentials, personal research ledger, raw empirical data or
manuscript files belong in this code release.
