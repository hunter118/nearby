# External inputs and data boundaries

The code release does not contain the empirical transaction archive. The replay
uses prepared inputs rather than fetching live API data. The executable schema
validation is in `src/paper_v2_inputs.py`; use its exporter and synthetic demo to
create a correctly structured, hash-bound input directory.

The portable directory contains `manifest.json`, `vectors.npz` and one
`normalized/<market_id>/` directory per market containing `market.json` and
`data.npz`. Its manifest binds every consumed file, the observation bounds,
market order, archived-source identity and current implementation hashes. The
synthetic command `demo --output demo_inputs` creates that directory at
`demo_inputs/inputs` and its separate specification at `demo_inputs/spec.json`.

## Source information needed for an actual replay

1. A stable market inventory, including each market identifier, question text,
   binary token/outcome mapping, creation time, scheduled closing time and the
   explicitly chosen resolution label and label-availability convention.
2. Chronological transactions for each market: timestamp, wallet identifier,
   trade side, token quantity, actual traded-token price and YES-equivalent
   price. Wallet and market arrays must use stable, matching integer mappings.
3. A vector for every market in exactly the recorded inventory order. The paper
   used normalized 1,024-dimensional BGE-large-en-v1.5 vectors with model revision
   `d4aa6901d3a41ba39fb536a557fa166f842b0e09`. Frozen-vector replay needs neither
   the model nor its training libraries. Changing the question text, checkpoint
   or vector order changes the strategy input.
4. Prepared wallet-market history records after volume-only admission, with
   scores, notional, wallet indices, label-release times and source manifests.
   The public implementation `prepare_causal_expert_history.py` exposes the
   volume admission and historical score calculation used by the tests.

The `paper_v2_inputs export` command must run in the bound workspace in which
the original prepared history was created. It preserves the original
history/source checks, including the recorded source location; simply moving
the original source and history directories does not satisfy those bindings.
The example paths in `REPRODUCING.md` are placeholders for the original bound
locations, not permission to rewrite their manifests. Export is an archive
translator, not an alternative raw-data collection service. It emits a new
portable directory with relative paths. That complete directory can be copied
to another machine together with its manifest; do not edit or reorder its files.
If you already have such a portable directory, skip export and use it directly
with `paper_v2_replay`.

## Units and causal conventions

Times are UTC Unix seconds in prepared arrays. Prices are token prices in [0,1],
quantities are token units and notional is actual-token price times token
quantity. A NO print is mapped to `1 - price` only for the YES-equivalent price;
its volume contribution remains the actual traded-token notional. Summing both
sides of a single transaction is not intended.

Volume-only historical admission first aggregates the complete crossing second,
then begins eligibility one second later. That crossing second is excluded from
historical score observations. Resolved histories become available only after
the recorded label-release time; target-market and not-yet-available histories
are excluded from the wallet skill estimate. The market activation rule uses
completed UTC hours and already observed trade prefixes. None of these rules
makes retrospectively downloaded metadata or candidate selection point-in-time.

The selected `all_after_admission` history is not a final-500/final-1,000-print
tail. Other prepared history variants preserved in the implementation are not
used by the released main configurations.

## Public collection versus exact reproduction

Polymarket's public Gamma metadata and Data API transaction endpoints provide
market information and public trade records. API pagination, retention,
corrections, market text and metadata can change. The paper used an archived
41,891-market candidate inventory and 183,742,192 valid transaction records
over 2023-09-18 through 2026-08-08. The inventory was not a full point-in-time
listing of all markets. Current API access alone cannot be asserted to recreate
the same inventory, text, vectors, records or historical labels.

For a new collection, preserve raw responses, request parameters, pagination
logs, observation windows, token mappings and content hashes before preparing
arrays. Report any difference from the paper's archived sample. Do not use a
newly downloaded sample to claim exact reproduction of the published numbers.
Conversely, the saved reference accounts in the anonymous supplement must not
be substituted for market transactions or treated as an independent replay.

No private keys, authenticated exchange credentials or personal access tokens
are needed for the offline replay. Wallet identifiers are pseudonymous on-chain
addresses, not verified identities or independent human experts.
