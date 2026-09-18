# Reproducing the strategy

There are three distinct checks: source tests on synthetic inputs, validation of
saved empirical results, and an actual replay from market transactions. Passing
the first two does not establish that the third has been executed.

## 1. Installation and synthetic checks

From the package or repository root, install `requirements-core.txt` in a fresh
Python 3.10+ environment. Frozen-vector replay does not import or download the
embedding model. Run `PYTHONPATH=src python -m pytest -q` and the synthetic demo
shown in the README. Output directories must be new; completed experiments are
not overwritten.

For exact empirical comparisons, prefer Python 3.12 and
`python -m pip install -r requirements-reference.txt`. Those pins record the
observed numerical library versions in the research environment (Python
3.12.14). `requirements-core.txt` gives broader dependency bounds for software
checks; it is not a claim that every permitted library combination reproduces
identical floating-point output. Passing tests and a synthetic demo in a newer
environment is not equivalent to a full historical replay in that environment.

## 2. Exact configuration groups

The JSON files in `config/paper_v2/` include complete resolved parameters and
their original configuration hashes. They are not loose overrides of defaults.
The disabled expert-score standard-deviation cap is represented by `Infinity`,
the existing Python JSON convention accepted by the supplied loader; parsers
that require strict RFC JSON should not silently replace it with a finite cap.
The main paper uses:

| Label | Case identifier | Return | Daily maximum drawdown |
|---|---|---:|---:|
| A | all_p60_max98_cap25_kernel50sq | 200.76% | 12.72% |
| B | k50_cap25_fast | 257.25% | 13.64% |
| C | k50_cap35_mid | 244.18% | 11.95% |
| D | k50_cap35_fast_max97 | 318.32% | 19.02% |

All use initial equity of 10,000 USDC and the common requested observation
window 2023-09-18 00:00:00 through 2026-08-08 06:42:30 UTC. Return is final
total equity divided by initial equity minus one. Maximum drawdown is computed
from saved daily marks and includes open positions' marked value, not only
closed profit. It is not an intraday drawdown statistic.

The A/control, main, exploratory and matched groups have overlapping cases;
duplicates are not additional independent experiments.

## 3. Actual replay with external prepared inputs

Export an existing archived source and prepared causal-wallet history **from
the bound workspace in which that history was created**. Export preserves the
original history/source validation, including its recorded source location.
The `external/source` and `external/history` arguments below are placeholders
for those original bound paths; moving the original directories elsewhere does
not make them valid export inputs. Do not edit their bindings to bypass a failed
check. If you already have a portable prepared-input directory, skip the export
command and run replay directly; that complete portable directory can be moved
to another machine without changing its manifest.

```sh
PYTHONPATH=src python -m paper_v2_inputs export \
  --source external/source --history-root external/history \
  --output prepared_inputs --mode copy
PYTHONPATH=src python -m paper_v2_replay \
  --inputs prepared_inputs --spec config/paper_v2/main.json \
  --output replay_main --cases all_p60_max98_cap25_kernel50sq k50_cap25_fast k50_cap35_mid k50_cap35_fast_max97 \
  --engine fast-checked
```

Omit `--cases` to include A's cosine control. Substitute the matched or exploratory
specification and a different output directory to run those groups. The portable
manifest uses relative paths and validates hashes of its component files. The
`reference` engine is the conservative engine choice. `fast-checked` runs both
reference and held-position-optimized engines on the same inputs and parameters,
then requires byte-identical portfolio files and identical economic summaries.
It costs more work than a single replay. Consult the emitted validation record,
not just the final return.

The export command translates an existing prepared archive; it is not a crawler
and does not manufacture the historical market data from the saved results.
See `DATA_FORMAT.md` for the required inputs and their limitations.

## 4. Saved-reference validation (anonymous supplement only)

The anonymous supplement additionally includes `reference_results/` and
`check_reference_results.py`. Run:

```sh
python verify_archive.py
python check_reference_results.py
```

The first command checks the allowlisted archive's checksums. The second
recomputes return, daily drawdown, cash movements, fill/position arithmetic and
the final marked-equity identities from saved records for every case. It never
claims to recover the original signals, replay the original tape or validate
order-book capacity. Raw transaction prices and expert scores require section 3.

This compact numerical evidence is not included in the public core-code export.

## 5. Reproducibility boundaries

The paper's candidate inventory contains 41,891 markets and its prepared tape
183,742,192 valid transaction records. It includes archived market metadata,
vectors and label-availability conventions that current API responses can
change or omit. The raw/prepared empirical archive is not bundled with this
code-only release. The portable full prepared archive requires several gigabytes
(approximately 4.5 GiB for the current sample), independently of the compact
code and saved-reference accounts. Therefore the package is not a self-contained reproduction
of the empirical sample, and public API access alone is not a guarantee of
bit-for-bit reconstruction. Exact replay needs the matching archived inputs.

The executable strategy starts activity only after recorded cumulative
transaction notional crosses its rule, but the collection inventory itself was
retrospectively assembled. History labels use the declared metadata-proxy
convention. Historical title changes and actual order-book capacity are not
reconstructed. Fees and slippage in the sensitivity cases are stated simulation
assumptions, not an estimate of actual exchange costs.

The implementation is an offline simulator. No command in this reproduction
workflow signs or submits exchange orders.
