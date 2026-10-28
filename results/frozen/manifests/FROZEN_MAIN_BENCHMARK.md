# Frozen Main Benchmark

Freeze timestamp: `2026-09-08T14:36:47+02:00`

## Benchmark contract

The main thesis benchmark is frozen to:

- q = {0.20, 0.50, 0.80, 0.95}
- model-training seeds = {0, 1, 2}
- 5-client benchmark: 120 logical runs
- 20-client benchmark: 120 logical runs
- total main benchmark: 240 logical runs

Methods:

- `fully_local`
- `centralized`
- `fedavg`
- `fedprox`
- `fedala`
- `apple`
- `gcflplus`
- `support_simplex`
- `support_simplex_postala`
- `learned_simplex_postala`

## Evaluation contract

Primary metric:

`macro_pr_auc`

Definition:

`trapezoidal_precision_recall_curve_auc`

Selection direction:

`max`

Every frozen logical run contains the required selected-test outputs for:

- VV: visible-validation-selected state evaluated on visible test labels
- VO: visible-validation-selected state evaluated on complete/oracle test labels
- OO: complete/oracle-validation-selected state evaluated on complete/oracle test labels

## Physical-output convention

Most methods use one result CSV per `(setup, method, q, seed)`.

Fully Local is intentionally different:

- 5-client: 5 client CSVs per logical run
- 20-client: 20 client CSVs per logical run

These client CSV groups collectively represent one logical Fully Local run.

## Integrity binding

Canonical artifact checksum manifest:

`canonical_artifacts_sha256.csv`

SHA-256:

`f9bc6acc276dd84db683dd457ae9eada50ab13e96894c75500f8b27f8805d3bd`

Frozen logical benchmark manifest:

`frozen_main_benchmark_manifest.csv`

SHA-256:

`696e26e7161b0694a94eccc628c1e06e20d8dac487ecbbf41b15c20e108b2eee`

## Status

This benchmark is frozen.

Performance tables and Figure 3 must be generated from this frozen
canonical suite, not from the archived Macro-AP analysis or deleted
pre-canonical result directories.
