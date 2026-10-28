# Final Frozen Main Comparison

Freeze timestamp: `2026-09-08T16:33:23+02:00`

## Status

This is the authoritative paper-facing main-comparison freeze.

The earlier 10-method freeze is retained as provenance and is not
overwritten:

- `frozen_main_benchmark_manifest.csv`
- `FROZEN_MAIN_BENCHMARK.md`

## Main-comparison registry

The final registry contains exactly 11 methods/references:

1. `fully_local`
2. `centralized`
3. `fedavg`
4. `fedprox`
5. `gcflplus`
6. `fedala`
7. `apple`
8. `task_weighted_fedavg`
9. `cwfedavg_task`
10. `support_simplex`
11. `support_simplex_postala`

`learned_simplex_postala` is not part of the main comparison. It is a
component ablation.

`support_only_postala` is also a component ablation.

Oracle physical-community, specialist-transfer, and hierarchical
experiments are collaboration diagnostics and are not ordinary main
competitors.

## Rectangular benchmark contract

q:

`{0.20, 0.50, 0.80, 0.95}`

Model-training seeds:

`{0, 1, 2}`

Entries per setup:

`11 x 4 x 3 = 132`

Total q-indexed main-comparison entries:

`132 + 132 = 264`

These are q-indexed benchmark entries, not raw CSV rows.

## Evaluation contract

Selector:

`macro_pr_auc`

Direction:

`max`

Metric definition:

`trapezoidal_precision_recall_curve_auc`

VV, VO, and OO final selected-test outputs are present.

VV and VO use the same visible-validation-selected model state.

Both valid phase-name conventions in the canonical runners are
accepted:

- `best_realistic_visible_test`
- `best_realistic_visible_macro_pr_auc_test`

and analogously for VO and OO.

## Figure 3 protocol

The primary protocol is locked before performance inspection:

`VO`

VO means visible-validation model selection followed by evaluation on
complete/oracle test labels.

## GCFL+ configurations

Five-client:

`epsilon_1 = 0.25, epsilon_2 = 0.70`

Twenty-client:

`epsilon_1 = 0.70, epsilon_2 = 0.70`

## Global-Centralized treatment

Global-Centralized does not use the q visibility restriction.

The canonical suite retains all four q-tagged copies per model seed for
provenance. Their complete task statistics are equivalent across q, but
the independently trained copies can finish at different selected
epochs and therefore need not have identical numerical PR-AUC values.

For paper-facing q plots and tables, q is therefore collapsed within
each model seed:

`c_s = mean_q(c_{q,s})`

The reported Centralized reference is then:

`mean_s(c_s) +/- sample_SD_s(c_s)`

using model-training seeds 0, 1, and 2.

That same mean +/- SD is repeated at all four q positions.

The frozen VO references are:

- Five-client: `94.175% +/- 0.270%`
- Twenty-client: `97.600% +/- 0.215%`

The exact machine-readable values are stored in:

`centralized_vo_pooled_reference.csv`

## Integrity

Canonical artifact checksum manifest:

`canonical_artifacts_sha256.csv`

SHA-256:

`f9bc6acc276dd84db683dd457ae9eada50ab13e96894c75500f8b27f8805d3bd`

Final 11-entry logical freeze:

`frozen_main_comparison_11.csv`

SHA-256:

`5e824a9cc7b691827e3c9d97d1e4d431855a9bfce4ad92880850b10b80ae1cc8`

Centralized pooled-VO reference:

`centralized_vo_pooled_reference.csv`

SHA-256:

`230a46d6d8ff0b6a5f4eb43838bb0da3d441f69148df749756e7f23866058756`

## Next step

Build the authoritative protocol-level master results table from this
freeze.

Expected paper-facing main-analysis shape:

`264 q-indexed logical entries x 3 protocols = 792 protocol-level rows`

before seed aggregation.

Figure 3 should not be drawn until those VO values have been inspected.
