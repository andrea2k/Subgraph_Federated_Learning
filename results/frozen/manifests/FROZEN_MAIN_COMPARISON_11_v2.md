# Final Main Comparison Freeze - Manifest Encoding Correction

## Status

The scientific freeze is unchanged.

The original manifest is preserved:

`frozen_main_comparison_11.csv`

A machine-readable corrected manifest was created:

`frozen_main_comparison_11_v2.csv`

## Reason for correction

The original `result_paths` field used the pipe character (`|`) as a
separator between multiple physical result paths.

Some legitimate five-client filenames themselves contain `|` between
client identifiers. Therefore blindly splitting `result_paths` on `|`
could incorrectly split one real filename into several fake paths.

The corrected manifest stores paths in `result_paths_json` as a JSON
array.

No experiment result, metric, method membership, q value, model seed,
protocol decision, or scientific freeze decision was changed.

## Frozen scientific contract

- 11 methods/references
- 132 five-client q-indexed entries
- 132 twenty-client q-indexed entries
- 264 total
- selector: `macro_pr_auc`
- metric definition: `trapezoidal_precision_recall_curve_auc`
- Figure 3 primary protocol: `VO`

## Integrity

Original manifest SHA-256:

`5e824a9cc7b691827e3c9d97d1e4d431855a9bfce4ad92880850b10b80ae1cc8`

Corrected v2 manifest SHA-256:

`0e30650a80865a59d597dfa339c554c2bde77caa07ccdda884e76e04606bf763`

Rows:

`264`

Resolved physical result paths:

`540`
