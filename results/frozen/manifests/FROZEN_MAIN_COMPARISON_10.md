# Final 10-Method Main Comparison Freeze

Freeze timestamp: `2026-09-08T22:25:29+02:00`

## Status

This is the authoritative thesis-facing main-comparison registry.

The prior 11-method freeze remains preserved as provenance:

`frozen_main_comparison_11_v2.csv`

Task-Weighted FedAvg is excluded from the final thesis-facing registry.

No experiment artifact was deleted or modified.

## Final main registry

1. `fully_local`
2. `centralized`
3. `fedavg`
4. `fedprox`
5. `gcflplus`
6. `fedala`
7. `apple`
8. `cwfedavg_task`
9. `support_simplex`
10. `support_simplex_postala`

## Rectangular contract

q = `{0.20, 0.50, 0.80, 0.95}`

model seeds = `{0,1,2}`

five-client = `120`

twenty-client = `120`

total = `240`

Expected rebuilt Step-2 protocol rows:

`240 x 3 = 720`

## Metric contract

selector = `macro_pr_auc`

direction = `max / maximize`

definition = `trapezoidal_precision_recall_curve_auc`

protocols = `VV, VO, OO`

VV and VO use the same visible-validation-selected state.

Figure 3 primary protocol remains `VO`.

## GCFL+

5C: `epsilon_1=0.25, epsilon_2=0.70`

20C: `epsilon_1=0.70, epsilon_2=0.70`

## Global-Centralized paper aggregation

1. pool q within each model seed;
2. compute mean and sample SD over model seeds 0,1,2;
3. repeat the same summary at every displayed q.

## Provenance

The old 11-method freeze, 792-row canonical extraction, and old Step-3
summaries remain provenance only.

Task-Weighted FedAvg raw results also remain on disk as historical
experiment data.

## Integrity

Source manifest SHA-256:

`0e30650a80865a59d597dfa339c554c2bde77caa07ccdda884e76e04606bf763`

Final 10-method manifest SHA-256:

`6171038450bb9749f8c90cfb6ec1eb8ca5da7518d6445838420c5c09c6a765d5`

Retained logical entries:

`240`

Retained physical result files:

`516`

Excluded Task-Weighted logical entries preserved:

`24`

Excluded Task-Weighted physical result files preserved:

`24`
