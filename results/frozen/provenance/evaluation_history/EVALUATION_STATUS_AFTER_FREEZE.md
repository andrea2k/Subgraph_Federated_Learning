# Evaluation Status After the 2026-08-06 Freeze

This file records implementation progress after the evaluation contract was frozen.
It does not change the selector, protocols, metrics, seeds, q values, checkpoints,
or statistical rules in `FROZEN_EVALUATION_CONTRACT_WITH_RATIONALE.md`.

## Completed work

The following items from the frozen work order are complete:

1. protocol-aware pooled micro F1 and task-macro positive F1;
2. visible-support and training `pos_weight` audit;
3. assigned-versus-non-assigned analysis across q;
4. canonical five-client and twenty-client result-table pipeline;
5. five-client and twenty-client DR diagnostics;
6. clear twenty-client DR maps.

## Final method scope

`APPLE-ALA` was an intermediate development variant. The finalized thesis method is
`APPLE-PostALA`.

For the final thesis presentation:

- `APPLE-ALA` is excluded from canonical manifests, tables, plots, and discussion;
- its raw runs may be retained in an archive for provenance;
- `APPLE-PostALA` remains in the main comparison;
- this scope cleanup is not a change to the frozen evaluation rule and is not based
  on choosing a favorable metric after observing the rankings.

## Canonical method counts after the scope cleanup

Five-client benchmark:

- 8 methods;
- q = 0.2, 0.5, 0.8;
- seeds 0, 1, 2;
- expected coverage rows: 8 x 3 x 3 x 3 = 216.

Twenty-client benchmark:

- 9 methods;
- q = 0.2, 0.5, 0.8, 0.95;
- seeds 0, 1, 2;
- expected coverage rows: 9 x 4 x 3 x 3 = 324.

## Remaining scientific work

1. rerun the corrected analysis pipeline and verify 216/216 and 324/324 coverage;
2. complete the single-head versus multi-head ablation under the unchanged contract;
3. prepare final thesis figures and tables;
4. write the results, discussion, limitations, and methodology chapters.
