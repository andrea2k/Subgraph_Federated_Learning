# Completed validation — 2026-10-27

Validation was completed before removing the migration harness. Temporary comparison tools were kept outside the project and then removed.

- **13 training paths**, over two rounds on masked synthetic graphs: exact agreement in reported metrics and selected model-state tensors before and after simplification.
- **600 label/mask tensor fingerprints** across both benchmarks, every q, and all splits: unchanged. All **18 original graph files** retain their SHA-256 hashes.
- **1,068 frozen experiment artifacts** and the **21-file numerical ledger** retain their original hashes.
- Thesis analysis rebuilt from empty output directories: **19 numerical reference tables** match. Figures 3, 4, and 6 match the original plot data; Figure 5's paired contrasts match the ledger. Thesis and appendix figures regenerate. Two historical presentation snapshots remain reference-only.
- **Eight CLI training cases** passed using the normal six-layer, hidden-dimension-64 architecture on temporary small graphs. They cover standard, independent-client, centralized, physical-region, hierarchical, specialist, and task-conditioned runs, including output paths and AA/AC/CC annotations.
- Twenty-client generation matches **15 original graphs / 90 tensor attributes**. Five-client materialization matches **3 graphs / 15 attributes**; seed reconstruction remains the limitation described in [reproduction](reproduction.md).
- All **15 method entrypoints** import, all **16 shell/job files** pass syntax checks, and editable installation and invocation outside the checkout pass. Frozen-output and incompatible-configuration protections were checked.

These checks used the existing local environment. They do not establish universal correctness and do not include full 80-round retraining, cluster/GPU execution, or installation into a fresh environment. The final project intentionally contains no migration test suite or automatic ledger comparison.
