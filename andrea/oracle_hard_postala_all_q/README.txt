Oracle hard-cluster APPLE-PostALA experiment

This directory contains the local organized copy of the complete
3 q-values x 3 seeds x 4 clusters experiment.

runs/
    Raw result and DR CSV files organized by q, seed, and cluster.

experiment_logs/
    Original experiment-log CSV files. Their contents are preserved
    exactly as recorded during execution.

experiment_manifest.csv
    Clean local manifest containing both the original DAIC paths and
    the current organized local paths.

reorganization_map.tsv
    Mapping from every original local directory to its organized path.

local_registry.tsv
    Original submission registry plus the organized local output root.

metadata/
    Copies of the experiment design, submission registry, validation
    inventory, and original official-log list.

The DAIC source files were not modified.
