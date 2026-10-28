# Canonical PR-AUC Suite Provenance

This directory contains provenance retained for the frozen final PR-AUC
experiment suite.

## Canonical experimental scope

- 5-client main benchmark: 120 logical runs
- 20-client main benchmark: 120 logical runs
- Additional diagnostic/baseline experiments: 84 logical runs
- cwFedAvg-Task experiments: 24 logical runs
- Total: 348 logical runs

The canonical experimental tree contains 1068 data artifacts.

The exact source-to-canonical mapping is stored in:

- ../manifests/canonical_prauc_migration_plan.csv

The canonical artifact SHA-256 manifest is stored in:

- ../manifests/canonical_artifacts_sha256.csv

## Retained 5-client provenance

`five_client/five_client_SHA256.txt` is the original checksum manifest
that accompanied the final 5-client result backup.

## Retained 20-client recovery provenance

`twenty_client/recovery_20260826/` preserves the recovery records used
to distinguish provenance-approved original runs from failed, partial,
superseded, repaired, and replaced runs.

The frozen 20-client logical partition is:

- 73 provenance-approved original-kept runs
- 35 repaired non-APPLE runs
- 12 APPLE replacement runs
- Total: 120 logical runs

The historical `original_partial` result tree itself is not canonical.
Only artifacts selected through the frozen migration plan are part of
the canonical suite.

## Submission registry availability

The following final submission-registry filenames were searched for
recursively in the local repository and were not available locally:

- final_prauc_20c_main_repair_20260826_jobs.tsv
- final_prauc_20c_apple_replacement_20260826_jobs.tsv
- final_prauc_additions_84_20260827_jobs.tsv
- final_prauc_additions_60_repair_20260829_jobs.tsv
- final_prauc_cwfedavg_24_20260830_jobs.tsv

Their absence from the local archive does not affect the canonical
artifact set. The final suite was independently reconciled at the
logical-run, artifact, provenance, and SHA-256 levels before cleanup.

No missing registry file has been reconstructed or fabricated.
