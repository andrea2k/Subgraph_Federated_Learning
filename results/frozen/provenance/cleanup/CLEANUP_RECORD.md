# Legacy experiment cleanup record

Cleanup timestamp: `2026-09-07T11:03:09+02:00`

The following obsolete pre-canonical trees were deleted only after:

- creation of `andrea/final_prauc_suite`;
- reconciliation of all 348 final logical runs;
- verification of all 1068 canonical artifacts;
- 1068/1068 source-to-canonical SHA-256 matches;
- preservation of the 5-client checksum manifest;
- preservation of the 20-client recovery provenance;
- archival of the old analysis/manifests.

## Deleted trees

### `andrea/final_recovery_20c_20260826`

- Files: 356
- Size before deletion: 6.69 GiB
- Zero-byte historical files: 25
- Reason: Historical 20C recovery pool. Canonical artifacts were selected through frozen provenance; required recovery provenance was copied into final_prauc_suite/provenance.

### `andrea/final_runs_ap_legacy_20260824`

- Files: 666
- Size before deletion: 10.86 GiB
- Zero-byte historical files: 0
- Reason: Legacy Average-Precision-era experiment tree. Not part of the frozen final PR-AUC thesis suite.

### `andrea/final_incoming_20260817`

- Files: 236
- Size before deletion: 2.03 GiB
- Zero-byte historical files: 11
- Reason: Historical staging/repair/oracle material superseded by the scientifically audited final PR-AUC suite.

## Current authoritative result location

`andrea/final_prauc_suite/`

The deleted trees must not be used as thesis-final result sources.

A filename/size inventory of the deleted trees is retained in:

`deleted_legacy_trees_inventory.csv`
