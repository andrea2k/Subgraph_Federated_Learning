FINAL THESIS NUMERIC LEDGER
===========================

Purpose
-------
This directory is a read-only snapshot of the frozen numerical evidence
used by the thesis figures, result paragraphs, collaboration diagnostics,
and communication analysis.

No experiment was rerun and no numerical quantity was recomputed when
this ledger was created. Every numbered CSV is a byte-for-byte copy of
an existing frozen output. MANIFEST.csv records the original source path
and SHA-256 checksum.

Core reporting conventions
--------------------------
- Model-training seeds: 0, 1, 2.
- Reported uncertainty: sample standard deviation across the three seeds.
- Primary metric: true trapezoidal Macro PR-AUC.
- Primary main-result protocol: VO = visible-validation-selected state,
  evaluated on complete test labels.
- VV and VO use the same selected state.
- OO uses complete-validation selection and complete-test evaluation.
- Figure 5 contrasts are calculated within matched seeds before their
  mean and sample standard deviation are reported.
- Figure 6 dominance is mean-based: communication volume and mean VO
  Macro PR-AUC determine the classification. SD is not used for
  dominance and no statistical significance claim follows from it.

Public method naming
--------------------
Historical result identifiers and historical display-name strings are
preserved inside the frozen source tables.

Final paper-facing mapping:
  Support-Simplex             -> FedSSR
  Support-Simplex + PostALA   -> FedSSR + PostALA

See 21_public_method_name_mapping.csv.

FedSSR denotes the complete proposed method.
Support-Simplex denotes the routing mechanism inside FedSSR.
PostALA is an optional head-only refinement.

Figure/result sources
---------------------
Figure 3:
  01_main_vo_macro_results.csv

Figure 4:
  02_protocol_analysis.csv

Figure 5:
  03_ablation_vo_summary.csv
  04_ablation_paired_contrasts.csv

Collaboration diagnostics / Table 2:
  05-14

Communication accounting:
  15-16

Figure 6:
  17_communication_performance_points.csv
  18_communication_dominance_audit.csv
  20_figure6_plotdata_v2.csv

Controlled/oracle diagnostic communication trade-off:
  19_communication_diagnostic_tradeoff_20c.csv

Provenance
----------
The files in this directory should be treated as a numeric reference
archive. Do not edit them to change paper terminology. Reader-facing
renames belong in plotting/manuscript display mappings.
