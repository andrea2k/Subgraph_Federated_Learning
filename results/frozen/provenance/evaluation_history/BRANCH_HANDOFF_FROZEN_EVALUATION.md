# Master Thesis Branch Handoff: Frozen Evaluation Contract

## Purpose

This note transfers the scientific decisions made in the evaluation branch to the single-head-versus-multi-head branch. The receiving branch must inherit these choices and must not reopen selector, protocol, metric, masking, or aggregation definitions unless an actual implementation error is discovered.

## Frozen main training setup

- Four heterogeneity levels: `q = 0.2, 0.5, 0.8, 0.95`.
- Three seeds: `0, 1, 2`.
- Twenty virtual clients: four physical communities times five assigned task roles.
- Positive labels are allocated by q; all negative labels remain visible.
- The masks apply to train, validation, and test.
- Main runs use the existing architecture, hyperparameters, 80 communication rounds, one local epoch, and current automatic BCE `pos_weight`.
- Alternative class weighting, negative masking, architecture, or communication scope is a separately named ablation and cannot silently replace the main experiment.

## Frozen checkpoint selector

Use only:

> visible-validation macro PR-AUC, maximized

The other recorded selectors are appendix robustness diagnostics. The method ranking must never be chosen after seeing which selector favors the proposed method.

## Frozen protocols

- `VV` = `realistic_visible`: visible validation selects the checkpoint; visible test evaluates it. This is the primary deployment-oriented result.
- `VO` = `realistic_selection_oracle`: the same visible-selected checkpoint is evaluated on complete test labels. This is the main complete-label generalization result.
- `OO` = `oracle_full`: complete validation selects the checkpoint; complete test evaluates it. This is an oracle diagnostic, not the primary ranking.

Final rows must be identified using explicit metadata when available:

```text
selector_metric == macro_pr_auc
split == test
selection_protocol in {
    realistic_visible,
    realistic_selection_oracle,
    oracle_full
}
```

Do not rely only on exact phase strings, because method helpers have used different phase naming conventions.

## Frozen checkpoint-selection unit

- Fully Local: one best epoch per independent client.
- FedAvg/FedProx: one best federation round shared by all clients.
- GCFL+: one best system round containing its cluster state and models.
- FedALA/APPLE: one best system round containing all personalized states.
- OracleHard: one best round per independent five-client federation.
- Global Centralized: one best centralized epoch.

Do not give FedAvg client-specific early stopping, and do not force independent Fully Local models to share one epoch.

## Frozen metrics

### Primary threshold-free metric

Mean-client macro PR-AUC:

1. calculate AP separately for cycle2-cycle6 inside each client;
2. average the five tasks inside the client;
3. average clients inside the seed;
4. report mean plus or minus sample SD across the three seeds.

### Required pooled client-view metrics

- pooled micro positive F1;
- pooled task-macro positive F1;
- pooled per-task precision, recall, positive F1, TP, FP, TN, FN, and support.

The word `client-view` is required because the five role-clients in one physical community reuse the same underlying graph nodes. Negative events are therefore repeated across client views.

### Task-specific and mechanism metrics

- cycle2-cycle6 mean-client per-task PR-AUC;
- cycle2-cycle6 pooled positive F1, precision, recall, and support;
- assigned-task AP;
- mean non-assigned-task AP.

Assigned/non-assigned results explain specialization versus broad transfer. They are not a replacement for the complete leaderboard.

### Prohibited shortcut

Do not reconstruct or label an exact pooled PR-AUC from client-level AP scalars. Exact pooled AP requires raw probabilities and labels. Mean-client AP and pooled F1 answer different valid questions and should both be reported.

## Existing q=0.95 assigned/non-assigned finding

At the macro-PR-AUC-selected checkpoint:

- VV: Fully Local has the strongest assigned-task AP; all methods have very weak non-assigned visible AP.
- VO: FedAvg sacrifices a small amount of assigned-task specialization but has a large advantage on non-assigned complete labels.
- APPLE-PostALA transfers more broadly than Fully Local but less effectively than FedAvg at q=0.95.
- OO gives the same broad conclusion: FedAvg's advantage is mainly non-assigned-task transfer.

These are descriptive observations. They do not prove the causal mechanism.

## Required work before the single-head ablation is finalized

1. Run protocol-aware pooled metrics for all ten methods, all four q values, three seeds, and VV/VO/OO.
2. Audit exact visible support and the actual train-derived automatic `pos_weight`.
3. Extend assigned-versus-non-assigned AP to every q for Fully Local, FedAvg, and APPLE-PostALA.
4. Archive and freeze the multi-head outputs.
5. Run the single-head-versus-multi-head comparison under this exact contract.

## Single-head branch rule

The single-head branch is an architecture ablation. It must inherit:

- the same data and masks;
- the same q values and seeds;
- the same visible-validation macro PR-AUC selector;
- the same VV/VO/OO definitions;
- the same client-level and pooled reporting;
- the same class-weighting rule unless class weighting is explicitly studied as another ablation.

The clean architecture comparison is Fully Local and FedAvg first. APPLE is not automatically a pure single-head-versus-multi-head comparison because task-specific APPLE donor heads are part of the algorithm itself.

## Interpretation rules

- `FedAvg < Fully Local` under VV means harmful aggregation for locally observable utility; it does not automatically imply poor complete-label generalization.
- VO evaluates the same realistically selected checkpoint as VV on a different label population.
- A global-versus-hard comparison changes both backbone aggregation scope and task-head donor scope.
- Correlations between donor mass and performance are descriptive, not causal.
- Mixed seed signs or very small differences are reported as inconclusive rather than collapse.
