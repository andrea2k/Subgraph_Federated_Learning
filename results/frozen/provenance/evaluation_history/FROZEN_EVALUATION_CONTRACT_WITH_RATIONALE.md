# Frozen Evaluation Contract with Decision Rationale

**Project:** 20-client federated graph learning thesis  
**Status:** FROZEN for the main multi-head experiment  
**Freeze date:** 2026-08-06  
**Purpose:** Fix the scientific design before final analysis, prevent post-hoc metric or protocol changes, and preserve the reasoning needed for the methodology and discussion chapters.

---

## 1. Main scientific question

The main experiment studies how local, federated, clustered, and personalized aggregation methods behave when rare positive task labels are distributed heterogeneously across clients while negative labels remain commonly observable.

The benchmark contains 20 virtual clients:

- four physical graph communities;
- five task-role clients per physical community;
- one role for each task: cycle2, cycle3, cycle4, cycle5, and cycle6.

The heterogeneity parameter \(q\) controls how the positive labels of each task are distributed among the five task-role clients belonging to the same physical graph community.

### Why this question is useful

The design represents a rare-event setting such as fraud-pattern detection. Ordinary negative events are broadly available, while positive examples of a specific pattern may be concentrated at particular institutions or specialists. The experiment therefore studies whether communication helps clients recover knowledge that is locally missing, and whether unrestricted communication eventually causes interference under extreme specialization.

---

## 2. Frozen training setup

The following components are fixed for the main experiment and cannot be changed in response to the observed results:

- \(q\in\{0.20,0.50,0.80,0.95\}\);
- seeds 0, 1, and 2;
- 20-client construction: four physical communities × five task roles;
- positive-only \(q\)-allocation;
- all negative labels remain visible;
- the same masking rule is applied to train, validation, and test splits;
- the original generated graph data, client registry, selected subsets, physical communities, and task-role assignments;
- 80 communication rounds or local epochs, as already executed;
- one local epoch per communication round, as already executed;
- the existing model architecture and hyperparameters;
- the existing automatic positive-class weighting for the main runs;
- the existing method implementations and names;
- the F1 decision threshold of 0.5.

### Why these \(q\) values are frozen

The four values form an interpretable heterogeneity curve:

- \(q=.20\): positive labels are distributed equally across the five role-clients;
- \(q=.50\): moderate specialization;
- \(q=.80\): strong specialization;
- \(q=.95\): extreme specialization and stress testing.

Keeping all four values prevents the thesis from focusing only on the setting where a preferred method performs best. Together, they show whether an algorithm behaves consistently or only works in a particular heterogeneity regime.

### Why only positive labels are allocated while negatives remain visible

This is the main benchmark’s modeling assumption. It represents rare-event detection where normal events are common, but positive fraud patterns are distributed unevenly across clients.

This choice intentionally creates two linked difficulties:

1. task specialization, because one client receives most positives for its assigned task;
2. increasing local class imbalance for non-assigned tasks.

These effects are part of the benchmark definition and must not be silently removed after seeing the results. A design that also partitions negatives would answer a different scientific question and must therefore be presented as a separate sensitivity analysis.

### Why the same masks are applied to train, validation, and test

The goal of VV is to imitate a client that has incomplete label visibility throughout the complete machine-learning pipeline. Using full validation or test labels in the primary protocol would give the client information it would not realistically possess.

The original complete labels remain stored only so that VO and OO can be calculated as diagnostic evaluations. They are not used for ordinary visible training.

### Why rounds, local epochs, architecture, and hyperparameters are frozen

These settings were fixed before the final comparison and were used by all completed methods. Changing them after observing the results would introduce post-hoc tuning and could selectively benefit one method. Any later change must be separately named and cannot replace the main experiment.

### Why automatic positive-class weighting remains part of the main experiment

All principal methods were trained using the same existing class-weighting rule. Removing or capping it only after seeing the \(q=.95\) behavior would change the training objective.

However, the exact visible positive support and resulting positive-class weights must be audited. A later unweighted or capped-weight experiment may be reported as a sensitivity analysis, but the current runs remain the main benchmark.

---

## 3. Frozen checkpoint selector

The final thesis checkpoint selector is:

> **Visible-validation macro PR-AUC, maximized.**

The runners saved several selectors, but only `macro_pr_auc` has primary status. Loss, micro-F1, macro positive F1, and micro PR-AUC remain robustness checks for the appendix.

### Why validation must be visible

Visible validation reflects the information that is realistically available to a client. Selecting the main checkpoint using oracle validation would create an unrealistic advantage and would answer a different question.

### Why PR-AUC is used instead of accuracy

The positive class is rare. A model can obtain high accuracy by predicting almost everything as negative while completely failing to identify positive events. PR-AUC evaluates whether positive examples are ranked above negative examples and is therefore more informative for rare-event detection.

### Why PR-AUC is used instead of F1 for checkpoint selection

F1 depends on the fixed probability threshold of 0.5. A model’s ranking can improve while its thresholded F1 remains unchanged or becomes worse because of calibration. PR-AUC is threshold-free and is therefore a more stable choice for checkpoint selection.

### Why macro PR-AUC is used instead of micro PR-AUC

Macro PR-AUC first evaluates every task separately and then gives cycle2 through cycle6 equal importance. This prevents a task with more evaluated entries from dominating the selector.

Micro PR-AUC pools valid entries and can be dominated by tasks with greater support or by differences in score calibration between heads. It remains useful as a secondary metric, but it is not the final selector.

### Why the selector is frozen now

The selector must not be changed depending on which metric gives the most favorable ranking. Freezing one selector before the final tables protects the thesis from post-hoc metric selection.

---

## 4. Frozen checkpoint-selection unit

The selected checkpoint must correspond to the natural deployable object produced by each algorithm.

| Method | Frozen checkpoint-selection unit |
|---|---|
| Fully Local | One independently selected epoch per client |
| FedAvg and FedProx | One selected federation round shared by all 20 clients |
| GCFL+ | One selected system round containing the learned partition and cluster models |
| FedALA and global APPLE variants | One selected system round containing all personalized client states |
| OracleHard APPLE-PostALA | One selected round per independent five-client hard federation |
| Global-Centralized | One selected centralized epoch |

### Why Fully Local selects a different epoch for each client

Fully Local consists of 20 genuinely independent training processes. Each client has its own model trajectory and local validation data. Allowing independent early stopping reflects the actual method.

Forcing every local client to use one common epoch would artificially couple independent models and would not represent Fully Local correctly.

### Why FedAvg selects one common round

FedAvg has one shared global model at every communication round. Selecting a different FedAvg round for every client would add client-specific personalized early stopping that is not part of the algorithm.

The selected FedAvg checkpoint is therefore the round with the best aggregated visible-validation macro PR-AUC across clients.

### Why personalized methods still select one system round

APPLE and FedALA can have different personalized client models within the same round. Nevertheless, one round represents one complete system state containing all personalized models and personalization parameters. The round number is shared, although the client models inside that round are different.

### Why OracleHard can select one round per hard federation

The four hard clusters are trained as four independent five-client federations. Each is therefore allowed to select its own best round, in the same way that independent Fully Local clients can select their own best epochs.

---

## 5. Frozen evaluation protocols

### 5.1 Primary protocol: VV

**Visible validation → visible test**

Interpretation:

> How well does the realistically selected model perform on the labels that the client can actually observe?

VV is the primary deployment-oriented result because both selection and evaluation use only available labels.

### 5.2 Secondary protocol: VO

**Visible validation → oracle/full test**

Interpretation:

> How well does the realistically selected model perform on the complete underlying task when all test labels are revealed only for analysis?

VO is the main complete-label generalization analysis because the checkpoint is still selected realistically.

### 5.3 Diagnostic protocol: OO

**Oracle/full validation → oracle/full test**

Interpretation:

> How well could the method perform if complete validation information were available for model selection?

OO is an idealized diagnostic and must not replace VV as the primary result.

### Why VV and VO are both necessary

VV and VO use the same visible-selected checkpoint. Only the evaluated test labels differ.

Their comparison reveals whether a model that appears weak on the client’s visible labels nevertheless learned useful knowledge about hidden tasks. This is especially important at \(q=.95\), where visible off-task positive support may contain only one or two examples.

### Why OO is retained but demoted

OO helps separate checkpoint-selection limitations from model-learning limitations. However, it uses information unavailable to real clients, so it cannot be the main deployment claim.

---

## 6. Frozen metric hierarchy

No single metric answers every scientific question. The thesis therefore uses a fixed hierarchy.

### 6.1 Primary threshold-free metric: mean-client macro PR-AUC

For every client:

1. calculate PR-AUC separately for cycle2, cycle3, cycle4, cycle5, and cycle6;
2. average the five task values to obtain that client’s macro PR-AUC;
3. aggregate client values using the established client-level system aggregation;
4. obtain one value per seed;
5. report the mean and sample standard deviation across seeds.

### Why this is the primary metric

It has four advantages:

1. it is threshold-free;
2. it is suitable for rare positive classes;
3. it gives every task equal importance;
4. it evaluates rankings within each client model instead of comparing raw probability scales across independently trained personalized models.

The fourth point matters for Fully Local, FedALA, APPLE, and clustered methods, where independently trained models may have different probability calibration.

### 6.2 Required pooled system metrics

The thesis must additionally report a pooled client-view confusion-matrix evaluation.

For every method, \(q\), seed, protocol, and selected checkpoint, pool TP, FP, TN, and FN across the 20 client views.

Report:

- pooled micro positive F1;
- pooled task-macro positive F1;
- pooled per-task positive F1;
- pooled per-task precision;
- pooled per-task recall;
- pooled confusion counts.

### Why pooled F1 is required

Mean-client metrics answer how well the average client performs. Pooled F1 answers how well the complete collection of client predictions performs at the event level after forming a large confusion matrix.

The pooled analysis is therefore an important fairness and robustness view for comparing Fully Local, shared, clustered, and personalized systems.

### Why pooled F1 does not replace mean-client macro PR-AUC

Pooled F1 is threshold-dependent and gives more influence to clients or tasks with more evaluated entries. Mean-client macro PR-AUC instead gives controlled importance to clients and tasks and avoids direct cross-model score comparison.

Reporting both prevents the conclusion from depending on only one aggregation rule.

### Why the pooled population is called “pooled client-view events”

The five virtual role-clients from one physical community reuse the same underlying graph nodes. In VV, each positive task label is visible to one role-client, while negatives are visible to all five role-clients.

The pooled confusion matrix therefore contains client-specific views of events, not a set of unique physical graph events. The terminology must make this explicit.

### 6.3 Required cycle-specific results

For cycle2, cycle3, cycle4, cycle5, and cycle6, report:

1. mean-client per-task PR-AUC;
2. pooled per-task positive F1;
3. pooled per-task precision;
4. pooled per-task recall;
5. positive support under the protocol.

### Why cycle-specific results are necessary

The all-task macro score can hide whether an algorithm succeeds only on one task or performs consistently across all five tasks. Per-task reporting reveals whether improvements are broad or caused by one easy or well-supported cycle type.

The tasks do not receive separate checkpoint selectors. All task-specific values are measured at the same macro-PR-AUC-selected checkpoint.

### 6.4 Assigned-task versus non-assigned-task diagnostic

Retain:

- assigned-task PR-AUC;
- mean non-assigned-task PR-AUC over the other four tasks.

### Why this diagnostic is necessary

It separates two behaviors:

- local specialization on the task for which a client receives most positives;
- transfer to tasks for which that client receives little positive supervision.

It explains where a method’s all-task result comes from, but it does not replace the all-task leaderboard.

### 6.5 Why exact pooled PR-AUC is not reconstructed from current scalar outputs

PR-AUC depends on the ranking of every individual prediction score. It cannot be reconstructed exactly from confusion counts or from client-level PR-AUC values.

Moreover, concatenating raw probabilities from independently trained models compares their calibration directly. A model can rank its own positives correctly but use a lower probability scale than another client model. For this reason, pooled PR-AUC is not treated as an automatically fairer universal primary metric.

If raw predictions are stored in a future run, pooled PR-AUC may be reported as an additional sensitivity result, not as a replacement for the frozen primary metric.

### 6.6 Metrics that do not have primary status

- Accuracy, because rare negatives dominate it.
- F1-selected checkpoints, because F1 depends on threshold 0.5.
- Micro PR-AUC as selector, because it pools entries and can overweight high-support tasks.
- Loss as selector, because its value is affected by class weighting and is harder to compare across clients and \(q\).
- Other saved selectors, which remain appendix robustness checks.

---

## 7. Frozen statistical reporting

For every metric:

1. calculate the result separately for seed 0;
2. calculate the result separately for seed 1;
3. calculate the result separately for seed 2;
4. report mean ± sample standard deviation across the three seeds;
5. use paired seed differences for direct method comparisons when possible.

Do not pool the three seeds before computing a metric.

### Why statistics are calculated per seed first

The random seed defines a complete replicated experiment. Pooling all predictions from all seeds into one result would treat correlated predictions as independent observations and would hide between-run variability.

### Why paired seed differences are preferred for method comparisons

Methods evaluated under the same seed share the same generated data and benchmark realization. Subtracting their results within each seed removes part of the variation caused by the benchmark instance and gives a clearer comparison.

### Why mixed seed signs must be described cautiously

With only three seeds, a mean difference can be driven by one run. If one method wins some seeds and loses others, the result must be described as inconsistent or inconclusive rather than as a reliable improvement.

---

## 8. Centralized upper bound

Global-Centralized remains a separate upper-bound result because it trains one model on an unsplit graph with complete labels.

It must be shown in a separate column, panel, or clearly marked reference.

### Why it is not an equal-population comparator

The decentralized methods operate on 20 masked client views and communicate under federated constraints. The centralized model receives complete structural and label information in one place. Its purpose is to show whether the underlying prediction problem is learnable when the federated information barriers are removed.

It must therefore not be presented as if it were another method under identical information access.

---

## 9. Frozen interpretation rules

The following wording rules are fixed to prevent overclaiming.

### Negative transfer

FedAvg below Fully Local under VV may be described as harmful aggregation for visible client utility.

It must not automatically be described as universal negative transfer if VO or OO still favor FedAvg.

### Cross-community borrowing

A relationship between donor mass and performance may be described as consistent with or supportive of a mechanism.

It must not be called causal unless a controlled ablation changes only that component.

### Global versus OracleHard

The comparison changes both:

- backbone aggregation scope;
- task-head donor scope.

Therefore, it supports conclusions about the complete global versus complete hard-cluster systems. It does not isolate the causal effect of head restriction alone.

### Small or inconsistent differences

Small mean differences, large variation, or mixed seed signs must be described as comparable, inconclusive, or not consistently different.

---

## 10. What is frozen and cannot be changed

The following decisions cannot be changed in response to observed method rankings:

- the four \(q\) values;
- positive-only allocation and universally visible negatives;
- masking train, validation, and test;
- client construction and community/task-role assignments;
- seeds, rounds, local epochs, architecture, and main hyperparameters;
- main-run automatic positive-class weighting;
- visible-validation macro PR-AUC as the final selector;
- VV as primary, VO as secondary, and OO as diagnostic;
- the F1 threshold of 0.5;
- mean-client macro PR-AUC as the primary threshold-free metric;
- pooled micro F1 and pooled task-macro positive F1 as required complements;
- cycle-specific reporting definitions;
- assigned/non-assigned diagnostic definitions;
- method-specific checkpoint-selection units;
- mean ± sample SD over three seeds;
- paired seed comparisons where possible.

---

## 11. Allowed changes after the freeze

Only two categories are allowed.

### 11.1 Genuine implementation correction

Examples:

- wrong protocol rows selected;
- duplicate rows;
- missing clients or tasks;
- incorrect task mapping;
- wrong checkpoint metadata;
- a confirmed code defect.

The correction must document:

1. the error;
2. how it was detected;
3. the corrected logic;
4. which outputs were affected;
5. whether retraining or only re-analysis was required.

### 11.2 Separately named ablation or sensitivity experiment

Examples:

- unweighted or capped positive-class weighting;
- single-head architecture;
- alternative negative masking;
- global backbone with hard heads;
- hard backbone with global heads.

Such experiments must inherit the frozen evaluation protocol unless the purpose of the ablation is explicitly to study evaluation itself. They cannot silently replace the principal experiment.

---

## 12. Remaining work under the frozen contract

The work order is fixed:

1. reconstruct pooled F1 and pooled per-task confusion metrics from existing selected-checkpoint CSVs;
2. audit visible positive support and automatic `pos_weight` for every \(q\), client, task, and split;
3. extend assigned-versus-non-assigned diagnostics across \(q\) using the frozen selector;
4. archive and freeze the final multi-head result tables;
5. complete the single-head versus multi-head ablation using the identical evaluation contract;
6. generate thesis-ready tables and figures;
7. write the results and discussion chapters without changing the contract.

### Why pooled analysis comes before the single-head ablation

The architecture ablation must inherit a stable evaluation definition. Completing the single-head branch before freezing the evaluation would risk comparing architectures under changing selectors or aggregation rules.

### Why the support and `pos_weight` audit comes before further mechanism claims

At high \(q\), non-assigned tasks may have extremely small visible positive support and very large automatic positive-class weights. These quantities must be measured before attributing poor performance only to federation, personalization, or client drift.

---

## 13. Thesis-ready rationale summary

The following paragraph may be adapted for the methodology chapter:

> The evaluation protocol was fixed before producing the final result tables to avoid post-hoc selection of checkpoints or metrics. Checkpoints were selected by visible-validation macro PR-AUC because the benchmark concerns rare positive events, PR-AUC is threshold-free, and macro averaging gives equal importance to all five graph tasks. Visible-visible evaluation was treated as the primary deployment-oriented protocol, while visible-oracle evaluation measured complete-label generalization of the same realistically selected checkpoint. Oracle-oracle evaluation was retained only as an idealized model-selection diagnostic. Mean-client macro PR-AUC was used as the primary threshold-free metric, and pooled micro and task-macro positive F1 were reported as complementary event-level evaluations. This combination captures both average client utility and the behavior of the pooled client-view confusion matrix without relying on a single aggregation rule.

The following paragraph may be adapted for the data-setting description:

> Heterogeneity was introduced by allocating positive labels according to the specialization parameter \(q\), while negative labels remained visible to all clients. This models a rare-event setting in which normal behavior is widely observed but positive fraud patterns are concentrated at particular clients. The design intentionally combines task specialization with increasingly severe local class imbalance. Alternative negative-label allocation would represent a different missing-label process and was therefore not used to replace the principal benchmark.

The following paragraph may be adapted for the limitations section:

> The pooled evaluation operates on client-view events rather than unique physical node-label events because the five task-role clients within a physical community reuse the same graph. In addition, the global-versus-hard comparison jointly changes backbone aggregation and task-head donor scope, so it identifies the effect of the complete communication design rather than isolating either component. Finally, only three seeds were available; consequently, small differences and mixed paired-seed signs were interpreted cautiously.

---

## 14. Sign-off rule

After this document is committed and checksummed:

- no principal result may be regenerated using a different selector or protocol;
- no method may receive a different evaluation definition because of its observed performance;
- all final analysis scripts must state the frozen selector and protocols explicitly;
- any deviation must be documented as an implementation correction or a separately named ablation.
