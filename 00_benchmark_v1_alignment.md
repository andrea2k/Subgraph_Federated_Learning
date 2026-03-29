# Benchmark v1 Alignment Note
Andrea Li
Date: 2026-03-25

## 1. Thesis objective

The current thesis objective is to build a controllable synthetic directed-multigraph benchmark/testbed for studying client heterogeneity in federated graph learning, how such heterogeneity should be measured, and how it relates to local and federated learnability.

This is intended as a benchmark / measurement / evaluation contribution built on top of existing graph-generation / GNN / FL infrastructure.

## 2. Frozen benchmark v1 scope

- Benchmark universe: current synthetic graph pool only (840 generated graph datasets)
- Current generators: chordal, watts
- No generator redesign inside benchmark v1

### Task set
- Main active task set: cycle2, cycle3, cycle4, cycle5
- Because cycle2 currently appears strongly generator-sensitive (especially limited support for watts), the main cross-regime analysis will focus primarily on cycle3-cycle5, while cycle2 is retained as an auxiliary task

### Primary heterogeneity axes
- label_prev_jsd_mean
- motif_profile_jsd_mean
- in_degree_jsd_mean

### Canonical suite size
- fixed subset size: 5 clients

### Canonical suites
- LP-low / LP-mid / LP-high
- MP-low / MP-mid / MP-high
- ID-low / ID-mid / ID-high

When selecting low/mid/high suites for one axis, the other two axes should be kept as non-extreme as possible.

## 3. Terminology

### local-single
One graph/client, one model.

### local-pooled-subset
A subset contains multiple clients. Each client trains its own separate local model, and results can be pooled at subset level.

### federated-global
A subset contains multiple clients. A global model is trained through federated aggregation (FedAvg first), and then evaluated on each client.

## 4. Immediate next package

The current proposed package for the coming weeks is:

1. freeze the benchmark-v1 scope;
2. characterize local-single learnability so that the base learner is not a major confounder;
3. select 9 canonical 5-client suites;
4. run the first subset-level comparison on the same suites:
   - local-pooled-subset
   - federated-global (FedAvg first);
5. then extend to one clustered or personalized method.

## 5. Explicit non-goals for benchmark v1

The following are not part of the current benchmark-v1 scope:
- designing a completely new graph generator immediately
- expanding the task set beyond cycle2-cycle5 now
- exhaustive deep hyperparameter search over all 840 graphs
- implementing many personalized/clustered methods at once

## 6. Questions for alignment

1. Is the above benchmark-v1 scope acceptable as the working scope for the coming weeks?
2. Is the immediate next package above sufficient as the target for the next review window?