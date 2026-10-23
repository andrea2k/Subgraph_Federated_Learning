# Personalized Federated Graph Learning for Synthetic Financial-Crime Pattern Detection

**MSc thesis | TU Delft (2024–2026)**

**Python · PyTorch · PyTorch Geometric · Graph Neural Networks · Synthetic data generation · Federated learning · Reproducible experimentation**

## Overview

How can graph models identify suspicious structural patterns when a graph is distributed across multiple clients and each client sees only part of the topology?

This thesis investigated **personalized federated graph learning** using **synthetic directed multigraphs** with node-level pattern labels. I developed an experimental workflow for generating graph data, deriving labels, creating federated client partitions, training graph models and comparing centralized, federated and personalized approaches.

The application is inspired by **cross-institution financial-crime detection**, but **all reported benchmark experiments are synthetic**. No real bank transactions or customer records were used.

## Thesis contribution and reported result

My thesis explored **FedPTHA (Personalized Task Head Aggregation)** as an approach to personalized federated learning. The experiments examined how models behave when clients have different graph structures, label distributions and local tasks.

| Metric / setup | FedPTHA | FedAvg baseline |
| --- | ---: | ---: |
| **Macro PR-AUC** on a synthetic **20-client** graph benchmark | **71.61%** | **63.44%** |

**Interpretation:** FedPTHA improved the reported Macro PR-AUC by **8.17 percentage points** for this particular thesis benchmark and configuration. This is **not** a claim of general improvement across datasets or real-world financial institutions.

> **Reproduction note:** the 20-client result comes from the thesis experiment. Some default settings described below use **15** or **32** clients. Those default commands should **not** be assumed to reproduce the 20-client FedPTHA number without the matching experiment configuration and result artifacts.

## Technical workflow

```text
Synthetic directed multigraph generation
                 |
                 v
    Structural-pattern labels
                 |
                 v
   Train / validation / test graphs
                 |
                 v
 Federated client subgraph partitioning
  (Louvain / METIS / pattern-aware)
                 |
                 v
  Graph neural network training
   (PNA + federated baselines)
                 |
                 v
 Personalized aggregation experiments
                 |
                 v
   Evaluation and error analysis
```

### 1. Synthetic graph generation and labels

The benchmark follows the structural-pattern setting used in directed-multigraph research. It contains **11 node-labelling tasks**: degree and fan-in/fan-out patterns, plus higher-order motifs including scatter–gather, directed bicliques and directed cycles. Nodes are labelled according to whether they satisfy the associated structural criteria.

The documented default generator uses **8,192 nodes**, average degree **6** and radius parameter **11.1**. Generated graphs and labels are saved to `data/` along with label-count checks.

### 2. Federated graph partitioning

Different partitioning strategies simulate different levels of heterogeneity across clients:

- **Louvain / METIS** community-based partitioning with approximately balanced client sizes.
- **Zipf-skewed** partitions for imbalanced client sizes.
- **Label-controlled** partitions for more controlled comparisons of label distributions.
- **Pattern-aware / witness-based** partitions that distribute some structural motifs across clients, including configurable treatment of cross-client edges.

### 3. Graph models and training

The repository contains **Principal Neighborhood Aggregation (PNA)** implementations, including a basic full-batch model and a variant with reverse message passing, identity/port features and neighborhood sampling. It also contains federated training code and experiment configurations for baseline comparisons.

The research workflow included implementation, training, debugging, comparison with baselines, and repeated discussions with academic supervisors. This is an **experimental codebase**, not a deployed financial-crime detection service.

## Repository structure

| Folder / file | Description |
| --- | --- |
| `scripts/data/` | Synthetic graph generation and client partitioning |
| `scripts/training/` | Graph-model training entry points |
| `models/` | Graph-model implementations |
| `fed_algo/` | Federated learning components and algorithm code |
| `configs/` | Training and data-generation settings |
| `slurm_ablation_studies/` and `*.slurm` | Experiment and cluster-job scripts |
| `utils/` and `task/` | Supporting utilities and task logic |
| `requirements.txt` | Listed Python dependencies |
| `00_benchmark_v1_alignment.md` | Additional benchmark notes |

## Getting started

The commands below are entry points **documented by the repository**; they have not been validated in a fresh environment as part of this README rewrite. Model training may require a suitable PyTorch/PyG environment and significant compute resources.

```bash
git clone https://github.com/andrea2k/Subgraph_Federated_Learning.git
cd Subgraph_Federated_Learning
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate synthetic graphs

```bash
python3 -m scripts.data.generate_synthetic
```

Expected generated artifacts documented in the repository include `data/train.pt`, `data/val.pt`, `data/test.pt`, `data/y_sums.csv` and label-statistic outputs.

### Generate community-based federated splits

```bash
python3 -m scripts.data.make_federated_splits
```

This generates client subgraphs for the supported Louvain and METIS partition strategies. Pattern-aware splits are described in the generator's configuration and the original codebase.

### Run documented model baselines

```bash
python3 -m scripts.training.train_pna_baseline
python3 -m scripts.training.train_pna_reverse_mp
python3 -m scripts.training.train_federated_pna
```

These are **baseline training entry points**, not a validated one-command reproduction of the reported FedPTHA result. To replicate a specific thesis experiment, use its exact configuration, seed, aggregation implementation, client count and corresponding evaluation procedure. Consult the configuration files and SLURM scripts for the relevant experiment setup.

## Evaluation and reproducibility

The thesis comparisons focused on model performance under different graph partitioning and federated learning settings. The reported headline metric is **Macro PR-AUC**, which is useful when positive structural patterns are sparse.

Experiments were organized around configuration files and controlled seeds. The repository includes scripts for ablation studies and cluster execution. Exact numerical replication can still depend on software versions, compute environment and the specific experiment configuration.

## Limitations

- **Synthetic benchmark only:** results do not establish performance on real financial-crime cases.
- **No formal privacy guarantee claimed:** federated training was investigated as a decentralized learning setup, not as proof of differential privacy or secure multi-party computation.
- **Partition choices matter:** client heterogeneity and treatment of cross-client edges change the learning problem.
- **Research prototype:** the repository is not an API, dashboard or production-ready AML system.

## Research context

This work builds on research in directed-multigraph pattern detection, PNA graph neural networks and federated graph-learning benchmarks. The original repository's documentation provides methodological references, including Egressy et al. (2023), Corso et al. (2020) and the OpenFGL benchmark (2024).

**Author:** Andrea Li · **Institution:** Delft University of Technology (TU Delft) · **Project type:** MSc thesis research
