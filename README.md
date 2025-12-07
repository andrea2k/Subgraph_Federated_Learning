# Subgraph_Federated_Learning

A repository for **synthetic subgraph-detection** benchmarking and **PNA** baselines on directed multigraphs.

This repository generates synthetic multigraphs with subgraph pattern labels, partitions them into federated subgraphs using Metis/Louvain, and trains centralized or federated PNA-based models for multi-task subgraph detection.

## Synthetic Graph Generation

This repository includes a **synthetic subgraph-detection dataset** used for benchmarking graph models for the pattern detection task. The graphs and labels are generated following the pseudocode/configurations described in **_Provably Powerful Graph Neural Networks for Directed Multigraphs_** (Egressy et al., 2023).

### Label Tasks

Each node is labeled for the presence of the following patterns (11 sub-tasks):

- `deg_in > 3`
- `deg_out > 3`
- `fan_in > 3`
- `fan_out > 3`
- `cycle2`
- `cycle3`
- `cycle4`
- `cycle5`
- `cycle6`
- `scatter_gather`
- `biclique`

---

### Default Generation Settings for Synthetic Graph

The default config (see the generator script `scripts/data/generate_synthetic.py`) follows the paper’s setup:

- Nodes `n = 8192`
- Average degree `d = 6`
- Radius parameter `r = 11.1`
- Directed multigraphs (for directed cycles)
- Generator: `chordal` / random-circulant-like
- One connected component per split (prevents data leakage)

---

### How to Generate

From the repository root, run:

```bash
python3 -m scripts.data.generate_synthetic
```

This command generates the synthetic pattern-detection graphs and saves the following files:

- `./data/train.pt`
- `./data/val.pt`
- `./data/test.pt`
- `./data/y_sums.csv` — positive-label counts per sub-task
- `./results/metrics/label_percentages.csv` — label percentages for sanity checking against the original paper statistics

## Federated Subgraph Partitioning

In the federated setting, each client is represented by a subgraph of the global synthetic graph. We use two community-detection–based partitioning techniques:

- **Metis:** balanced k-way graph partitioning
- **Louvain:** modularity-based community detection

Both follow the methodology of
**_OpenFGL: A Comprehensive Benchmark for Federated Graph Learning_** (Li et al., 2024), extended here for multi-task labels.

### Original Splits (Equal-Sized Clients)

The default experimental setup uses **approximately equal-sized clients**. After detecting communities, we assign them to clients using a greedy bin-packing strategy, producing subgraphs with similar node counts. This provides a controlled and stable federated environment for evaluating performance differences between centralized and decentralized training.

### Original Splits (Zipf-Skewed Clients)

To simulate more realistic financial crime settings with different client sizes, we additionally support **Zipf-skewed** splits. Communities are assigned to clients according to a Zipf-like distribution, producing:

- a few large clients,
- many small clients.

These splits model strongly **non-uniform client sizes**, common in real-world networks.

### Label-Imbalance Splits (LIS-Based)

We also provide **label-imbalance–aware** splits following the OpenFGL LIS strategy. Communities are clustered by their multi-task label distributions and grouped to reduce extreme label skew across clients.
These splits are useful for controlled experiments where label imbalance should be minimized.

---

### How to Generate Splits

From the repository root:

```bash
python3 -m scripts.data.make_federated_splits
```

This produces six datasets under `./data/`:

- `fed_louvain_splits/` — Louvain, equal-sized
- `fed_metis_splits/` — Metis, equal-sized
- `fed_louvain_splits_zipf_skewed/` — Louvain, Zipf-skewed
- `fed_metis_splits_zipf_skewed/` — Metis, Zipf-skewed
- `fed_louvain_imbalance_splits/` — Louvain, label-imbalance handling
- `fed_metis_imbalance_splits/` — Metis, label-imbalance handling

The training script selects the appropriate directory automatically using:

```json
"partition_strategy": "<strategy_name>"
```

(e.g., `"metis original"`, `"louvain original skewed"`, `"metis imbalance split"`)

## Principal Neighborhood Aggregation (PNA)

This repository provides two implementations of the **Principal Neighborhood Aggregation (PNA)** model, one baseline version using standard message passing, and an enhanced version that incorporates **Reverse Message Passing**, **Ego IDs**, **Port IDs**, and **mini-batch neighborhood sampling** for scalable training.

Both implementations follow the PNA architecture introduced in
**_Principal Neighbourhood Aggregation for Graph Nets_** (Corso et al., 2020).

### 1. Baseline PNA (Full-Batch Training)

The baseline model uses the original PNAConv layers from PyTorch Geometric and is trained in the **full-batch** setting.

To train and evaluate the baseline model:

```bash
python3 -m scripts.training.train_pna_baseline
```

The baseline model:

- operates directly on the homogeneous directed multigraph,
- uses full-batch message passing over the entire graph,
- does **not** use Ego IDs or Port IDs,
- serves as the reference for evaluating all incremental adaptations.

---

### 2. PNA with Reverse Message Passing (Mini-Batch Training)

This extended version incorporates several adaptations designed to improve pattern detection in directed multigraphs:

- **Reverse Message Passing** (direction-aware PNA aggregation)
- **Heterogeneous graph transformation** (`fwd` and `rev` edge types)
- **Ego ID embeddings** (to preserve seed-identity across sampled mini-batches)
- **Port ID embeddings** (to encode in/out-port numbers)
- **Mini-batch neighborhood sampling** using PyG’s `NeighborLoader`
- **Configurable fanout per hop** (default: `[10, 4]`)

To train and evaluate this model:

```bash
python3 -m scripts.training.train_pna_reverse_mp
```

This version serves as the foundation for future **federated** extensions.

---

### 3. Training Configuration

Both PNA variants share the following core hyperparameters (taken from `default_hparams`):

- **`hidden_dim = 64`**
  Dimensionality of node embeddings throughout the network.

- **`num_layers = 2`**
  Number of GNN layers in the model.

- **`dropout = 0.1`**
  Dropout rate applied during training to reduce overfitting.

- **`lr = 0.001`**
  Learning rate used by the Adam optimizer.

- **`weight_decay = 0.0001`**
  L2 regularization strength to prevent overfitting.

Additional hyperparameters apply to the **`reverse_mp_with_port_and_ego`** mini-batch model:

- **`batch_size = 32`**
  Number of seed nodes sampled per mini-batch.

- **`neighbors_per_hop = [10, 4]`**
  Number of neighbors sampled at each hop for scalable neighborhood expansion.

- **`ego_dim = 32`**
  Embedding dimension used to encode ego-node identity across batches.

- **`port_emb_dim = 8`**
  Embedding dimension for port IDs, capturing in-/out-port structural information.

- **`minority_class_weight = "auto"`**
  Class-weighting strategy. When set to _auto_, the loss function computes per-task positive-class weights from the training labels.

All configurations are available in `.configs/pna_configs.json` file.

## PNA Training Under Federated Setting

To train and evaluate the PNA model in the federated learning setting:

```bash
python3 -m scripts.training.train_federated_pna
```

### Federated Learning Configuration

The federated setting introduces additional hyperparameters governing both the **client-generation process** and the **federated training procedure**. This section documents the default configuration used throughout the experiments, along with a brief rationale for each choice.

#### Federated Dataset Simulation

- **`num_clients = 32`**
  The 8,192-node global graph is partitioned into 32 subgraphs, yielding approximately 256 nodes per client.
  This creates a **realistically challenging** federated scenario: clients are small enough to introduce non-IID behavior but large enough to support stable local training.

- **`louvain_resolution = 1.0`**
  Uses the default modularity resolution for Louvain community detection.

- **`metis_num_coms = 32`**
  The Metis partitioning strategy is configured to produce exactly 32 partitions, ensuring that **each client corresponds to one contiguous graph community**, which maximizes structural separation between clients.

#### Federated Learning Hyperparameters

- **`partition_strategy`**
  Selects the partitioning strategy used in the experiment.
  Supported options:

  - `metis original`
  - `louvain original`
  - `metis original skewed`
  - `louvain original skewed`
  - `metis imbalance split`
  - `louvain imbalance split`

- **`global_epochs = 100`**
  The total number of global training rounds.

- **`local_epochs = 2`**
  Each client performs two passes over its local subgraph during every communication round.

- **`client_fraction = 1.0`**
  All clients participate in every communication round.

- **`algorithm`**
  Specifies the federated learning algorithm used in the experiment.

All configurations are available in `.configs/fed_configs.json` file.

## Reproducibility

All datasets, federated splits, and training results in this repository are **fully reproducible**.

The entire pipeline uses a shared seed-derivation mechanism:

- A global `BASE_SEED` is defined in the config.
- Each script calls `set_seed(BASE_SEED)`.
- Task-specific seeds (e.g., `"train"`, `"val"`, `"louvain"`, `"metis"`) are deterministically obtained via `derive_seed(BASE_SEED, tag)`.

This ensures:

- **Synthetic graphs** (`train.pt`, `val.pt`, `test.pt`) are identical across runs.
- **Federated subgraph partitions** (Metis/Louvain, equal-sized, Zipf-skewed, LIS) are reproduced exactly.
- **Training runs**—centralized and federated—are stable and repeatable, including model initialization, mini-batch sampling, and client sampling.

Changing the `BASE_SEED` produces a new, independent experiment instance while preserving internal consistency across all components.
