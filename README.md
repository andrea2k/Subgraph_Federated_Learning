# Subgraph_Federated_Learning

A repository for **synthetic subgraph-detection** benchmarking and **PNA** baselines on directed multigraphs.

## Synthetic Graph Generation

This repository includes a **synthetic subgraph-detection dataset** used for benchmarking graph models for the pattern detection task. The graphs and labels are generated following the pseudocode/configurations described in **_Provably Powerful Graph Neural Networks for Directed Multigraphs_** (Egressy et al., 2023).

### What’s Included

- Three splits: **train**, **val**, **test**
- Saved as PyTorch tensors under `./data/`:

  - `train.pt`, `val.pt`, `test.pt` objects with node-level labels
  - `y_sums.csv` — per-split counts of positive labels per sub-task

- Per-split label percentages and mean across splits are stored under `./results/metrics/`, useful to sanity-check against the paper’s reported marginals.

---

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

### Reproducibility

Graph instances are **reproducible**. A single `BASE_SEED` deterministically derives distinct seeds for each split (train/val/test), ensuring:

- different graphs **within** a run for the splits,
- identical graphs **across** runs with the same `BASE_SEED`.

---

### Default Generation Settings

The default config (see the generator script `scripts/data/generate_synthetic.py`) follows the paper’s setup:

- Nodes `n = 8192`
- Average degree `d = 6`
- Radius parameter `r = 11.1`
- Directed multigraphs (for directed cycles)
- Generator: “chordal” / random-circulant-like
- One connected component per split (prevents data leakage)

---

### How to Generate

From the repo root:

```bash
python3 -m scripts.data.generate_synthetic
```

After step (1), you’ll find `train.pt`, `val.pt`, `test.pt`, and `y_sums.csv` under `./data/`. The `label_percentages.csv` will be saved under `./results/metrics/`.

## Federated Subgraph Partitioning

In the subgraph federated learning setting, each client is modeled as a subgraph extracted from a larger global graph. To simulate such clients, we apply two community-detection–based graph partitioning techniques to the global synthetic pattern-detection graph:

- **Metis-based split:** balanced k-way graph partitioning
- **Louvain-based split:** modularity-based community detection

Both approaches follow the methodology described in
**_OpenFGL: A Comprehensive Benchmark for Federated Graph Learning_** (Li et al., 2024).

### Original Splits (Zipf-Skewed Client Sizes)

To better reflect real-world financial crime detection environments, where institutions differ widely in size, we extend the original Metis and Louvain strategies with a **Zipf-skewed community assignment**. This mechanism ensures:

- a few large clients (analogous to large banks), and
- many small clients (smaller institutions),

which aligns with the heavy-tailed distribution of entity sizes commonly observed in financial networks.

### Label-Imbalance Splits (Controlled Setting)

For comparison, we also generate an easier and more controlled federated setup using **label imbalance handling**.
Here, communities are reassigned to clients based on their multi-task label distributions (following the OpenFGL LIS strategy), without Zipf-based skew.

These splits are useful for:

- validating the correctness of the federated learning pipeline,
- studying model behavior under balanced label and client-size distributions.

---

### How to Generate Federated Splits

From the repository root:

```bash
python3 -m scripts.data.make_federated_splits
```

This command produces four sets of federated datasets:

- `fed_louvain_splits/` — Louvain (original, Zipf-skewed)
- `fed_metis_splits/` — Metis (original, Zipf-skewed)
- `fed_louvain_imbalance_splits/` — Louvain with label imbalance handling
- `fed_metis_imbalance_splits/` — Metis with label imbalance handling

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
  Available options are: _metis original_, _louvain original_, _metis imbalance split_, and _louvain imbalance split_.

- **`global_epochs = 100`**
  The total number of global training rounds.

- **`local_epochs = 2`**
  Each client performs two passes over its local subgraph during every communication round.

- **`client_fraction = 1.0`**
  All clients participate in every communication round.

- **`algorithm`**
  Specifies the federated learning algorithm used in the experiment.

All configurations are available in `.configs/fed_configs.json` file.
