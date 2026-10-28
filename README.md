# FedPTHA: Personalized Task Head Aggregation for Federated Graph Learning

Code and benchmarks for an MSc thesis at TU Delft. FedPTHA shares a Multi-PNA backbone across clients and personalizes the prediction heads for individual graph tasks. PostALA provides optional local head refinement.

Each node has five binary labels indicating membership in directed cycles of lengths 2–6. Experiments use **5 or 20 clients**, **q ∈ {0.20, 0.50, 0.80, 0.95}**, and **seeds 0, 1, 2**. The original benchmark graphs are included in this repository.

## Setup

Use Python **3.10–3.12**. These commands install the reference PyTorch 2.6.0 stack on Linux with CPU support:

```bash
git clone https://github.com/andrea2k/Subgraph_Federated_Learning.git
cd Subgraph_Federated_Learning
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pyg_lib torch_scatter torch_sparse --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
python -m pip install -e ".[report]"
```

See [environment setup](docs/environment.md) for macOS and CUDA installations. Sampling extensions must match the installed PyTorch/CUDA version; the commands above follow the official [PyTorch](https://pytorch.org/get-started/previous-versions/#v260) and [PyG](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html) installation instructions.

## Run an experiment

Start with one seed, one q value, and two rounds:

```bash
python -m fedptha train \
  --method fedptha --benchmark five_client \
  --seeds 0 --q 0.95 --rounds 2 \
  --output-dir results/runs/example
```

This uses the full model architecture. To run the complete FedPTHA twenty-client experiment with **80 rounds, all three seeds, and all four q values**:

```bash
python -m fedptha train \
  --method fedptha --benchmark twenty_client \
  --output-dir results/runs/paper
```

Change `--method` to run a baseline or ablation. [Methods and configuration](docs/methods.md) lists the available methods. Use `--dry-run` to inspect settings or `--device cpu` / `--device cuda` to select a device.

Training writes:

```text
results/runs/example/five_client/fedptha/
├── configuration.json
└── q_0.95/seed_0/
    ├── metrics.csv
    └── routing.csv
```

`metrics.csv` contains training and evaluation results, including the selected **AA, AC, and CC** evaluations. Methods add refinement, cluster, or representative CSVs where applicable. Model selection uses validation Macro PR-AUC; selected states remain in memory. Use a separate output directory when changing configuration.

## Thesis results and figures

The published AC summary is [results/paper_results.csv](results/paper_results.csv). Figures 1–2 are in [figures/source](figures/source); Figures 3–6 and appendix figures are in [figures/generated](figures/generated).

The reporting code rebuilds tables and figures from saved experiment logs. These logs are local and excluded from Git; see [analysis and reproduction](docs/reproduction.md#rebuild-tables-and-figures) for the command and expected inputs.

## Project structure

| Location | Contents |
|---|---|
| `src/fedptha/data/` | Graph generation, loading, and client label masks |
| `src/fedptha/models/` | Multi-PNA backbone and task heads |
| `src/fedptha/algorithms/` | FedPTHA, baselines, routing, and ablations |
| `src/fedptha/training/` | Experiment execution, training steps, and in-memory model selection |
| `src/fedptha/evaluation/` | Metrics and AA/AC/CC definitions |
| `src/fedptha/reporting/` | Result extraction, statistics, tables, and figures |
| `configs/` | Shared training settings, benchmark registries, and method parameters |
| `data/` | Original benchmark graph tensors and their checksum manifest |
| `results/` | Published summary and local training/analysis outputs |
| `figures/` | Thesis and appendix figures |
| `slurm/` | Cluster job scripts |
| `docs/` | Setup, code structure, methods, evaluation, and reproduction guides |

See [code structure](docs/structure.md) for the role of each module and [evaluation rules](docs/evaluation.md) for the metric definitions. Cluster setup and graph regeneration are covered in [reproduction](docs/reproduction.md).
