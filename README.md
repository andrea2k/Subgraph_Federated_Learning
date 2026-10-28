# FedPTHA: Personalized Task Head Aggregation for Federated Graph Learning

TU Delft MSc thesis research on synthetic directed multigraphs. Each node has five binary labels: membership in directed cycles of lengths 2–6. The benchmarks have **5 or 20 clients**, four heterogeneity levels (`q = 0.20, 0.50, 0.80, 0.95`), and three training seeds (`0, 1, 2`). FedPTHA shares a Multi-PNA backbone and personalizes the task heads; PostALA adds optional head refinement.

## Setup

Use the existing local environment:

```bash
source thesis/bin/activate
```

For a new environment, use Python 3.10+ with compatible PyTorch/PyG sampling extensions and install the package:

```bash
pip install -e ".[report,sampling]"
```

See [environment notes](docs/environment.md). Graph tensors and large frozen result CSVs are local artifacts excluded from Git; another machine needs a copy of them.

## Train

Run FedPTHA with the original parameters: **80 rounds, all three seeds, all four q values**.

```bash
python -m fedptha train \
  --method fedptha \
  --benchmark twenty_client \
  --output-dir results/runs/paper
```

Choose one method, seed, and q:

```bash
python -m fedptha train \
  --method fedavg \
  --benchmark five_client \
  --seeds 0 --q 0.95 --rounds 80 \
  --output-dir results/runs/fedavg_trial
```

For a short run, use `--rounds 2` and a separate output directory. It uses the normal model architecture. `--dry-run` displays configuration without training; `--device cpu` or `--device cuda` selects the device. [Methods and parameters](docs/methods.md) lists the supported methods; `python -m fedptha train --help` lists the options.

Results have predictable paths:

```text
results/runs/paper/twenty_client/fedptha/
    configuration.json
    q_0.95/seed_0/
        metrics.csv
        routing.csv
```

Methods add `refinement.csv`, `clusters.csv`, `representatives.csv`, or `task_distributions.csv` where needed. Fully Local uses `client_<id>/` subdirectories. Oracle physical-community runs use `region_<id>/`; hierarchical runs also retain their intra-region outputs and `hierarchy_weights.csv`. `metrics.csv` includes per-round results and selected AA/AC/CC evaluations. Best model states stay in memory; no model checkpoints are written.

Existing completed metric files are reused for the same configuration. A changed configuration requires a separate output directory. New training cannot write into `results/frozen/`.

## Data and thesis analysis

```bash
python -m fedptha generate-data --benchmark twenty_client
python -m fedptha reproduce
```

`generate-data` writes a separate dataset under `data/generated/`. The five-client command copies the original cached graph realization; its recorded seed recipe cannot reconstruct that realization. See [reproduction](docs/reproduction.md) for details.

`reproduce` reads **frozen experiments**, rebuilds thesis tables under `results/generated/`, and generates Figures 3–6 and appendix figures under `figures/generated/`. It does not train models or automatically use new runs. Figures 1–2 are in [figures/source](figures/source).

Submit a cluster job from the repository root:

```bash
mkdir -p logs/slurm
BENCHMARK_SETUP=twenty_client sbatch slurm/fedptha.slurm
```

## Layout

| Location | Purpose |
|---|---|
| `src/fedptha/data`, `models` | Graph generation/loading, client masks, and Multi-PNA |
| `src/fedptha/algorithms` | Method-specific updates and routing |
| `src/fedptha/training` | Shared loaders, training/evaluation steps, in-memory state selection, and run dispatch |
| `src/fedptha/evaluation`, `reporting` | Metrics, result extraction, statistics, and thesis figures |
| `configs` | JSON training settings, benchmark registries, and method parameters |
| `slurm` | Method jobs and shared container launcher |
| `data` | Original and separately generated graph tensors |
| `results/frozen` | Original experiment CSVs, scientific manifests, and provenance |
| `results/runs`, `results/generated` | New training results and rebuilt analysis |
| `figures/source`, `figures/generated` | Source diagrams and generated thesis/appendix figures |
| `experiments/scratch` | Independent exploration outside the thesis pipeline |

Read [evaluation rules](docs/evaluation.md) and the [completed validation record](docs/validation.md). Migration tests, source archives, and automatic comparison tools have been removed after validation.
