# Reproducing the experiments

## Train on the thesis benchmarks

The repository includes the original **18 graph files** used by both benchmarks. After [installation](environment.md), training needs no additional data download:

```bash
python -m fedptha train \
  --method fedavg --benchmark five_client \
  --seeds 0 --q 0.95 --rounds 2 \
  --output-dir results/runs/fedavg_example
```

Omitting `--seeds`, `--q`, and `--rounds` uses seeds 0–2, all four benchmark q values, and 80 rounds. Settings are defined in `configs/training.json`, `configs/benchmarks/`, and `configs/methods/`. See [methods](methods.md) and [evaluation](evaluation.md).

Outputs follow `results/runs/<group>/<benchmark>/<method>/q_<q>/seed_<seed>/`. Fully Local adds `client_<id>/`; physical-community methods add region directories. Existing completed results are reused for the same configuration. Choose another output directory for a different configuration.

## Generate a separate dataset

```bash
python -m fedptha generate-data --benchmark twenty_client
python -m fedptha generate-data --benchmark five_client
```

Outputs go to `data/generated/<benchmark>/`; `--output-dir` selects another destination. Each generated dataset includes `registry.csv` and `selected_subsets.csv`. Update the corresponding benchmark JSON's `registry` and `selected_subsets` paths to train on it.

The twenty-client command reconstructs the planted-community graphs from their recorded recipe. The five-client command copies the supplied benchmark realization because its recorded seeds do not reconstruct the cached graph exactly. Use the included graphs when comparing with the thesis results.

## SLURM and Apptainer

Use an Apptainer image containing the PyTorch/PyG stack described in [environment setup](environment.md). Submit from the repository root:

```bash
mkdir -p logs/slurm
export FEDPTHA_CONTAINER=/absolute/path/to/pytorch-pyg.sif
BENCHMARK_SETUP=twenty_client sbatch slurm/fedptha.slurm
```

Supply your cluster's account, partition, or QoS with `sbatch --account=... --partition=... --qos=...` as required. Each method job specifies time, CPU, GPU, and memory requests; adjust them for your cluster.

`TRAIN_SEED`, `TRAIN_Q`, and `TRAIN_ROUNDS` narrow a job. `RUN_TAG` chooses a run group under `results/runs/`; `FEDPTHA_RUNS_ROOT` overrides that group path. `FEDPTHA_PROJECT_ROOT` selects the checkout when submission happens elsewhere.

## Rebuild tables and figures

The [published AC summary](../results/paper_results.csv) and [thesis figures](../figures/generated) are included. Reporting scripts regenerate the analysis from the original experiment CSV logs under `results/frozen/`. Those logs remain local and are excluded from this repository.

For a checkout with the logs available:

```bash
python -m fedptha reproduce
```

This writes tables under `results/generated/` and figures under `figures/generated/`. New training outputs in `results/runs/` remain separate. Derived analysis files are regenerated when the command is repeated.

Use `--output-dir` and `--figures-dir` for separate destinations. Stage logs are in the output's `logs/` directory; `--start-at STAGE` resumes using existing intermediate outputs. Stage names are listed in `src/fedptha/reporting/pipeline.py`.
