# Reproduction

`results/frozen/{main,additions,cwfedavg}` contains 1,068 canonical artifacts representing 348 logical runs. The final main-comparison manifest selects 240 method/benchmark/q/seed entries. Manifests identify accepted experiments; provenance explains repaired and replaced historical runs. These remain useful scientific records.

Large CSVs and graph tensors are excluded from Git. Copy them from the original experiment backup when setting up another machine. There is no public artifact download URL in this repository.

```bash
python -m fedptha reproduce
```

This rebuilds tables under `results/generated/` and Figures 3–6 plus appendix figures under `figures/generated/`. It reads frozen experiments, without retraining or switching to `results/runs/`. Historical method/protocol IDs are translated when presenting results. The preserved numerical reference tables are evidence; they are not inputs used to manufacture new scores.

Use `--output-dir` and `--figures-dir` for separate outputs. An interrupted run can resume with `--start-at STAGE` using its existing intermediate files. Stage logs are in the output's `logs/` directory. The normal pipeline no longer performs automatic regression comparisons.

## Graph data

```bash
python -m fedptha generate-data --benchmark twenty_client
python -m fedptha generate-data --benchmark five_client
```

These write separate datasets under `data/generated/<benchmark>/`. Repeated requests reuse completed output with the same recipe; `--output-dir` selects another destination.

The twenty-client generator reproduced the original 15 graphs and 90 tensor attributes in the validated environment.

**Five-client limitation:** the recorded seeds do not reconstruct the cached paper graph. The five-client command therefore materializes the original graph tensors and registries. The original tensors remain the benchmark source and are retained byte-for-byte. A small decoder in `data/io.py` resolves the class path encoded inside the old files.

Generated datasets include `registry.csv` and `selected_subsets.csv`. To train on them, update the corresponding benchmark JSON's `registry` and `selected_subsets` paths. A different graph realization changes the experiment.

## Cluster

Submit from the repository root:

```bash
mkdir -p logs/slurm
BENCHMARK_SETUP=twenty_client sbatch slurm/fedptha.slurm
```

`FEDPTHA_PROJECT_ROOT`, `FEDPTHA_CONTAINER`, and `FEDPTHA_CONTAINER_ROOT` configure locations. `TRAIN_SEED`, `TRAIN_Q`, and `TRAIN_ROUNDS` narrow a job. `RUN_TAG` or `FEDPTHA_RUNS_ROOT` selects a run group. JSON supplies the original algorithm parameters and the job files retain the original resource requests.
