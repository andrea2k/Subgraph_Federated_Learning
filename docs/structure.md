# Code structure

The public commands follow two workflows:

```text
train:      JSON settings + graph tensors → training/algorithms → results/runs/
reproduce:  frozen experiment CSVs → reporting → results/generated/ + figures/generated/
```

## Entry and configuration

`src/fedptha/cli.py` defines the commands. `config.py` combines shared, benchmark, and method JSON settings; `paths.py` resolves paths relative to the checkout or `--project-root`. `__main__.py` enables `python -m fedptha`.

| Configuration | Purpose |
|---|---|
| `configs/training.json` | Architecture, optimizer, rounds, seeds, and participation |
| `configs/benchmarks/<benchmark>.json` | Graph/registry locations, q values, and benchmark overrides |
| `configs/benchmarks/<benchmark>/cluster_generation_parameters.csv` | Individual client records, graph locations, and task/mask metadata |
| `configs/benchmarks/<benchmark>/selected_subset.csv` | Clients participating in each q experiment |
| `configs/methods/<method>.json` | Algorithm parameters, routing, and optional refinement |

## Graphs and model

All module paths below are relative to `src/fedptha/`.

| Module | Responsibility |
|---|---|
| `data/generate.py`, `data/planted.py` | Benchmark generation and planted-community construction |
| `data/simulator.py`, `data/motifs.py`, `data/tasks.py` | Graph simulation, cycle detection, and node labels |
| `data/graph.py`, `data/hetero.py` | Graph objects, port/ego features, and forward/reverse edge views |
| `data/io.py` | Graph loading, including the class path encoded in the supplied tensors |
| `data/clients.py`, `data/allocation.py` | Client views and deterministic q-controlled positive-label allocation |
| `data/preprocessing.py` | Sampling-loader preparation and feature dimensions |
| `models/pna.py`, `models/heads.py` | Multi-PNA backbone and task heads |

## Algorithms and execution

| Module | Responsibility |
|---|---|
| `algorithms/fedptha.py` | FedPTHA, PostALA, and shared ablation training |
| `algorithms/routing.py` | Learned source weights, support priors, and personalized task-head mixing |
| `algorithms/averaging.py` | Fully Local, Global-Centralized, FedAvg, and FedProx |
| `algorithms/apple.py`, `fedala.py`, `gcflplus.py`, `cwfedavg.py`, `specialist.py` | Corresponding baseline or diagnostic updates |
| `training/experiment.py` | Method dispatch, seed/q iteration, output layout, and result annotations |
| `training/runtime.py` | Model/optimizer/loader construction and common result handling |
| `training/steps.py` | Batching, masked loss, training steps, and evaluation |
| `training/state.py`, `training/selection.py` | Parameter/buffer operations and validation Macro PR-AUC maximization |
| `training/random.py` | Random seeds |
| `training/oracle.py`, `training/hierarchical.py` | Physical-community grouping and hierarchical transfer |
| `evaluation/metrics.py`, `evaluation/protocols.py` | Metric definitions and AA/AC/CC protocols |

Selected states stay in memory. Training produces CSV metrics and method-specific sidecars. See [methods](methods.md) for method names and [evaluation](evaluation.md) for selection rules.

## Reporting

`reporting/pipeline.py` runs the analysis stages. `names.py` maps raw identifiers to paper names; `export.py` produces the public AC summary. `payload.py` and `regional_payload.py` compute communication payloads.

Within `reporting/tasks/`:

- `extract_results.py`, `summarize_results.py`, `extra_experiments.py`: extract and summarize experiment results.
- `protocol_analysis.py`, `ablations.py`: compare evaluation protocols and method components.
- `collaboration.py`, `collaboration_comparison.py`: analyze region, representative, hierarchical, and cluster diagnostics.
- `communication.py`, `communication_performance.py`, `diagnostic_tradeoff.py`: calculate exchange costs and performance tradeoffs.
- `appendix_tables.py`: build appendix tables.
- `selected_figures.py`, `figure_ablation.py`: create thesis Figures 3, 4, 6 and Figure 5, respectively.
- `figure_main_all.py`, `figure_protocols_all.py`, `figure_communication_all.py`: create appendix figures.

`slurm/<method>.slurm` submits a method job; `slurm/run.sh` invokes the same CLI inside Apptainer.
