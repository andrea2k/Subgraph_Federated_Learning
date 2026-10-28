# Methods and configuration

`configs/training.json` contains shared architecture and training values. `configs/benchmarks/*.json` selects graph data, client registries, q values, and benchmark-specific settings. `configs/methods/*.json` contains only method parameters, optional routing mode, and optional head refinement.

| CLI method | Meaning |
|---|---|
| `fully_local` | Independent training and model selection for each client |
| `centralized` | Complete-label training on the unsplit graph |
| `fedavg`, `fedprox` | Global model averaging; FedProx adds its local proximal penalty |
| `gcflplus` | Learned clustering and aggregation within clusters |
| `fedala` | Adaptive local aggregation before local training |
| `apple` | Original whole-model, unconstrained source weights |
| `cwfedavg_task` | Positive-support-based, task-conditioned aggregation of whole models |
| `fedptha` | Global backbone and personalized task heads |
| `fedptha_postala` | FedPTHA plus local head refinement |
| `learned_only_postala` | Ablation without the positive-label support prior |
| `support_only_postala` | Ablation without learned source weights |
| `oracle_physical` | Four independent physical-community federations; 20-client only |
| `hierarchical` | Physical-community training followed by inter-community head transfer; 20-client only |
| `specialist` | Specialist representative training and transfer; 20-client only |

Shared values stay at six layers, hidden dimension 64, batch size 64, Adam learning rate 0.001, weight decay 0.0001, dropout 0.1, and positive-class weight cap 100. GCFL+ keeps its different five-/twenty-client quantiles.

Training state means the current models, optimizer states, routing coefficients, and best validation-selected states. These are in memory. The published CSV-only workflow does not save model checkpoints.

Reporting translates the raw identifiers `support_simplex` and `support_simplex_postala` to FedPTHA and FedPTHA + PostALA.
