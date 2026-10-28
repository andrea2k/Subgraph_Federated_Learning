# Benchmark graphs

The repository includes the original train/validation/test graph tensors for both thesis benchmarks:

| Directory | Graphs |
|---|---|
| `shared_graphs/data_4200_3_10.0_chordal/` | Five-client shared graph: 3 splits |
| `planted/global/` | Connected four-community graph for Global-Centralized: 3 splits |
| `planted/communities/` | Four physical-community graphs: 3 splits each |

The **18 `.pt` files** total approximately **5 MB**. [manifest.csv](manifest.csv) records their paths, sizes, and SHA-256 hashes. Client registries under `configs/benchmarks/` identify the graphs and task assignments; client label masks are applied during loading.

`python -m fedptha generate-data --benchmark twenty_client` writes a separate dataset under `data/generated/`. See [reproduction](../docs/reproduction.md) for generation settings and the five-client graph-reconstruction constraint.
