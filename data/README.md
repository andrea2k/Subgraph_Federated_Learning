# Local graph data

- `shared_graphs/data_4200_3_10.0_chordal/`: the fixed five-client paper graph, with train/validation/test tensors.
- `planted/global/`: the connected four-region graph for the centralized reference.
- `planted/communities/`: four community graphs used by the twenty-client benchmark.
- `generated/`: separate outputs from the generation command.
- `exploration_pool/`: 569 historical parameter-search datasets, outside the active benchmark. Retained locally as historical input evidence; no current paper entrypoint reads them.

Tensors are excluded from Git. The benchmark JSON and CSV registries select the active data. The five-client cache has a historical seed-reconstruction limitation; see `docs/reproduction.md`.
