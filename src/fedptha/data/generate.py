"""Generate the two published benchmarks into an explicit output directory."""

from __future__ import annotations
import json
import shutil
import sys
from fedptha.config import read_config
from fedptha.paths import absolute


def generate(benchmark: str, output_dir=None, *, overwrite=False):
    import pandas as pd
    import torch

    config = read_config(f"configs/benchmarks/{benchmark}.json")
    output = absolute(output_dir or f"data/generated/{benchmark}")
    signature = json.dumps(
        dict(generation=config["generation"], q_values=config["q_values"]),
        sort_keys=True,
    )
    marker = output / "generation_parameters.json"
    if output.exists() and any(output.iterdir()) and not overwrite:
        if marker.is_file() and marker.read_text() == signature + "\n":
            registry = pd.read_csv(output / "registry.csv")
            for directory in set(registry.data_dir):
                if not all(
                    (absolute(directory) / f"{split}.pt").is_file()
                    for split in ("train", "val", "test")
                ):
                    raise FileNotFoundError(f"Incomplete cached benchmark: {directory}")
            print(f"Using generated benchmark: {output}")
            return output
        raise FileExistsError(
            f"{output} already exists; choose a new --output-dir or use --overwrite."
        )
    # Never overwrite the canonical graph trees or the experiment evidence.
    if (
        output == absolute("data")
        or absolute("results") == output
        or absolute("results") in output.parents
    ):
        raise ValueError(
            "Generation output must be a dedicated directory outside results/."
        )
    if overwrite and output.exists():
        if not marker.is_file():
            raise ValueError(
                "Refusing to overwrite a directory not created by this generator."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if benchmark == "five_client":
        # The historical five-client graph cache does not match its recorded seed recipe.
        # Preserve the published graph realization rather than silently replacing it.
        from .io import load_graph

        spec = config["generation"]
        graph_dir = output / spec["dataset_id"]
        graph_dir.mkdir()
        source_registry = pd.read_csv(absolute(config["registry"]))
        sources = set(source_registry.data_dir)
        if len(sources) != 1:
            raise ValueError(
                "Five-client benchmark requires one canonical shared graph."
            )
        source = absolute(next(iter(sources)))
        for split in ("train", "val", "test"):
            torch.save(load_graph(source / f"{split}.pt"), graph_dir / f"{split}.pt")
        print(
            "Five-client benchmark: reusing the published graph realization; recorded seed recipe is historical."
        )
        registry = pd.read_csv(absolute(config["registry"]))
        registry["data_dir"] = str(graph_dir)
        registry.to_csv(output / "registry.csv", index=False)
        shutil.copy2(
            absolute(config["selected_subsets"]), output / "selected_subsets.csv"
        )
    else:
        from . import planted

        spec = config["generation"]
        argv = [
            "generate-planted",
            "--nodes-per-community",
            str(spec["nodes_per_community"]),
            "--community-ds",
            ",".join(map(str, spec["community_ds"])),
            "--community-deltas",
            ",".join(map(str, spec["community_deltas"])),
            "--bridge-edges-per-pair",
            str(spec["bridge_edges_per_pair"]),
            "--seed",
            str(spec["seed"]),
            "--q-values",
            ",".join(map(str, config["q_values"])),
            "--global-out-root",
            str(output / "global"),
            "--community-client-root",
            str(output / "communities"),
            "--q-manifest-dir",
            str(output / "manifests"),
        ]
        old = sys.argv
        try:
            sys.argv = argv
            planted.main()
        finally:
            sys.argv = old
        shutil.copy2(
            output / "manifests/cluster_generation_parameters.csv",
            output / "registry.csv",
        )
        shutil.copy2(
            output / "manifests/selected_subset.csv", output / "selected_subsets.csv"
        )
    marker.write_text(signature + "\n")
    print(f"Generated benchmark: {output}")
    return output
