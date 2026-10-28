"""Run the published methods with shared JSON configuration."""

from __future__ import annotations
import importlib
import inspect
import json
import os
from fedptha.config import experiment_config
from fedptha.paths import absolute, frozen_root

METHODS = {
    "fully_local": ("fedptha.algorithms.averaging", "run_local"),
    "centralized": ("fedptha.algorithms.averaging", "run_local"),
    "fedavg": ("fedptha.algorithms.averaging", "run_fedavg"),
    "fedprox": ("fedptha.algorithms.averaging", "run_fedprox"),
    "fedala": ("fedptha.algorithms.fedala", "run_fedala"),
    "gcflplus": ("fedptha.algorithms.gcflplus", "run_gcfl"),
    "apple": ("fedptha.algorithms.apple", "run_apple"),
    "cwfedavg_task": ("fedptha.algorithms.cwfedavg", "run_task_cwfedavg_empirical"),
    **{
        m: ("fedptha.algorithms.fedptha", "run_fedptha")
        for m in (
            "fedptha",
            "fedptha_postala",
            "learned_only_postala",
            "support_only_postala",
            "oracle_physical",
        )
    },
    "specialist": ("fedptha.algorithms.specialist", "run_spec_select_train_bcast"),
    "hierarchical": ("fedptha.training.hierarchical", "run_experiment"),
}


def build_client_metadata(row, clients):
    def mapping(key):
        text = row.get(key)
        return json.loads(text) if isinstance(text, str) and text else {}

    families, tasks = mapping("graph_to_family_json"), mapping("graph_to_task_json")
    result = {}
    for client in clients:
        gid, mask = str(client.graph_id), client.mask_meta or {}
        result[gid] = dict(
            mask,
            family=families.get(gid),
            assigned_task=tasks.get(gid, mask.get("assigned_task")),
        )
    return result


def run(
    method, benchmark, *, seeds=None, q=None, rounds=None, output_dir=None, device=None
):
    import pandas as pd
    import torch
    from fedptha.data.clients import (
        load_clients,
        parse_subset_clients,
        load_global_centralized_client_from_subset_row,
    )
    from fedptha.training.runtime import iter_sweep_cfgs
    from fedptha.evaluation.protocols import PROTOCOLS

    settings = experiment_config(method, benchmark)
    if settings.get("benchmark_only", benchmark) != benchmark:
        raise ValueError(f"{method} requires {settings['benchmark_only']}.")
    bench = settings["benchmark_config"]
    chosen = pd.read_csv(absolute(bench["selected_subsets"]))
    active_q = bench["q_values"] if q is None else q
    if set(active_q) - set(bench["q_values"]):
        raise ValueError("q values must belong to the selected benchmark.")
    chosen = chosen[
        pd.to_numeric(chosen.q_value).round(10).isin([round(x, 10) for x in active_q])
    ].copy()
    if chosen.empty:
        raise ValueError("No benchmark subsets match the requested q values.")
    active_seeds = settings["seeds"] if seeds is None else seeds
    if not active_seeds or any(seed < 0 for seed in active_seeds):
        raise ValueError("At least one nonnegative seed is required.")
    rounds = settings["rounds"] if rounds is None else rounds
    if rounds < 1:
        raise ValueError("rounds must be positive.")
    model = dict(settings["model"], routing_mode=settings.get("routing", "adaptive"))
    refinement = settings.get("refinement", {})
    for key, value in refinement.items():
        model["apple_use_head_ala" if key == "enabled" else "apple_ala_" + key] = value
    model["result_algorithm"] = method
    model["selection_tag"] = "macro_pr_auc"
    root = absolute(output_dir or "results/runs/default") / benchmark / method
    if root == frozen_root() or frozen_root() in root.parents:
        raise ValueError("New training outputs must be outside results/frozen/.")
    root.mkdir(parents=True, exist_ok=True)
    resolved = dict(settings, rounds=rounds, model=model)
    config_path = root / "configuration.json"
    encoded = json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != encoded:
        raise ValueError(
            f"Different configuration already exists at {root}; choose a new --output-dir."
        )
    config_path.write_text(encoded)
    os.environ["FEDPTHA_METHOD"] = method
    registry = str(absolute(bench["registry"]))
    if method == "oracle_physical":
        from .oracle import expand_oracle_hard_clusters

        chosen = expand_oracle_hard_clusters(chosen)
    clients = None
    if method not in {"centralized", "hierarchical"}:
        clients = load_clients(
            chosen,
            csv_path=registry,
            verbose=False,
            mask_splits=tuple(bench["mask_splits"]),
        )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module, function = METHODS[method]
    train = getattr(importlib.import_module(module), function)
    parameters = settings["parameters"]
    sweep = iter_sweep_cfgs(
        model,
        seeds=active_seeds,
        mcw=[model["minority_class_weight"]],
        num_layers=[model["num_layers"]],
        lrs=[model["lr"]],
        weight_decays=[model["weight_decay"]],
        dropouts=[model["dropout"]],
        hidden_dims=[model["hidden_dim"]],
        use_ego_ids=[model["use_ego_ids"]],
        batch_sizes=[model["batch_size"]],
    )
    for cfg, meta in sweep:
        seed = int(meta["seed"])
        for _, row in chosen.iterrows():
            directory = root / f"q_{float(row.q_value):.2f}" / f"seed_{seed}"
            if method == "oracle_physical":
                directory = directory / f"region_{int(row.oracle_hard_cluster_id)}"
            if method == "hierarchical":
                outputs = [
                    train(
                        row,
                        cfg,
                        seed,
                        directory,
                        registry=registry,
                        rounds=rounds,
                        local_epochs=settings["local_epochs"],
                        parameters=parameters,
                        refinement=refinement,
                        device=device,
                    )
                ]
            else:
                if method == "centralized":
                    target = load_global_centralized_client_from_subset_row(
                        row, registry, verbose=False
                    )
                else:
                    target = [
                        clients[x] for x in parse_subset_clients(row["subset_clients"])
                    ]
                arguments = dict(
                    rounds=rounds,
                    local_epochs=settings["local_epochs"],
                    device=device,
                    client_fraction=settings["client_fraction"],
                    client_metadata=build_client_metadata(
                        row, target if isinstance(target, list) else [target]
                    ),
                    **parameters,
                )
                arguments = {
                    key: value
                    for key, value in arguments.items()
                    if key in inspect.signature(train).parameters
                }
                if method == "fully_local":
                    outputs = [
                        train(
                            client,
                            cfg,
                            seed,
                            directory / f"client_{client.graph_id}",
                            **arguments,
                        )
                        for client in target
                    ]
                else:
                    outputs = [train(target, cfg, seed, directory, **arguments)]
            for output in outputs:
                frame = pd.read_csv(output.csv_path)
                frame["protocol"] = frame["phase"].map(
                    lambda phase: next(
                        (
                            name
                            for name, (prefix, _, _) in PROTOCOLS.items()
                            if str(phase).startswith(prefix)
                        ),
                        None,
                    )
                )
                frame["method"] = method
                frame["q_value"] = float(row.q_value)
                frame["benchmark"] = benchmark
                frame.to_csv(output.csv_path, index=False)
