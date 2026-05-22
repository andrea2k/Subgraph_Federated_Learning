from __future__ import annotations

import time
from pathlib import Path

import os
import pandas as pd
import torch
import json
from typing import Dict

from andrea.helper_funcs.fl_run_helper import (
    iter_sweep_cfgs,
    upsert_experiment_rows,
    build_local_log_row,
    run_local,
)
from andrea.helper_funcs.load_client_helper import (
    ProgressPrinter,
    parse_subset_clients,
    load_clients,
    audit_loaded_q_label_masks,
)


SELECT_SUBSET_PATH = "clustering_q_label_heterogeneity"
# SELECT_SUBSET_PATH = "clustering_masked"
# SELECT_SUBSET_PATH = "clustering_rep"
# SELECT_SUBSET_PATH = "clustering"

SELECT_SUBSET = "selected_subset"

EXPERIMENT_LOG_FOLDER = "local_clustering_experiment"
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

# SELECT_SUBSET_PATH = "heterogeneity"
# SELECT_SUBSET = "selected_pairs"

# EXPERIMENT_LOG_FOLDER = "pairwise_selection_experiment"
# DATA_DIR = "test_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = "andrea/runs"

# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------
ROUNDS = 80
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1.0
SELECTION_METRIC = "loss"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Sweep setup
# -----------------------------------------------------------------------------
SEEDS = [0, 1, 2]
MCW = ["auto"]
NUM_LAYERS = [6]
LRS = [0.001]
WEIGHT_DECAYS = [0.0001]
DROPOUTS = [0.1]
HIDDEN_DIMS = [64]
USE_EGO_IDS = [True]
BATCH_SIZE = [64]


def load_cfg(config_path: str, key: str) -> Dict:
    with open(config_path, "r") as f:
        all_cfg = json.load(f)
    cfg_obj = all_cfg[key]
    cfg = dict(cfg_obj.get("default_hparams", {}))
    cfg["model_name"] = cfg_obj.get("model_name", key)
    cfg["use_ego_ids"] = cfg_obj.get("use_ego_ids", True)
    cfg["use_port_ids"] = cfg_obj.get("use_port_ids", True)
    cfg["use_mini_batch"] = cfg_obj.get("use_mini_batch", False)
    cfg["batch_size"] = cfg_obj.get("batch_size", cfg.get("batch_size", 256))
    cfg["port_emb_dim"] = cfg_obj.get("port_emb_dim", cfg.get("port_emb_dim", 8))
    cfg["num_epochs"] = cfg_obj.get("num_epochs", 100)

    return cfg


def add_subset_metadata(log_row: Dict, row: pd.Series) -> None:
    """
    Add old-compatible and task-specialized metadata to an experiment-log row.
    """
    log_row["subset_clients"] = str(row["subset_clients"])

    log_row["gamma"] = (
        row["gamma"]
        if "gamma" in row.index and pd.notna(row["gamma"])
        else row.get("q_value", row.get("mask_fraction", None))
    )

    extra_cols = [
        # New q-controlled heterogeneity fields.
        "q_value",
        "q_iid",
        "q_other_share",
        "q_assigned_share",
        "q_allocation_mode",
        "global_visible_positive_support_fraction_ideal",
        # Backward-compatible old names.
        "mask_fraction",
        "specialization_fraction",
        "designed_heterogeneity",
        "global_visible_support_fraction_ideal",
        # Diagnostics.
        "task_profile_jsd_mean",
        "task_profile_jsd_median",
        "task_profile_jsd_max",
        "between_family_centroid_jsd_mean",
        # Benchmark metadata.
        "controlled_benchmark",
        "mask_mode",
        "family",
        "subset_id",
    ]

    for col in extra_cols:
        if col in row.index:
            log_row[f"manifest_{col}" if col == "subset_id" else col] = row.get(
                col, None
            )


def main():
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)
    id_to_client = load_clients(chosen_df, ALL_DATA_LOGS)

    audit_loaded_q_label_masks(chosen_df, id_to_client, strict=True)

    if os.environ.get("MASK_AUDIT_ONLY") == "1":
        print("MASK_AUDIT_ONLY=1 -> stopping after mask audit.")
        return

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    cols = [
        "task_profile_jsd_mean",
        "subset_id",
        "subset_size",
        "gamma",
        "mask_fraction",
    ]
    cols = [col for col in cols if col in chosen_df.columns]
    print(chosen_df[cols])

    cols = [
        "family_counts_json",
    ]
    cols = [col for col in cols if col in chosen_df.columns]
    chosen_df["family_counts"] = chosen_df["family_counts_json"].apply(
        lambda s: ", ".join(f"{k}: {v}" for k, v in json.loads(s).items())
    )

    print(chosen_df[cols].to_string(index=False))

    ONLY_SEED = os.environ.get("ONLY_SEED")
    if ONLY_SEED is not None:
        SEEDS = [int(ONLY_SEED)]
        print("working only with seed:", SEEDS)
        EXPERIMENT_LOG_CSV = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{ONLY_SEED}.csv"
        )
    else:
        SEEDS = [0, 1, 2]
        print("working with all seed:", SEEDS)
        EXPERIMENT_LOG_CSV = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv"
        )

    sweep = list(
        iter_sweep_cfgs(
            base_cfg,
            seeds=SEEDS,
            mcw=MCW,
            num_layers=NUM_LAYERS,
            lrs=LRS,
            weight_decays=WEIGHT_DECAYS,
            dropouts=DROPOUTS,
            hidden_dims=HIDDEN_DIMS,
            use_ego_ids=USE_EGO_IDS,
            batch_sizes=BATCH_SIZE,
        )
    )

    total_runs = len(sweep) * len(id_to_client)
    print("Number of runs:", total_runs)
    progress = ProgressPrinter(total_runs)

    for cfg, meta in sweep:
        seed = int(meta["seed"])

        for _, row in chosen_df.iterrows():

            subset_id = parse_subset_clients(row["subset_clients"])
            graph_to_family = json.loads(row["graph_to_family_json"])

            run_start = time.perf_counter()

            local_log_rows = []
            print()
            print(
                f"Starting with {len(id_to_client)} local training",
                row["subset_clients"],
            )
            print(meta)
            print("SEED:", seed, "ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)

            for graph_id in subset_id:
                client = id_to_client[graph_id]
                family = graph_to_family[str(graph_id)]
                mask_meta = client.mask_meta
                print(
                    f"starting with graph: {graph_id}, family: {family}, mask_meta: {mask_meta}"
                )
                local_paths = run_local(
                    client,
                    cfg,
                    seed,
                    RUNS_ROOT,
                    rounds=ROUNDS,
                    local_epochs=1,
                    device=DEVICE,
                    selection_metric=SELECTION_METRIC,
                )
                local_log_row = build_local_log_row(
                    client=client,
                    cfg=cfg,
                    seed=seed,
                    run_paths=local_paths,
                    rounds=ROUNDS,
                    local_epochs=1,
                    selection_metric=SELECTION_METRIC,
                )

                add_subset_metadata(local_log_row, row)
                local_log_row["family"] = str(family)

                local_log_rows.append(local_log_row)

                progress.step(
                    label=f"local done -> {local_paths.csv_path}",
                    run_start_time=run_start,
                )

            upsert_experiment_rows(EXPERIMENT_LOG_CSV, local_log_rows)


if __name__ == "__main__":
    main()
