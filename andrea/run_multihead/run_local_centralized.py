from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.helper_funcs_multihead.fl_run_helper import (
    iter_sweep_cfgs,
    upsert_experiment_rows,
    build_local_log_row,
    run_local,
)
from andrea.helper_funcs_multihead.load_client_helper import (
    ProgressPrinter,
    load_local_centralized_client_from_subset_row,
    audit_local_centralized_client,
    print_client_debug_summary,
)


SELECT_SUBSET_PATH = "clustering_q_label_heterogeneity"
SELECT_SUBSET = "selected_subset"

EXPERIMENT_LOG_FOLDER = "local_centralized_multihead_clustering_experiment"
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

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
    Add old-compatible and q-controlled metadata to an experiment-log row.
    """
    log_row["subset_clients"] = str(row["subset_clients"])

    log_row["gamma"] = (
        row["gamma"]
        if "gamma" in row.index and pd.notna(row["gamma"])
        else row.get("q_value", row.get("mask_fraction", None))
    )

    extra_cols = [
        "q_value",
        "q_iid",
        "q_other_share",
        "q_assigned_share",
        "q_allocation_mode",
        "global_visible_positive_support_fraction_ideal",
        "mask_fraction",
        "specialization_fraction",
        "designed_heterogeneity",
        "global_visible_support_fraction_ideal",
        "task_profile_jsd_mean",
        "task_profile_jsd_median",
        "task_profile_jsd_max",
        "between_family_centroid_jsd_mean",
        "controlled_benchmark",
        "mask_mode",
        "family",
        "subset_id",
        "base_graph_ids_json",
        "base_dataset_ids_json",
    ]

    for col in extra_cols:
        if col in row.index:
            log_row[f"manifest_{col}" if col == "subset_id" else col] = row.get(
                col, None
            )


def relabel_result_csv(
    csv_path,
    *,
    run_type: str,
    algorithm: str,
    extra: Dict | None = None,
) -> None:
    """
    run_local(...) writes local-style rows.
    Patch them to local_centralized for clean plotting later.
    """
    df = pd.read_csv(csv_path)

    df["run_type"] = run_type
    df["algorithm"] = algorithm

    if extra:
        for key, value in extra.items():
            df[key] = value

    df.to_csv(csv_path, index=False)


def main():
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)
    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    cols = [
        "task_profile_jsd_mean",
        "subset_id",
        "subset_size",
        "gamma",
        "mask_fraction",
        "q_value",
        "base_graph_ids_json",
        "base_dataset_ids_json",
    ]
    cols = [col for col in cols if col in chosen_df.columns]
    print(chosen_df[cols])

    ONLY_SEED = os.environ.get("ONLY_SEED")
    if ONLY_SEED is not None:
        SEEDS_USED = [int(ONLY_SEED)]
        print("working only with seed:", SEEDS_USED)
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{ONLY_SEED}.csv"
        )
    else:
        SEEDS_USED = [0, 1, 2]
        print("working with all seeds:", SEEDS_USED)
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv"
        )

    sweep = list(
        iter_sweep_cfgs(
            base_cfg,
            seeds=SEEDS_USED,
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

    total_runs = len(sweep) * len(chosen_df)
    print("Number of local-centralized runs:", total_runs)
    progress = ProgressPrinter(total_runs)

    for cfg, meta in sweep:
        seed = int(meta["seed"])

        for _, row in chosen_df.iterrows():
            run_start = time.perf_counter()

            client = load_local_centralized_client_from_subset_row(
                row,
                ALL_DATA_LOGS,
                verbose=True,
            )
            audit_local_centralized_client(client, strict=True)
            if os.environ.get("DEBUG_CLIENT", "0") == "1":
                print_client_debug_summary(
                    client,
                    name=f"LOCAL_CENTRALIZED | q={row.get('q_value', None)}",
                )
            print()
            print("Starting local-centralized upper-bound training")
            print("subset_clients:", row["subset_clients"])
            print("base_graph_ids_json:", row.get("base_graph_ids_json", None))
            print("q_value:", row.get("q_value", None))
            print(meta)
            print("SEED:", seed, "ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)

            local_paths = run_local(
                client,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                device=DEVICE,
                selection_metric=SELECTION_METRIC,
            )

            extra_result_cols = {
                "training_scope": "unmasked_base_graph",
                "communication": "none",
                "baseline_role": "upper_bound_full_supervision",
                "subset_clients": str(row["subset_clients"]),
                "q_value": row.get("q_value", None),
                "mask_fraction": row.get("mask_fraction", None),
                "base_graph_ids_json": row.get("base_graph_ids_json", None),
                "base_dataset_ids_json": row.get("base_dataset_ids_json", None),
            }

            relabel_result_csv(
                local_paths.csv_path,
                run_type="local_centralized",
                algorithm="local_centralized",
                extra=extra_result_cols,
            )

            local_log_row = build_local_log_row(
                client=client,
                cfg=cfg,
                seed=seed,
                run_paths=local_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                selection_metric=SELECTION_METRIC,
            )

            add_subset_metadata(local_log_row, row)

            local_log_row["run_type"] = "local_centralized"
            local_log_row["algorithm"] = "local_centralized"
            local_log_row["training_scope"] = "unmasked_base_graph"
            local_log_row["communication"] = "none"
            local_log_row["baseline_role"] = "upper_bound_full_supervision"
            local_log_row["display_name"] = "Local-centralized upper bound"

            # Important: local-centralized is subset-level, not one row per virtual client family.
            local_log_row["family"] = "local_centralized_base_graph"
            local_log_row["subset_clients"] = str(row["subset_clients"])
            local_log_row["base_graph_ids_json"] = row.get("base_graph_ids_json", None)
            local_log_row["base_dataset_ids_json"] = row.get(
                "base_dataset_ids_json", None
            )

            upsert_experiment_rows(experiment_log_csv, [local_log_row])

            progress.step(
                label=f"local_centralized done -> {local_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
