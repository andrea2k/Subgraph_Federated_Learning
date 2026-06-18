from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.helper_funcs_multihead.fedala_run_helper import (
    build_fedala_log_row,
    run_fedala,
)
from andrea.helper_funcs_multihead.fl_run_helper import (
    iter_sweep_cfgs,
    upsert_experiment_rows,
)
from andrea.helper_funcs_multihead.load_client_helper import (
    ProgressPrinter,
    audit_loaded_q_label_masks,
    load_clients,
    parse_subset_clients,
)


SELECT_SUBSET_PATH = "clustering_q_label_heterogeneity"
SELECT_SUBSET = "selected_subset"

EXPERIMENT_LOG_FOLDER = "fedala_head_only_multihead_clustering_experiment"
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = "andrea/runs"

# -----------------------------------------------------------------------------
# Training setup. Kept aligned with the FedAvg/FedProx baselines.
# -----------------------------------------------------------------------------
ROUNDS = 80
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1.0
SELECTION_METRIC = "loss"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# FedALA setup.
#
# FEDALA_MODE="head_only" is the head-only FedALA ablation:
#   - server aggregation is still weighted FedAvg
#   - only output_head.heads.* parameters are ALA-interpolated client-side
#   - backbone/non-head parameters and buffers are copied from the global model
#
# alpha = 1 means use global parameter.
# alpha = 0 means preserve old local parameter.
# -----------------------------------------------------------------------------
FEDALA_MODE = "head_only"
FEDALA_ALA_LR = 1.0
FEDALA_RAND_PERCENT = 20.0
FEDALA_CONVERGENCE_STD = 0.1
FEDALA_CONVERGENCE_WINDOW = 10
FEDALA_MAX_STEPS = 100

# -----------------------------------------------------------------------------
# Sweep setup: same grid as the current baseline scripts.
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
    """Add old-compatible and q-specialized metadata to an experiment-log row."""
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
    ]
    for col in extra_cols:
        if col in row.index:
            log_row[f"manifest_{col}" if col == "subset_id" else col] = row.get(col, None)


def main() -> None:
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)
    id_to_client = load_clients(chosen_df, ALL_DATA_LOGS)
    audit_loaded_q_label_masks(chosen_df, id_to_client, strict=True)

    if os.environ.get("MASK_AUDIT_ONLY") == "1":
        print("MASK_AUDIT_ONLY=1 -> stopping after mask audit.")
        return

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    cols = ["task_profile_jsd_mean", "subset_id", "subset_size", "gamma", "mask_fraction"]
    cols = [col for col in cols if col in chosen_df.columns]
    print(chosen_df[cols])

    if "family_counts_json" in chosen_df.columns:
        chosen_df["family_counts"] = chosen_df["family_counts_json"].apply(
            lambda s: ", ".join(f"{k}: {v}" for k, v in json.loads(s).items())
        )
        print(chosen_df[["family_counts_json", "family_counts"]].to_string(index=False))

    only_seed = os.environ.get("ONLY_SEED")
    if only_seed is not None:
        active_seeds = [int(only_seed)]
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{only_seed}.csv"
        )
        print("working only with seed:", active_seeds)
    else:
        active_seeds = SEEDS
        experiment_log_csv = EXPERIMENT_LOG_CSV
        print("working with all seed:", active_seeds)

    debug = os.environ.get("FEDALA_DEBUG", "1") != "0"
    print("FEDALA_DEBUG:", debug)
    print("FEDALA_MODE:", FEDALA_MODE)
    print("FEDALA_ALA_LR:", FEDALA_ALA_LR)
    print("FEDALA_RAND_PERCENT:", FEDALA_RAND_PERCENT)

    sweep = list(
        iter_sweep_cfgs(
            base_cfg,
            seeds=active_seeds,
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

    print("\nMULTI-HEAD CONFIG SANITY CHECK")
    print("Number of sweep configs:", len(sweep))
    for debug_cfg, debug_meta in sweep[:1]:
        print("debug meta:", debug_meta)
        print("cfg output_head :", debug_cfg.get("output_head"))
        print("cfg architecture:", debug_cfg.get("architecture"))
        print("cfg num_layers  :", debug_cfg.get("num_layers"))
        print("cfg lr          :", debug_cfg.get("lr"))
        print("cfg batch_size  :", debug_cfg.get("batch_size"))

    total_runs = len(sweep) * len(chosen_df)
    print("Number of runs:", total_runs)
    progress = ProgressPrinter(total_runs)

    for cfg, meta in sweep:
        seed = int(meta["seed"])

        for _, row in chosen_df.iterrows():
            subset_ids = parse_subset_clients(row["subset_clients"])
            subset_clients = [id_to_client[graph_id] for graph_id in subset_ids]
            run_start = time.perf_counter()

            print()
            print(
                f"Starting with {len(subset_clients)}-client FedALA-HeadOnly training",
                row["subset_clients"],
            )
            print(meta)
            print("SEED:", seed, "ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
            for client in subset_clients:
                print(
                    "client:",
                    client.graph_id,
                    "dataset:",
                    client.dataset_id,
                    "mask_meta:",
                    client.mask_meta,
                )

            run_paths = run_fedala(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                device=DEVICE,
                mode=FEDALA_MODE,
                ala_lr=FEDALA_ALA_LR,
                ala_rand_percent=FEDALA_RAND_PERCENT,
                ala_convergence_std=FEDALA_CONVERGENCE_STD,
                ala_convergence_window=FEDALA_CONVERGENCE_WINDOW,
                ala_max_steps=FEDALA_MAX_STEPS,
                selection_metric=SELECTION_METRIC,
                debug=debug,
            )

            log_row = build_fedala_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=run_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                mode=FEDALA_MODE,
                ala_lr=FEDALA_ALA_LR,
                ala_rand_percent=FEDALA_RAND_PERCENT,
                ala_convergence_std=FEDALA_CONVERGENCE_STD,
                ala_convergence_window=FEDALA_CONVERGENCE_WINDOW,
                ala_max_steps=FEDALA_MAX_STEPS,
                selection_metric=SELECTION_METRIC,
            )
            add_subset_metadata(log_row, row)
            upsert_experiment_rows(experiment_log_csv, [log_row])

            progress.step(
                label=f"fedala_head_only done -> {run_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
