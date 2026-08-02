from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.helper_funcs_multihead.fedala_run_helper_multi_select import (
    build_fedala_log_row,
    run_fedala,
)
from andrea.helper_funcs_multihead.fl_run_helper_multi_select import (
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
RUN_TAG = os.environ.get("RUN_TAG", "multiselect_fedala")
EXPERIMENT_LOG_FOLDER = f"fedala_head_only_multihead_clustering_experiment_multi_select_{RUN_TAG}"
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = os.environ.get("RUNS_ROOT", "andrea/runs_multiselect_fedala")

ROUNDS = int(os.environ.get("ROUNDS", 80))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", 1.0))
SELECTION_METRICS = ["loss", "micro_f1", "macro_pos_f1", "micro_pr_auc", "macro_pr_auc"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEDALA_MODE = "head_only"
FEDALA_ALA_LR = float(os.environ.get("FEDALA_ALA_LR", 1.0))
FEDALA_RAND_PERCENT = float(os.environ.get("FEDALA_RAND_PERCENT", 20.0))
FEDALA_CONVERGENCE_STD = float(os.environ.get("FEDALA_CONVERGENCE_STD", 0.1))
FEDALA_CONVERGENCE_WINDOW = int(os.environ.get("FEDALA_CONVERGENCE_WINDOW", 10))
FEDALA_MAX_STEPS = int(os.environ.get("FEDALA_MAX_STEPS", 100))

SEEDS = [0, 1, 2]
MCW = ["auto"]
NUM_LAYERS = [6]
LRS = [0.001]
WEIGHT_DECAYS = [0.0001]
DROPOUTS = [0.1]
HIDDEN_DIMS = [64]
USE_EGO_IDS = [True]
BATCH_SIZE = [64]
MASK_SPLITS = ("train", "val", "test")


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
    log_row["subset_clients"] = str(row["subset_clients"])
    log_row["gamma"] = row["gamma"] if "gamma" in row.index and pd.notna(row["gamma"]) else row.get("q_value", row.get("mask_fraction", None))
    extra_cols = [
        "q_value", "q_iid", "q_other_share", "q_assigned_share", "q_allocation_mode",
        "global_visible_positive_support_fraction_ideal", "mask_fraction",
        "specialization_fraction", "designed_heterogeneity", "global_visible_support_fraction_ideal",
        "task_profile_jsd_mean", "task_profile_jsd_median", "task_profile_jsd_max",
        "between_family_centroid_jsd_mean", "controlled_benchmark", "mask_mode", "family", "subset_id",
    ]
    for col in extra_cols:
        if col in row.index:
            log_row[f"manifest_{col}" if col == "subset_id" else col] = row.get(col, None)


def main() -> None:
    print("RUN VARIANT: FedALA-HeadOnly (multi-selection) | selection_metrics=" + "|".join(SELECTION_METRICS))
    print("RUNS_ROOT:", RUNS_ROOT)
    print("EXPERIMENT_LOG_FOLDER:", EXPERIMENT_LOG_FOLDER)
    print("ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
    print("SAVE_CHECKPOINTS:", os.environ.get("SAVE_CHECKPOINTS", "0"))

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    only_q = os.environ.get("ONLY_Q")
    if only_q is not None and "q_value" in chosen_df.columns:
        q = float(only_q)
        chosen_df = chosen_df[pd.to_numeric(chosen_df["q_value"], errors="coerce").round(10).eq(round(q, 10))].copy()
        print("ONLY_Q filter:", only_q, "remaining subsets:", len(chosen_df))

    max_subsets = os.environ.get("MAX_SUBSETS")
    if max_subsets is not None:
        chosen_df = chosen_df.head(int(max_subsets)).copy()
        print("MAX_SUBSETS filter:", max_subsets, "remaining subsets:", len(chosen_df))

    id_to_client = load_clients(chosen_df, ALL_DATA_LOGS, mask_splits=MASK_SPLITS)
    audit_loaded_q_label_masks(chosen_df, id_to_client, strict=True, mask_splits=MASK_SPLITS)

    if os.environ.get("MASK_AUDIT_ONLY") == "1":
        print("MASK_AUDIT_ONLY=1 -> stopping after mask audit.")
        return

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    if "family_counts_json" in chosen_df.columns:
        print(chosen_df[["family_counts_json"]].to_string(index=False))

    only_seed = os.environ.get("ONLY_SEED")
    if only_seed is not None:
        active_seeds = [int(only_seed)]
        experiment_log_csv = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{only_seed}.csv")
        print("working only with seed:", active_seeds)
    else:
        active_seeds = SEEDS
        experiment_log_csv = EXPERIMENT_LOG_CSV
        print("working with all seed:", active_seeds)

    debug = os.environ.get("FEDALA_DEBUG", "0") == "1"
    print("FEDALA_DEBUG:", debug)
    print("FEDALA_MODE:", FEDALA_MODE)

    sweep = list(iter_sweep_cfgs(
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
    ))

    print("Number of sweep configs:", len(sweep))
    print("Number of runs:", len(sweep) * len(chosen_df))
    progress = ProgressPrinter(len(sweep) * len(chosen_df))

    for cfg, meta in sweep:
        seed = int(meta["seed"])
        for _, row in chosen_df.iterrows():
            subset_ids = parse_subset_clients(row["subset_clients"])
            subset_clients = [id_to_client[graph_id] for graph_id in subset_ids]
            run_start = time.perf_counter()

            print()
            print(f"Starting with {len(subset_clients)}-client FedALA-HeadOnly training", row["subset_clients"])
            print(meta)
            print("SEED:", seed, "ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)

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
                selection_metrics=SELECTION_METRICS,
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
                selection_metrics=SELECTION_METRICS,
            )
            add_subset_metadata(log_row, row)
            upsert_experiment_rows(experiment_log_csv, [log_row])

            progress.step(label=f"FedALA-HeadOnly done -> {run_paths.csv_path}", run_start_time=run_start)


if __name__ == "__main__":
    main()
