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
)
from andrea.helper_funcs.gcflplus_run_helper import (
    build_gcfl_log_row,
    run_gcfl,
)
from andrea.helper_funcs.load_client_helper import (
    ProgressPrinter,
    parse_subset_clients,
    load_clients,
)

SELECT_SUBSET_PATH = "clustering"
SELECT_SUBSET = "selected_subset"

EXPERIMENT_LOG_FOLDER = "gcflplus_clustering_experiment"
DATA_DIR = "clustering/cluster_generation_parameters.csv"

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
# GCFL+ setup
# -----------------------------------------------------------------------------
WARMUP_ROUNDS = 10
MIN_CLUSTER_SIZE = 4
MIN_CHILD_SIZE = 4

GRAD_SEQ_LEN = 10
EPS1_QUANTILE = 0.25
EPS2_QUANTILE = 0.75

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


def main():
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    id_to_client = load_clients(chosen_df, ALL_DATA_LOGS)
    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

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

    total_runs = len(sweep) * len(chosen_df)
    print("Number of runs:", total_runs)
    progress = ProgressPrinter(total_runs)

    for cfg, meta in sweep:
        seed = int(meta["seed"])

        for _, row in chosen_df.iterrows():
            subset_id = parse_subset_clients(row["subset_clients"])
            run_start = time.perf_counter()

            subset_clients = [id_to_client[graph_id] for graph_id in subset_id]

            print()
            print(
                f"Starting with {len(subset_clients)}-client gcfl+ training",
                row["subset_clients"],
            )
            print(meta)
            print(
                ROUNDS,
                LOCAL_EPOCHS,
                WARMUP_ROUNDS,
                MIN_CLUSTER_SIZE,
                MIN_CHILD_SIZE,
                GRAD_SEQ_LEN,
                EPS1_QUANTILE,
                EPS2_QUANTILE,
            )

            gcfl_paths = run_gcfl(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                warmup_rounds=WARMUP_ROUNDS,
                min_cluster_size=MIN_CLUSTER_SIZE,
                min_child_size=MIN_CHILD_SIZE,
                grad_seq_len=GRAD_SEQ_LEN,
                eps1_quantile=EPS1_QUANTILE,
                eps2_quantile=EPS2_QUANTILE,
                device=DEVICE,
                selection_metric=SELECTION_METRIC,
            )

            gcfl_row = build_gcfl_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=gcfl_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                warmup_rounds=WARMUP_ROUNDS,
                min_cluster_size=MIN_CLUSTER_SIZE,
                min_child_size=MIN_CHILD_SIZE,
                grad_seq_len=GRAD_SEQ_LEN,
                eps1_quantile=EPS1_QUANTILE,
                eps2_quantile=EPS2_QUANTILE,
                selection_metric=SELECTION_METRIC,
            )
            gcfl_row.update(
                {
                    "subset_clients": str(row["subset_clients"]),
                }
            )

            upsert_experiment_rows(EXPERIMENT_LOG_CSV, [gcfl_row])

            progress.step(
                label=f"gcfl+ done -> {gcfl_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
