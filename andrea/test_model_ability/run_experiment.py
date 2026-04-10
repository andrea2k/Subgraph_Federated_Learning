from __future__ import annotations

import time
from pathlib import Path

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
)

SELECTED_SUBSETS_CSV_PATH = "./andrea/test_model_ability/graphs_profile_selected.csv"
ALL_DATA_LOGS = "./andrea/test_generation_parameters.csv"
EXPERIMENT_LOG_CSV = Path("./andrea/test_model_ability/experiment_medium_graph_log.csv")
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = "andrea/test_model_ability/medium_graph_runs"

# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------
ROUNDS = 50
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


def main():
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    id_to_client = load_clients(chosen_df)
    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

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
            graph_id = row["graph_id"]

            run_start = time.perf_counter()

            client = id_to_client[graph_id]

            run_paths = run_local(
                client,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                device=DEVICE,
                selection_metric=SELECTION_METRIC,
            )
            local_log_row = build_local_log_row(
                client=client,
                cfg=cfg,
                seed=seed,
                run_paths=run_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                selection_metric=SELECTION_METRIC,
            )
            upsert_experiment_rows(EXPERIMENT_LOG_CSV, [local_log_row])

            progress.step(
                label=f"fedavg done -> {run_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
