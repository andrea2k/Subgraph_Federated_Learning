from __future__ import annotations

import time
from pathlib import Path

import os
import pandas as pd
import torch
import json
from typing import Dict

from andrea.helper_funcs_multihead.benchmark_config import resolve_benchmark_paths
from andrea.helper_funcs_multihead.fl_run_helper_multi_select import (
    iter_sweep_cfgs,
    upsert_experiment_rows,
    build_fedprox_log_row,
    run_fedprox,
)
from andrea.helper_funcs_multihead.load_client_helper import (
    ProgressPrinter,
    parse_subset_clients,
    load_clients,
    audit_loaded_q_label_masks,
)

BENCHMARK = resolve_benchmark_paths()
SELECT_SUBSET_PATH = BENCHMARK.select_subset_path
SELECT_SUBSET = BENCHMARK.select_subset

BASE_EXPERIMENT_LOG_FOLDER = "fedprox_multihead_clustering_experiment_multi_select"
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
EXPERIMENT_LOG_FOLDER = (
    f"{BASE_EXPERIMENT_LOG_FOLDER}_{RUN_TAG}" if RUN_TAG else BASE_EXPERIMENT_LOG_FOLDER
)
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = os.environ.get("RUNS_ROOT", "andrea/runs")

# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------
ROUNDS = int(os.environ.get("ROUNDS", 80))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_FRACTION = 1.0
FEDPROX_MU = float(os.environ.get("FEDPROX_MU", 0.01))
SELECTION_METRICS = ["loss", "micro_f1", "macro_pos_f1", "micro_pr_auc", "macro_pr_auc"]
RESULT_RUN_TYPE = "fedprox_multi_select"
RESULT_ALGORITHM = "fedprox_multi_select"
DISPLAY_NAME = "FedProx (multi-selection)"
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
            log_row[f"manifest_{col}" if col == "subset_id" else col] = row.get(
                col, None
            )


def relabel_result_csv(
    csv_path, *, run_type: str, algorithm: str, extra: Dict | None = None
) -> None:
    df = pd.read_csv(csv_path)
    df["run_type"] = run_type
    df["algorithm"] = algorithm
    if extra:
        for key, value in extra.items():
            df[key] = value
    df.to_csv(csv_path, index=False)


def main():
    print("BENCHMARK_SETUP:", BENCHMARK.setup)
    print("SELECT_SUBSET_PATH:", SELECT_SUBSET_PATH)
    print("SELECT_SUBSET:", SELECT_SUBSET)
    print(
        "RUN VARIANT: FedProx (multi-selection) | selection_metrics=loss|micro_f1|macro_pos_f1|micro_pr_auc|macro_pr_auc"
    )

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    only_q = os.environ.get("ONLY_Q")
    if only_q is not None:
        if "q_value" not in chosen_df.columns:
            raise ValueError(
                "ONLY_Q was set, but selected_subset.csv has no q_value column."
            )
        chosen_df = chosen_df[
            pd.to_numeric(chosen_df["q_value"], errors="coerce").eq(float(only_q))
        ].copy()
        print("ONLY_Q filter:", only_q, "remaining subsets:", len(chosen_df))

    max_subsets = int(os.environ.get("MAX_SUBSETS", "0"))
    if max_subsets > 0:
        chosen_df = chosen_df.head(max_subsets).copy()
        print("MAX_SUBSETS filter:", max_subsets, "remaining subsets:", len(chosen_df))

    if chosen_df.empty:
        raise ValueError(
            "No selected subsets remain after ONLY_Q/MAX_SUBSETS filtering."
        )

    print("RUNS_ROOT:", RUNS_ROOT)
    print("EXPERIMENT_LOG_FOLDER:", EXPERIMENT_LOG_FOLDER)
    print("ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS, "FEDPROX_MU:", FEDPROX_MU)

    id_to_client = load_clients(
        chosen_df,
        csv_path=ALL_DATA_LOGS,
        verbose=True,
        mask_splits=MASK_SPLITS,
    )

    if os.environ.get("SKIP_Q_MASK_AUDIT", "0") == "1":
        print("SKIP_Q_MASK_AUDIT=1 -> skipping q-mask audit.")
    else:
        audit_loaded_q_label_masks(
            chosen_df,
            id_to_client,
            strict=True,
            mask_splits=MASK_SPLITS,
        )

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

    cols = ["family_counts_json"]
    cols = [col for col in cols if col in chosen_df.columns]
    chosen_df["family_counts"] = chosen_df["family_counts_json"].apply(
        lambda s: ", ".join(f"{k}: {v}" for k, v in json.loads(s).items())
    )
    print(chosen_df[cols].to_string(index=False))

    global SEEDS, EXPERIMENT_LOG_CSV
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

    print("\nMULTI-HEAD CONFIG SANITY CHECK")
    print("Number of sweep configs:", len(sweep))
    for debug_cfg, debug_meta in sweep[:1]:
        print("debug meta:", debug_meta)
        print("cfg output_head :", debug_cfg.get("output_head"))
        print("cfg architecture:", debug_cfg.get("architecture"))

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
                f"Starting with {len(subset_clients)}-client fedprox training",
                row["subset_clients"],
            )
            print(meta)
            print(
                "SEED:",
                seed,
                "ROUNDS:",
                ROUNDS,
                "LOCAL_EPOCHS:",
                LOCAL_EPOCHS,
                "MU:",
                FEDPROX_MU,
            )

            for client in subset_clients:
                print(
                    "client:",
                    client.graph_id,
                    "dataset:",
                    client.dataset_id,
                    "mask_meta:",
                    client.mask_meta,
                )

            fed_paths = run_fedprox(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                fedprox_mu=FEDPROX_MU,
                device=DEVICE,
                selection_metrics=SELECTION_METRICS,
            )

            relabel_result_csv(
                fed_paths.csv_path,
                run_type=RESULT_RUN_TYPE,
                algorithm=RESULT_ALGORITHM,
                extra={
                    "training_scope": "federated_masked_virtual_client_graphs",
                    "communication": "fedprox",
                    "baseline_role": "fedprox_masked_clients",
                    "fedprox_mu": FEDPROX_MU,
                    "mask_splits": "|".join(MASK_SPLITS),
                    "selection_metric": "multi",
                    "selection_metrics": "|".join(SELECTION_METRICS),
                    "selection_direction": "mixed",
                    "selection_modes": "full|visible",
                    "eval_protocols": "oracle_full|realistic_visible|realistic_selection_oracle",
                },
            )

            fed_row = build_fedprox_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=fed_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                fedprox_mu=FEDPROX_MU,
                selection_metric="multi",
            )
            add_subset_metadata(fed_row, row)

            fed_row["run_type"] = RESULT_RUN_TYPE
            fed_row["algorithm"] = RESULT_ALGORITHM
            fed_row["display_name"] = DISPLAY_NAME
            fed_row["selection_metric"] = "multi"
            fed_row["selection_metrics"] = "|".join(SELECTION_METRICS)
            fed_row["selection_direction"] = "mixed"
            fed_row["training_scope"] = "federated_masked_virtual_client_graphs"
            fed_row["communication"] = "fedprox"
            fed_row["baseline_role"] = "fedprox_masked_clients"
            fed_row["mask_splits"] = "|".join(MASK_SPLITS)
            fed_row["selection_modes"] = "full|visible"
            fed_row["eval_protocols"] = (
                "oracle_full|realistic_visible|realistic_selection_oracle"
            )

            upsert_experiment_rows(EXPERIMENT_LOG_CSV, [fed_row])

            progress.step(
                label=f"fedprox done -> {fed_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
