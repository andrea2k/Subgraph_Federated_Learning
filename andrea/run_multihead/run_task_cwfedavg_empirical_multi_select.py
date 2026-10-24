from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.helper_funcs_multihead.benchmark_config import resolve_benchmark_paths
from andrea.helper_funcs_multihead.fl_run_helper_multi_select import (
    build_fedavg_log_row,
    iter_sweep_cfgs,
    upsert_experiment_rows,
)
from andrea.helper_funcs_multihead.load_client_helper import (
    ProgressPrinter,
    audit_loaded_q_label_masks,
    load_clients,
    parse_subset_clients,
)
from andrea.helper_funcs_multihead.task_cwfedavg_empirical_run_helper_multi_select import (
    CLASSWISE_AGGREGATION,
    DISTRIBUTION_SOURCE,
    PERSONALIZATION,
    WDR_MODE,
    run_task_cwfedavg_empirical,
)


BENCHMARK = resolve_benchmark_paths()
SELECT_SUBSET_PATH = BENCHMARK.select_subset_path
SELECT_SUBSET = BENCHMARK.select_subset

BASE_EXPERIMENT_LOG_FOLDER = "task_cwfedavg_empirical_experiment_multi_select"
RUN_TAG = os.environ.get("RUN_TAG", "prauc_20260824").strip()
EXPERIMENT_LOG_FOLDER = (
    f"{BASE_EXPERIMENT_LOG_FOLDER}_{RUN_TAG}"
    if RUN_TAG
    else BASE_EXPERIMENT_LOG_FOLDER
)
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = os.environ.get(
    "RUNS_ROOT",
    f"andrea/final_runs_prauc_20260824/{BENCHMARK.setup}/task_cwfedavg_empirical",
)

ROUNDS = int(os.environ.get("ROUNDS", 80))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_FRACTION = 1.0
SELECTION_METRICS = ["macro_pr_auc"]
RESULT_RUN_TYPE = "task_cwfedavg_empirical_multi_select"
RESULT_ALGORITHM = "task_cwfedavg_empirical"
DISPLAY_NAME = "Task-cwFedAvg (Empirical, no WDR)"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def relabel_result_csv(csv_path, extra: Dict | None = None) -> None:
    df = pd.read_csv(csv_path)
    df["run_type"] = RESULT_RUN_TYPE
    df["algorithm"] = RESULT_ALGORITHM
    if extra:
        for key, value in extra.items():
            df[key] = value
    df.to_csv(csv_path, index=False)


def main() -> None:
    print("BENCHMARK_SETUP:", BENCHMARK.setup)
    print("SELECT_SUBSET_PATH:", SELECT_SUBSET_PATH)
    print("SELECT_SUBSET:", SELECT_SUBSET)
    print(
        "RUN VARIANT: Task-cwFedAvg (Empirical, no WDR) | "
        "task-conditioned full global models + receiver-specific full-model recombination"
    )

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)
    only_q = os.environ.get("ONLY_Q")
    if only_q is not None:
        if "q_value" not in chosen_df.columns:
            raise ValueError("ONLY_Q was set, but selected_subset.csv has no q_value")
        chosen_df = chosen_df[
            pd.to_numeric(chosen_df["q_value"], errors="coerce").eq(float(only_q))
        ].copy()
        print("ONLY_Q filter:", only_q, "remaining subsets:", len(chosen_df))

    max_subsets = int(os.environ.get("MAX_SUBSETS", "0"))
    if max_subsets > 0:
        chosen_df = chosen_df.head(max_subsets).copy()
        print("MAX_SUBSETS filter:", max_subsets, "remaining subsets:", len(chosen_df))
    if chosen_df.empty:
        raise ValueError("No selected subsets remain after filtering")

    print("RUNS_ROOT:", RUNS_ROOT)
    print("EXPERIMENT_LOG_FOLDER:", EXPERIMENT_LOG_FOLDER)
    print("ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
    print("CHECKPOINT SELECTOR: macro_pr_auc (true trapezoidal PR-AUC)")
    print("DISTRIBUTION SOURCE:", DISTRIBUTION_SOURCE)
    print("WDR MODE:", WDR_MODE)

    id_to_client = load_clients(
        chosen_df,
        csv_path=ALL_DATA_LOGS,
        verbose=True,
        mask_splits=MASK_SPLITS,
    )
    if os.environ.get("SKIP_Q_MASK_AUDIT", "0") == "1":
        print("SKIP_Q_MASK_AUDIT=1 -> skipping q-mask audit")
    else:
        audit_loaded_q_label_masks(
            chosen_df,
            id_to_client,
            strict=True,
            mask_splits=MASK_SPLITS,
        )
    if os.environ.get("MASK_AUDIT_ONLY") == "1":
        print("MASK_AUDIT_ONLY=1 -> stopping after mask audit")
        return

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    only_seed = os.environ.get("ONLY_SEED")
    if only_seed is not None:
        seeds = [int(only_seed)]
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{only_seed}.csv"
        )
    else:
        seeds = list(SEEDS)
        experiment_log_csv = EXPERIMENT_LOG_CSV

    sweep = list(
        iter_sweep_cfgs(
            base_cfg,
            seeds=seeds,
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
    progress = ProgressPrinter(len(sweep) * len(chosen_df))

    for cfg, meta in sweep:
        seed = int(meta["seed"])
        for _, row in chosen_df.iterrows():
            subset_ids = parse_subset_clients(row["subset_clients"])
            subset_clients = [id_to_client[graph_id] for graph_id in subset_ids]
            run_start = time.perf_counter()

            print(
                f"Starting {len(subset_clients)}-client Task-cwFedAvg (Empirical)",
                row["subset_clients"],
            )
            print("SEED:", seed, "ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)

            paths = run_task_cwfedavg_empirical(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                device=DEVICE,
                selection_metrics=SELECTION_METRICS,
            )

            distribution_csv = paths.csv_path.with_name(
                paths.csv_path.stem + "_cwfedavg_task_distributions.csv"
            )
            common_meta = {
                "training_scope": "federated_masked_virtual_client_graphs",
                "communication": "one_full_model_up_and_one_personalized_full_model_down_per_client_per_round_plus_support_metadata",
                "baseline_role": "cwfedavg_style_empirical_task_personalized_full_model_aggregation",
                "mask_splits": "|".join(MASK_SPLITS),
                "selection_metric": "macro_pr_auc",
                "selection_metrics": "|".join(SELECTION_METRICS),
                "selection_direction": "max",
                "selection_modes": "full|visible",
                "pr_auc_definition": "trapezoidal_precision_recall_curve_auc",
                "eval_protocols": "oracle_full|realistic_visible|realistic_selection_oracle",
                "distribution_source": DISTRIBUTION_SOURCE,
                "classwise_aggregation": CLASSWISE_AGGREGATION,
                "personalization": PERSONALIZATION,
                "wdr_mode": WDR_MODE,
                "task_distributions_csv": str(distribution_csv),
                "cwfedavg_relation": "eq6_eq7_style_multitask_empirical_distribution_adaptation_no_wdr",
            }
            relabel_result_csv(paths.csv_path, common_meta)

            log_row = build_fedavg_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                selection_metric="macro_pr_auc",
            )
            add_subset_metadata(log_row, row)
            log_row.update(common_meta)
            log_row["run_type"] = RESULT_RUN_TYPE
            log_row["algorithm"] = RESULT_ALGORITHM
            log_row["display_name"] = DISPLAY_NAME

            upsert_experiment_rows(experiment_log_csv, [log_row])
            progress.step(
                label=f"task-cwfedavg empirical done -> {paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
