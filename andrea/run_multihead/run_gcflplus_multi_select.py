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
)
from andrea.helper_funcs_multihead.gcflplus_run_helper_multi_select import (
    build_gcfl_log_row,
    run_gcfl,
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

BASE_EXPERIMENT_LOG_FOLDER = "gcflplus_multihead_clustering_experiment_multi_select"
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
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", 1.0))
SELECTION_METRICS = ["loss", "micro_f1", "macro_pos_f1", "micro_pr_auc", "macro_pr_auc"]
RESULT_RUN_TYPE = "gcfl_plus_multi_select"
RESULT_ALGORITHM = "gcfl_plus_multi_select"
DISPLAY_NAME = "GCFL+ (multi-selection)"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# GCFL+ setup
# -----------------------------------------------------------------------------
WARMUP_ROUNDS = int(os.environ.get("WARMUP_ROUNDS", 5))
MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", 2))
MIN_CHILD_SIZE = int(os.environ.get("MIN_CHILD_SIZE", 1))
GRAD_SEQ_LEN = int(os.environ.get("GRAD_SEQ_LEN", 5))
EPS1_QUANTILE = float(os.environ.get("EPS1_QUANTILE", 0.25))
EPS2_QUANTILE = float(os.environ.get("EPS2_QUANTILE", 0.7))

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
    """Add old-compatible and q-controlled metadata to an experiment-log row."""
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
    csv_path,
    *,
    run_type: str,
    algorithm: str,
    extra: Dict | None = None,
) -> None:
    """Patch helper-written result CSV rows to GCFL+ multi-select naming."""
    df = pd.read_csv(csv_path)
    df["run_type"] = run_type
    df["algorithm"] = algorithm
    if extra:
        for key, value in extra.items():
            df[key] = value
    df.to_csv(csv_path, index=False)


def relabel_cluster_csv(
    csv_path,
    *,
    run_type: str,
    algorithm: str,
    extra: Dict | None = None,
) -> None:
    """Patch GCFL+ cluster-diagnostic CSV if it contains rows."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return
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
        "RUN VARIANT: GCFL+ (multi-selection) | selection_metrics=loss|micro_f1|macro_pos_f1|micro_pr_auc|macro_pr_auc"
    )

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    # Optional smoke-test filters. Full runs are unaffected when unset.
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
    print("ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
    print(
        "GCFL+:",
        "WARMUP_ROUNDS",
        WARMUP_ROUNDS,
        "MIN_CLUSTER_SIZE",
        MIN_CLUSTER_SIZE,
        "MIN_CHILD_SIZE",
        MIN_CHILD_SIZE,
        "GRAD_SEQ_LEN",
        GRAD_SEQ_LEN,
        "EPS1_QUANTILE",
        EPS1_QUANTILE,
        "EPS2_QUANTILE",
        EPS2_QUANTILE,
    )

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
        "q_value",
    ]
    cols = [col for col in cols if col in chosen_df.columns]
    print(chosen_df[cols])

    cols = ["family_counts_json"]
    cols = [col for col in cols if col in chosen_df.columns]
    if "family_counts_json" in chosen_df.columns:
        chosen_df["family_counts"] = chosen_df["family_counts_json"].apply(
            lambda s: ", ".join(f"{k}: {v}" for k, v in json.loads(s).items())
        )
        print(chosen_df[cols].to_string(index=False))

    ONLY_SEED = os.environ.get("ONLY_SEED")
    if ONLY_SEED is not None:
        SEEDS_USED = [int(ONLY_SEED)]
        print("working only with seed:", SEEDS_USED)
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{ONLY_SEED}.csv"
        )
    else:
        SEEDS_USED = [0, 1, 2]
        print("working with all seed:", SEEDS_USED)
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

    print("\nMULTI-HEAD CONFIG SANITY CHECK")
    print("Number of sweep configs:", len(sweep))
    for debug_cfg, debug_meta in sweep[:1]:
        print("debug meta:", debug_meta)
        print("cfg output_head :", debug_cfg.get("output_head"))
        print("cfg architecture:", debug_cfg.get("architecture"))
        print("cfg model_tag-ish fields:")
        print("  num_layers:", debug_cfg.get("num_layers"))
        print("  lr:", debug_cfg.get("lr"))
        print("  weight_decay:", debug_cfg.get("weight_decay"))
        print("  dropout:", debug_cfg.get("dropout"))
        print("  hidden_dim:", debug_cfg.get("hidden_dim"))
        print("  batch_size:", debug_cfg.get("batch_size"))

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
                selection_metrics=SELECTION_METRICS,
            )

            common_extra = {
                "training_scope": "federated_masked_virtual_client_graphs",
                "communication": "clustered_federated",
                "baseline_role": "gcflplus_masked_clients",
                "mask_splits": "|".join(MASK_SPLITS),
                "selection_metric": "multi",
                "selection_metrics": "|".join(SELECTION_METRICS),
                "selection_direction": "mixed",
                "selection_modes": "full|visible",
                "eval_protocols": "oracle_full|realistic_visible|realistic_selection_oracle",
            }

            relabel_result_csv(
                gcfl_paths.csv_path,
                run_type=RESULT_RUN_TYPE,
                algorithm=RESULT_ALGORITHM,
                extra=common_extra,
            )
            relabel_cluster_csv(
                gcfl_paths.clusters_csv_path,
                run_type=RESULT_RUN_TYPE,
                algorithm=RESULT_ALGORITHM,
                extra=common_extra,
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
                selection_metrics=SELECTION_METRICS,
            )

            add_subset_metadata(gcfl_row, row)
            gcfl_row["run_type"] = RESULT_RUN_TYPE
            gcfl_row["algorithm"] = RESULT_ALGORITHM
            gcfl_row["display_name"] = DISPLAY_NAME
            gcfl_row["selection_metric"] = "multi"
            gcfl_row["selection_metrics"] = "|".join(SELECTION_METRICS)
            gcfl_row["selection_direction"] = "mixed"
            gcfl_row["training_scope"] = "federated_masked_virtual_client_graphs"
            gcfl_row["communication"] = "clustered_federated"
            gcfl_row["baseline_role"] = "gcflplus_masked_clients"
            gcfl_row["mask_splits"] = "|".join(MASK_SPLITS)
            gcfl_row["selection_modes"] = "full|visible"
            gcfl_row["eval_protocols"] = (
                "oracle_full|realistic_visible|realistic_selection_oracle"
            )

            upsert_experiment_rows(experiment_log_csv, [gcfl_row])

            progress.step(
                label=f"gcfl+ done -> {gcfl_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
