from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.helper_funcs_multihead.apple_fedavg_backbone_task_head_run_helper import (
    build_apple_log_row,
    run_apple,
)
from andrea.helper_funcs_multihead.fl_run_helper import (
    iter_sweep_cfgs,
    upsert_experiment_rows,
)
from andrea.helper_funcs_multihead.load_client_helper import (
    ProgressPrinter,
    load_clients,
    parse_subset_clients,
    audit_loaded_q_label_masks,
)


SELECT_SUBSET_PATH = "clustering_q_label_heterogeneity"
SELECT_SUBSET = "selected_subset"

EXPERIMENT_LOG_FOLDER = "apple_fedavg_backbone_taskhead_clustering_experiment"
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = "andrea/runs"

# -----------------------------------------------------------------------------
# Training setup.
#
# Initial APPLE setting, kept here for reference:
#   ROUNDS = 80
#   LOCAL_EPOCHS = 1
#   CLIENT_FRACTION = 1.0
#   SELECTION_METRIC = "loss"
#
# New APPLE diagnostic setting:
#   run longer because APPLE after-local curves were still improving at round 80.
# -----------------------------------------------------------------------------
ROUNDS = 80
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1.0
SELECTION_METRIC = "loss"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# APPLE setup.
#
# Initial APPLE setting, kept here for reference:
#   APPLE_DR_LR = 1e-3
#   APPLE_MU = 1e-3
#   APPLE_SCHEDULER_TYPE = "cosine"
#   APPLE_SCHEDULER_FRACTION = 0.2
#   APPLE_DR_INIT = "sample_size"
#   APPLE_DR_CONSTRAINT = "unconstrained"
#   APPLE_DOWNLOAD_STRATEGY = "full"
#
# Current fedavg-backbone+task-head APPLE setting.
# -----------------------------------------------------------------------------
APPLE_DR_LR = 1e-3
APPLE_MU = 1e-3
APPLE_SCHEDULER_TYPE = "cosine"
APPLE_SCHEDULER_FRACTION = 0.2
APPLE_DR_INIT = "sample_size"
APPLE_DR_CONSTRAINT = "unconstrained"
APPLE_DOWNLOAD_STRATEGY = "full"

# -----------------------------------------------------------------------------
# Sweep setup: same model/config grid as the current baseline scripts.
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
    Mirrors the current Local/FedAvg/FedProx/GCFL+ runner behavior.
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


def _safe_json_loads(value, default):
    if value is None or pd.isna(value):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def build_client_metadata(row: pd.Series, subset_clients) -> Dict[str, Dict]:
    """
    Build optional graph-level metadata for APPLE DR diagnostics.

    The main training pipeline does not depend on this metadata. It is only used
    to make the DR matrix easier to interpret later, e.g. by family/task.
    """
    graph_to_family = _safe_json_loads(row.get("graph_to_family_json", None), {})
    graph_to_task = _safe_json_loads(row.get("graph_to_task_json", None), {})

    out: Dict[str, Dict] = {}
    for client in subset_clients:
        gid = str(client.graph_id)
        mask_meta = client.mask_meta or {}
        out[gid] = {
            "family": graph_to_family.get(gid),
            "assigned_task": graph_to_task.get(gid, mask_meta.get("assigned_task")),
            "mask_task": mask_meta.get("mask_task"),
            "mask_fraction": mask_meta.get("mask_fraction"),
            "q_value": mask_meta.get("q_value"),
            "q_other_share": mask_meta.get("q_other_share"),
            "q_assigned_share": mask_meta.get("q_assigned_share"),
            "q_allocation_mode": mask_meta.get("q_allocation_mode"),
        }
    return out


def main():
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    id_to_client = load_clients(chosen_df, ALL_DATA_LOGS)

    audit_loaded_q_label_masks(chosen_df, id_to_client, strict=True)

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    # Backbone+Task-head APPLE-specific architecture/mixing flags.
    # These are carried into iter_sweep_cfgs because it copies base_cfg.
    base_cfg["output_head"] = "multi"
    base_cfg["apple_mixing_mode"] = "fedavg_backbone_task_head"

    print("\nFEDAVG-BACKBONE + TASK-HEAD APPLE CONFIG")
    print("  output_head:", base_cfg["output_head"])
    print("  apple_mixing_mode:", base_cfg["apple_mixing_mode"])
    print("  runs_root:", RUNS_ROOT)
    print("  experiment_log_folder:", EXPERIMENT_LOG_FOLDER)

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
    if "family_counts_json" in chosen_df.columns:
        chosen_df["family_counts"] = chosen_df["family_counts_json"].apply(
            lambda s: ", ".join(f"{k}: {v}" for k, v in json.loads(s).items())
        )
    print(chosen_df[cols].to_string(index=False))

    only_seed = os.environ.get("ONLY_SEED")
    if only_seed is not None:
        seeds = [int(only_seed)]
        print("working only with seed:", seeds)
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{only_seed}.csv"
        )
    else:
        seeds = [0, 1, 2]
        print("working with all seed:", seeds)
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv"
        )

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

    total_runs = len(sweep) * len(chosen_df)
    print("Number of runs:", total_runs)
    progress = ProgressPrinter(total_runs)

    for cfg, meta in sweep:
        seed = int(meta["seed"])

        for _, row in chosen_df.iterrows():
            subset_id = parse_subset_clients(row["subset_clients"])
            run_start = time.perf_counter()

            subset_clients = [id_to_client[graph_id] for graph_id in subset_id]
            client_metadata = build_client_metadata(row, subset_clients)

            print()
            print(
                f"Starting with {len(subset_clients)}-client APPLE fedavg-backbone+task-head training",
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
                "APPLE_DR_LR:",
                APPLE_DR_LR,
                "APPLE_MU:",
                APPLE_MU,
                "SCHEDULER:",
                APPLE_SCHEDULER_TYPE,
                "SCHEDULER_FRACTION:",
                APPLE_SCHEDULER_FRACTION,
                "OUTPUT_HEAD:",
                cfg.get("output_head"),
                "APPLE_MIXING_MODE:",
                cfg.get("apple_mixing_mode"),
            )

            for client in subset_clients:
                print(
                    "client:",
                    client.graph_id,
                    "dataset:",
                    client.dataset_id,
                    "mask_meta:",
                    client.mask_meta,
                    "dr_metadata:",
                    client_metadata.get(str(client.graph_id), {}),
                )

            apple_paths = run_apple(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                dr_lr=APPLE_DR_LR,
                apple_mu=APPLE_MU,
                scheduler_type=APPLE_SCHEDULER_TYPE,
                scheduler_fraction=APPLE_SCHEDULER_FRACTION,
                dr_init=APPLE_DR_INIT,
                dr_constraint=APPLE_DR_CONSTRAINT,
                download_strategy=APPLE_DOWNLOAD_STRATEGY,
                device=DEVICE,
                selection_metric=SELECTION_METRIC,
                client_metadata=client_metadata,
            )

            apple_row = build_apple_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=apple_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                dr_lr=APPLE_DR_LR,
                apple_mu=APPLE_MU,
                scheduler_type=APPLE_SCHEDULER_TYPE,
                scheduler_fraction=APPLE_SCHEDULER_FRACTION,
                dr_init=APPLE_DR_INIT,
                dr_constraint=APPLE_DR_CONSTRAINT,
                download_strategy=APPLE_DOWNLOAD_STRATEGY,
                selection_metric=SELECTION_METRIC,
            )

            add_subset_metadata(apple_row, row)
            upsert_experiment_rows(experiment_log_csv, [apple_row])

            progress.step(
                label=f"apple done -> {apple_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
