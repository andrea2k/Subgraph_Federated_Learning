from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.helper_funcs_multihead.benchmark_config import resolve_benchmark_paths
from andrea.helper_funcs_multihead.apple_post_ala_run_helper_multi_select import build_apple_log_row
from andrea.helper_funcs_multihead.task_specialist_pool_run_helper_multi_select import (
    run_task_specialist_pool,
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

BENCHMARK = resolve_benchmark_paths()
SELECT_SUBSET_PATH = BENCHMARK.select_subset_path
SELECT_SUBSET = BENCHMARK.select_subset

RUN_TAG = os.environ.get("RUN_TAG", "task_specialist_pool_support_simplex_v1")
BASE_EXPERIMENT_LOG_FOLDER = "task_specialist_pool_apple_post_ala_experiment_multi_select"
EXPERIMENT_LOG_FOLDER = (
    f"{BASE_EXPERIMENT_LOG_FOLDER}_{RUN_TAG}" if RUN_TAG else BASE_EXPERIMENT_LOG_FOLDER
)
SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
ALL_DATA_LOGS = f"./andrea/{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = os.environ.get(
    "RUNS_ROOT", "andrea/runs_task_specialist_pool_apple_post_ala_support_simplex"
)

ROUNDS = int(os.environ.get("ROUNDS", 80))
WARMUP_ROUNDS = int(os.environ.get("SPECIALIST_WARMUP_ROUNDS", 20))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", 1.0))
SPECIALIST_SELECTION_VIEW = "fixed_pool"
SELECTION_METRICS = ["loss", "micro_f1", "macro_pos_f1", "micro_pr_auc", "macro_pr_auc"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

APPLE_DR_LR = float(os.environ.get("APPLE_DR_LR", 1e-3))
APPLE_SUPPORT_PSEUDO_COUNT = float(os.environ.get("APPLE_SUPPORT_PSEUDO_COUNT", 1.0))
APPLE_USE_HEAD_ALA = os.environ.get("APPLE_USE_HEAD_ALA", "True") == "True"
APPLE_ALA_LR = float(os.environ.get("APPLE_ALA_LR", 1.0))
APPLE_ALA_RAND_PERCENT = float(os.environ.get("APPLE_ALA_RAND_PERCENT", 100.0))
APPLE_ALA_CONVERGENCE_STD = float(os.environ.get("APPLE_ALA_CONVERGENCE_STD", 0.1))
APPLE_ALA_CONVERGENCE_WINDOW = int(os.environ.get("APPLE_ALA_CONVERGENCE_WINDOW", 10))
APPLE_ALA_MAX_STEPS = int(os.environ.get("APPLE_ALA_MAX_STEPS", 100))
APPLE_ALA_DEBUG = os.environ.get("APPLE_ALA_DEBUG", "0") == "1"

SEEDS = [0, 1, 2]
NUM_LAYERS = [6]
LRS = [0.001]
WEIGHT_DECAYS = [0.0001]
DROPOUTS = [0.1]
HIDDEN_DIMS = [64]
USE_EGO_IDS = [True]
BATCH_SIZE = [64]
MASK_SPLITS = ("train", "val", "test")

RESULT_RUN_TYPE = "task_specialist_pool_apple_post_ala_support_simplex_multi_select"
RESULT_ALGORITHM = "task_specialist_pool_apple_post_ala_support_simplex_multi_select"
DISPLAY_NAME = "TaskSpecialistPool-APPLE-PostALA Support-Simplex"


def parse_mcw_values_from_env():
    raw = os.environ.get("MCW_VALUES", "auto").strip() or "auto"
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        low = token.lower()
        if low == "auto":
            values.append("auto")
        elif low in {"none", "null", "off", "unweighted"}:
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError(f"MCW_VALUES produced no settings: {raw!r}")
    return values


MCW = parse_mcw_values_from_env()


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


def _safe_json(value, default):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return json.loads(str(value))
    except Exception:
        return default


def build_client_metadata(row: pd.Series, subset_clients) -> Dict[str, Dict]:
    graph_to_task = _safe_json(row.get("graph_to_task_json"), {})
    graph_to_community = _safe_json(row.get("graph_to_community_json"), {})
    membership = _safe_json(row.get("membership_json"), [])
    membership_by_gid = {str(item.get("graph_id")): item for item in membership}
    out = {}
    for client in subset_clients:
        gid = str(client.graph_id)
        mask_meta = client.mask_meta or {}
        member = membership_by_gid.get(gid, {})
        out[gid] = {
            "assigned_task": graph_to_task.get(gid, mask_meta.get("assigned_task")),
            "physical_community": graph_to_community.get(
                gid, member.get("planted_community_id")
            ),
            "q_value": mask_meta.get("q_value"),
        }
    return out


def add_subset_metadata(log_row: Dict, row: pd.Series) -> None:
    log_row["subset_clients"] = str(row["subset_clients"])
    for col in [
        "q_value",
        "q_iid",
        "q_other_share",
        "q_assigned_share",
        "q_allocation_mode",
        "mask_fraction",
        "controlled_benchmark",
        "mask_mode",
        "subset_id",
    ]:
        if col in row.index:
            log_row[f"manifest_{col}" if col == "subset_id" else col] = row.get(col)


def main() -> None:
    print("BENCHMARK_SETUP:", BENCHMARK.setup)
    if BENCHMARK.setup != "twenty_client":
        raise ValueError(
            "TaskSpecialistPool is 20-client-only. "
            f"Got BENCHMARK_SETUP={BENCHMARK.setup!r}."
        )
    os.environ["APPLE_EXPERIMENT_ALGORITHM"] = RESULT_ALGORITHM
    print("RUN VARIANT:", DISPLAY_NAME)
    print("ROUNDS:", ROUNDS, "WARMUP_ROUNDS:", WARMUP_ROUNDS)
    print("SPECIALIST_SELECTION_VIEW:", SPECIALIST_SELECTION_VIEW)
    print("RUNS_ROOT:", RUNS_ROOT)

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)
    only_q = os.environ.get("ONLY_Q")
    if only_q is not None:
        q = float(only_q)
        chosen_df = chosen_df[
            pd.to_numeric(chosen_df["q_value"], errors="coerce")
            .round(10)
            .eq(round(q, 10))
        ].copy()
        print("ONLY_Q filter:", only_q, "remaining:", len(chosen_df))
    max_subsets = os.environ.get("MAX_SUBSETS")
    if max_subsets is not None:
        chosen_df = chosen_df.head(int(max_subsets)).copy()
    if chosen_df.empty:
        raise ValueError("No 20-client subsets remain")

    id_to_client = load_clients(
        chosen_df, csv_path=ALL_DATA_LOGS, verbose=True, mask_splits=MASK_SPLITS
    )
    if os.environ.get("SKIP_Q_MASK_AUDIT", "0") != "1":
        audit_loaded_q_label_masks(
            chosen_df, id_to_client, strict=True, mask_splits=MASK_SPLITS
        )

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    base_cfg["output_head"] = "multi"
    base_cfg["apple_mixing_mode"] = "all20_fedavg_backbone_task_specialist_pool_4_per_task"
    base_cfg["selection_tag"] = (
        f"task_specialist_pool_4per_task_w{WARMUP_ROUNDS}"
    )
    base_cfg["result_algorithm"] = RESULT_ALGORITHM
    base_cfg["apple_use_head_ala"] = APPLE_USE_HEAD_ALA
    base_cfg["apple_ala_lr"] = APPLE_ALA_LR
    base_cfg["apple_ala_rand_percent"] = APPLE_ALA_RAND_PERCENT
    base_cfg["apple_ala_convergence_std"] = APPLE_ALA_CONVERGENCE_STD
    base_cfg["apple_ala_convergence_window"] = APPLE_ALA_CONVERGENCE_WINDOW
    base_cfg["apple_ala_max_steps"] = APPLE_ALA_MAX_STEPS
    base_cfg["apple_ala_debug"] = APPLE_ALA_DEBUG
    base_cfg["apple_support_pseudo_count"] = APPLE_SUPPORT_PSEUDO_COUNT
    base_cfg["apple_routing_mode"] = "task_specialist_pool_support_simplex"

    only_seed = os.environ.get("ONLY_SEED")
    if only_seed is not None:
        active_seeds = [int(only_seed)]
        experiment_log_csv = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{only_seed}.csv"
        )
    else:
        active_seeds = SEEDS
        experiment_log_csv = EXPERIMENT_LOG_CSV

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
    progress = ProgressPrinter(len(sweep) * len(chosen_df))

    for cfg, meta in sweep:
        seed = int(meta["seed"])
        for _, row in chosen_df.iterrows():
            subset_ids = parse_subset_clients(row["subset_clients"])
            subset_clients = [id_to_client[graph_id] for graph_id in subset_ids]
            if len(subset_clients) != 20:
                raise ValueError(f"Expected 20 clients, got {len(subset_clients)}")
            client_metadata = build_client_metadata(row, subset_clients)
            run_start = time.perf_counter()

            paths = run_task_specialist_pool(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                warmup_rounds=WARMUP_ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                dr_lr=APPLE_DR_LR,
                support_pseudo_count=APPLE_SUPPORT_PSEUDO_COUNT,
                specialist_selection_view=SPECIALIST_SELECTION_VIEW,
                device=DEVICE,
                selection_metrics=SELECTION_METRICS,
                client_metadata=client_metadata,
            )

            # Reuse the standard manifest fields, then add bank-specific fields.
            standard_paths = type("P", (), {
                "csv_path": paths.csv_path,
                "ckpt_path": paths.ckpt_path,
                "dr_csv_path": paths.dr_csv_path,
            })()
            log_row = build_apple_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=standard_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                dr_lr=APPLE_DR_LR,
                support_pseudo_count=APPLE_SUPPORT_PSEUDO_COUNT,
                selection_metrics=SELECTION_METRICS,
            )
            add_subset_metadata(log_row, row)
            log_row.update(
                {
                    "run_type": RESULT_RUN_TYPE,
                    "algorithm": RESULT_ALGORITHM,
                    "display_name": DISPLAY_NAME,
                    "specialist_warmup_rounds": WARMUP_ROUNDS,
                    "specialist_selection_view": "fixed_pool",
                    "specialist_pool_mode": "all_four_same_task_specialists",
                    "specialist_num_donors_per_task": 4,
                    "specialist_unique_donor_clients": 20,
                    "specialist_pool_map_csv": str(paths.representatives_csv_path),
                    "backbone_aggregation_clients": 20,
                    "task_head_donor_clients_per_task": 4,
                    "dr_storage_shape": "20x5x20",
                    "effective_nonzero_donors_per_task": 4,
                }
            )
            upsert_experiment_rows(experiment_log_csv, [log_row])
            progress.step(
                label=f"TaskSpecialistPool done -> {paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
