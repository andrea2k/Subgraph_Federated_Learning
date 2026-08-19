from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from andrea.multigraph_generation import TASKS

from andrea.helper_funcs_multihead.benchmark_config import resolve_benchmark_paths
from andrea.helper_funcs_multihead.apple_post_ala_run_helper_multi_select import (
    build_apple_log_row,
    run_apple,
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

RUN_TAG = os.environ.get("RUN_TAG", "oracle_hard_support_simplex_v2")
_ONLY_ORACLE_CLUSTER_AT_IMPORT = os.environ.get("ONLY_ORACLE_CLUSTER")
if _ONLY_ORACLE_CLUSTER_AT_IMPORT is not None:
    _cluster_suffix = f"_c{int(_ONLY_ORACLE_CLUSTER_AT_IMPORT)}"
    if not RUN_TAG.endswith(_cluster_suffix):
        RUN_TAG = f"{RUN_TAG}{_cluster_suffix}"

BASE_EXPERIMENT_LOG_FOLDER = (
    "oracle_hard_cluster_apple_post_ala_clustering_experiment_multi_select"
)
EXPERIMENT_LOG_FOLDER = (
    f"{BASE_EXPERIMENT_LOG_FOLDER}_{RUN_TAG}" if RUN_TAG else BASE_EXPERIMENT_LOG_FOLDER
)
DATA_DIR = f"{SELECT_SUBSET_PATH}/cluster_generation_parameters.csv"

SELECTED_SUBSETS_CSV_PATH = f"./andrea/{SELECT_SUBSET_PATH}/{SELECT_SUBSET}.csv"
EXPERIMENT_LOG_CSV = Path(f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv")
ALL_DATA_LOGS = f"./andrea/{DATA_DIR}"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"
RUNS_ROOT = os.environ.get("RUNS_ROOT", "andrea/runs_oracle_hard_cluster_apple_post_ala_support_simplex")

ROUNDS = int(os.environ.get("ROUNDS", 80))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
CLIENT_FRACTION = float(os.environ.get("CLIENT_FRACTION", 1.0))
SELECTION_METRICS = ["loss", "micro_f1", "macro_pos_f1", "micro_pr_auc", "macro_pr_auc"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

APPLE_DR_LR = float(os.environ.get("APPLE_DR_LR", 1e-3))
APPLE_SUPPORT_PSEUDO_COUNT = float(os.environ.get("APPLE_SUPPORT_PSEUDO_COUNT", 1.0))

# PostAPPLE head-only ALA options.
APPLE_USE_HEAD_ALA = os.environ.get("APPLE_USE_HEAD_ALA", "True") == "True"
APPLE_ALA_LR = float(os.environ.get("APPLE_ALA_LR", 1.0))
APPLE_ALA_RAND_PERCENT = float(os.environ.get("APPLE_ALA_RAND_PERCENT", 100.0))
APPLE_ALA_CONVERGENCE_STD = float(os.environ.get("APPLE_ALA_CONVERGENCE_STD", 0.1))
APPLE_ALA_CONVERGENCE_WINDOW = int(os.environ.get("APPLE_ALA_CONVERGENCE_WINDOW", 10))
APPLE_ALA_MAX_STEPS = int(os.environ.get("APPLE_ALA_MAX_STEPS", 100))
APPLE_ALA_DEBUG = os.environ.get("APPLE_ALA_DEBUG", "0") == "1"

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

RESULT_RUN_TYPE = "oracle_hard_cluster_apple_post_ala_support_simplex_multi_select"
RESULT_ALGORITHM = "oracle_hard_cluster_apple_post_ala_support_simplex_multi_select"
DISPLAY_NAME = "OracleHardCluster-APPLE-PostALA Support-Simplex"


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


def _safe_json_loads(value, default):
    if value is None or pd.isna(value):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def build_client_metadata(row: pd.Series, subset_clients) -> Dict[str, Dict]:
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



def _load_json_value(value, default):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(str(value))
    except Exception as exc:
        raise ValueError(
            f"Could not parse JSON value: {value}"
        ) from exc


def _json_dump(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def expand_oracle_hard_clusters(
    parent_df: pd.DataFrame,
) -> pd.DataFrame:
    """Split every 20-client oracle row into four 5-client rows.

    Each generated row contains exactly one true physical community:
    one client for every task and one shared base graph. APPLE-PostALA
    therefore runs as a fully independent five-client federation.
    """
    rows = []

    for parent_row_idx, parent_row in parent_df.iterrows():
        parent_clients = parse_subset_clients(
            parent_row["subset_clients"]
        )

        if len(parent_clients) != 20:
            raise ValueError(
                f"Parent row {parent_row_idx}: expected 20 clients, "
                f"found {len(parent_clients)}: {parent_clients}"
            )

        membership = _load_json_value(
            parent_row.get("membership_json"),
            [],
        )

        if len(membership) != 20:
            raise ValueError(
                f"Parent row {parent_row_idx}: expected 20 "
                f"membership records, found {len(membership)}."
            )

        membership_by_community = {}

        for item in membership:
            if "planted_community_id" not in item:
                raise ValueError(
                    "membership_json has no planted_community_id."
                )

            community_id = int(
                item["planted_community_id"]
            )

            membership_by_community.setdefault(
                community_id,
                [],
            ).append(dict(item))

        if len(membership_by_community) != 4:
            raise ValueError(
                f"Parent row {parent_row_idx}: expected four "
                f"communities, found "
                f"{sorted(membership_by_community)}."
            )

        parent_subset_id = str(
            parent_row.get(
                "subset_id",
                f"parent_{parent_row_idx}",
            )
        )
        parent_subset_clients = str(
            parent_row["subset_clients"]
        )

        for community_id in sorted(
            membership_by_community
        ):
            members = sorted(
                membership_by_community[community_id],
                key=lambda item: int(item["graph_id"]),
            )

            if len(members) != 5:
                raise ValueError(
                    f"Community {community_id}: expected five "
                    f"clients, found {len(members)}."
                )

            member_ids = [
                int(item["graph_id"])
                for item in members
            ]
            member_id_set = set(member_ids)

            if not member_id_set.issubset(
                set(parent_clients)
            ):
                raise ValueError(
                    f"Community {community_id}: membership IDs "
                    f"{member_ids} are not contained in parent "
                    f"clients {parent_clients}."
                )

            assigned_tasks = [
                str(item["assigned_task"])
                for item in members
            ]

            if set(assigned_tasks) != set(TASKS):
                raise ValueError(
                    f"Community {community_id}: expected one "
                    f"client for every task {TASKS}, found "
                    f"{assigned_tasks}."
                )

            base_graph_ids = {
                int(item["base_graph_id"])
                for item in members
            }

            if len(base_graph_ids) != 1:
                raise ValueError(
                    f"Community {community_id}: expected one "
                    f"base graph, found {base_graph_ids}."
                )

            base_dataset_ids = sorted(
                {
                    str(item["base_dataset_id"])
                    for item in members
                    if item.get("base_dataset_id") is not None
                }
            )

            graph_to_task = {
                str(item["graph_id"]): str(
                    item["assigned_task"]
                )
                for item in members
            }
            graph_to_community = {
                str(item["graph_id"]): int(
                    item["planted_community_id"]
                )
                for item in members
            }
            graph_to_family = {
                str(item["graph_id"]): (
                    f"q_task_{item['assigned_task']}"
                )
                for item in members
            }

            family_to_graph_ids = {
                f"q_task_{item['assigned_task']}": [
                    int(item["graph_id"])
                ]
                for item in members
            }

            family_counts = {
                family: len(graph_ids)
                for family, graph_ids
                in family_to_graph_ids.items()
            }

            hard_row = parent_row.to_dict()

            hard_row["oracle_parent_subset_id"] = (
                parent_subset_id
            )
            hard_row["oracle_parent_subset_clients"] = (
                parent_subset_clients
            )
            hard_row["oracle_hard_cluster_id"] = int(
                community_id
            )
            hard_row["oracle_hard_base_graph_id"] = int(
                next(iter(base_graph_ids))
            )
            hard_row["oracle_hard_community_source"] = (
                "membership_json.planted_community_id"
            )

            hard_row["family"] = (
                "oracle_hard_cluster_q_task_label_heterogeneity"
            )
            hard_row["subset_id"] = (
                f"{parent_subset_id}"
                f"__oracle_hard_c{community_id}"
            )
            hard_row["subset_clients"] = "|".join(
                str(graph_id)
                for graph_id in member_ids
            )
            hard_row["subset_size"] = 5

            hard_row["num_structural_clusters"] = 1
            hard_row["clients_per_structural_cluster"] = 5

            hard_row["graph_ids_json"] = _json_dump(
                member_ids
            )
            hard_row["dataset_ids_json"] = _json_dump(
                [
                    str(item["dataset_id"])
                    for item in members
                ]
            )
            hard_row["base_graph_ids_json"] = _json_dump(
                sorted(base_graph_ids)
            )
            hard_row["base_dataset_ids_json"] = _json_dump(
                base_dataset_ids
            )
            hard_row["membership_json"] = _json_dump(
                members
            )
            hard_row["graph_to_task_json"] = _json_dump(
                graph_to_task
            )
            hard_row["graph_to_community_json"] = _json_dump(
                graph_to_community
            )
            hard_row["graph_to_family_json"] = _json_dump(
                graph_to_family
            )
            hard_row["community_to_graph_ids_json"] = (
                _json_dump(
                    {
                        str(community_id): member_ids,
                    }
                )
            )
            hard_row["family_to_graph_ids_json"] = (
                _json_dump(family_to_graph_ids)
            )
            hard_row["family_counts_json"] = _json_dump(
                family_counts
            )

            rows.append(hard_row)

    expanded = pd.DataFrame(rows)

    if expanded.empty:
        raise ValueError(
            "Oracle hard-cluster expansion produced no rows."
        )

    return expanded.reset_index(drop=True)

def main() -> None:
    print("BENCHMARK_SETUP:", BENCHMARK.setup)
    if BENCHMARK.setup != "twenty_client":
        raise ValueError(
            "OracleHardCluster-APPLE-PostALA is defined only for the 20-client planted-community benchmark. "
            f"Got BENCHMARK_SETUP={BENCHMARK.setup!r}."
        )
    print("SELECT_SUBSET_PATH:", SELECT_SUBSET_PATH)
    print("SELECT_SUBSET:", SELECT_SUBSET)
    os.environ["APPLE_EXPERIMENT_ALGORITHM"] = RESULT_ALGORITHM
    print(
        "RUN VARIANT: OracleHardCluster-APPLE-PostALA (multi-selection) | selection_metrics="
        + "|".join(SELECTION_METRICS)
    )
    print("RUNS_ROOT:", RUNS_ROOT)
    print("EXPERIMENT_LOG_FOLDER:", EXPERIMENT_LOG_FOLDER)
    print("ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
    print("SAVE_CHECKPOINTS:", os.environ.get("SAVE_CHECKPOINTS", "0"))

    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    only_q = os.environ.get("ONLY_Q")
    if only_q is not None:
        if "q_value" not in chosen_df.columns:
            raise ValueError(
                "ONLY_Q was set, but selected_subset.csv has no q_value column."
            )
        q = float(only_q)
        chosen_df = chosen_df[
            pd.to_numeric(chosen_df["q_value"], errors="coerce")
            .round(10)
            .eq(round(q, 10))
        ].copy()
        print("ONLY_Q filter:", only_q, "remaining subsets:", len(chosen_df))

    max_subsets = os.environ.get("MAX_SUBSETS")
    if max_subsets is not None:
        chosen_df = chosen_df.head(int(max_subsets)).copy()
        print("MAX_SUBSETS filter:", max_subsets, "remaining subsets:", len(chosen_df))

    if chosen_df.empty:
        raise ValueError(
            "No selected subsets remain after ONLY_Q/MAX_SUBSETS filtering."
        )

    parent_subset_count = len(chosen_df)

    chosen_df = expand_oracle_hard_clusters(
        chosen_df
    )

    print(
        "Oracle hard-cluster expansion:",
        parent_subset_count,
        "parent rows ->",
        len(chosen_df),
        "five-client rows",
    )

    only_oracle_cluster = os.environ.get(
        "ONLY_ORACLE_CLUSTER"
    )

    if only_oracle_cluster is not None:
        oracle_cluster_id = int(
            only_oracle_cluster
        )

        chosen_df = chosen_df[
            pd.to_numeric(
                chosen_df["oracle_hard_cluster_id"],
                errors="raise",
            ).astype(int).eq(oracle_cluster_id)
        ].copy()

        print(
            "ONLY_ORACLE_CLUSTER filter:",
            oracle_cluster_id,
            "remaining subsets:",
            len(chosen_df),
        )

    if chosen_df.empty:
        raise ValueError(
            "No hard-cluster subsets remain after "
            "oracle-cluster filtering."
        )

    print(
        chosen_df[
            [
                "q_value",
                "oracle_hard_cluster_id",
                "oracle_hard_base_graph_id",
                "subset_size",
                "subset_clients",
            ]
        ].to_string(index=False)
    )

    id_to_client = load_clients(
        chosen_df, csv_path=ALL_DATA_LOGS, verbose=True, mask_splits=MASK_SPLITS
    )
    if os.environ.get("SKIP_Q_MASK_AUDIT", "0") == "1":
        print("SKIP_Q_MASK_AUDIT=1 -> skipping q-mask audit.")
    else:
        audit_loaded_q_label_masks(
            chosen_df, id_to_client, strict=True, mask_splits=MASK_SPLITS
        )

    if os.environ.get("MASK_AUDIT_ONLY") == "1":
        print("MASK_AUDIT_ONLY=1 -> stopping after mask audit.")
        return

    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    base_cfg["output_head"] = "multi"
    base_cfg["apple_mixing_mode"] = "oracle_hard_cluster_fedavg_backbone_support_simplex_task_head"
    base_cfg["selection_tag"] = "ms5_oracle_hard_cluster_postala_supportsimplex_v2"
    base_cfg["result_algorithm"] = RESULT_ALGORITHM
    base_cfg["apple_use_head_ala"] = APPLE_USE_HEAD_ALA
    base_cfg["apple_ala_lr"] = APPLE_ALA_LR
    base_cfg["apple_ala_rand_percent"] = APPLE_ALA_RAND_PERCENT
    base_cfg["apple_ala_convergence_std"] = APPLE_ALA_CONVERGENCE_STD
    base_cfg["apple_ala_convergence_window"] = APPLE_ALA_CONVERGENCE_WINDOW
    base_cfg["apple_ala_max_steps"] = APPLE_ALA_MAX_STEPS
    base_cfg["apple_ala_debug"] = APPLE_ALA_DEBUG
    base_cfg["apple_support_pseudo_count"] = APPLE_SUPPORT_PSEUDO_COUNT
    base_cfg["apple_routing_mode"] = "adaptive_visible_support_simplex"

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
    if "family_counts_json" in chosen_df.columns:
        print(chosen_df[["family_counts_json"]].to_string(index=False))

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

    print("Number of sweep configs:", len(sweep))
    print("Number of runs:", len(sweep) * len(chosen_df))
    progress = ProgressPrinter(len(sweep) * len(chosen_df))

    for cfg, meta in sweep:
        seed = int(meta["seed"])
        for _, row in chosen_df.iterrows():
            subset_ids = parse_subset_clients(row["subset_clients"])
            subset_clients = [id_to_client[graph_id] for graph_id in subset_ids]
            client_metadata = build_client_metadata(row, subset_clients)
            run_start = time.perf_counter()

            print()
            print(
                f"Starting with {len(subset_clients)}-client OracleHardCluster-APPLE-PostALA training",
                row["subset_clients"],
            )
            print(meta)
            print("SEED:", seed, "ROUNDS:", ROUNDS, "LOCAL_EPOCHS:", LOCAL_EPOCHS)
            print("APPLE_DR_LR:", APPLE_DR_LR)
            print("APPLE_ROUTING: adaptive_visible_support_simplex")
            print("APPLE_SUPPORT_PSEUDO_COUNT:", APPLE_SUPPORT_PSEUDO_COUNT)
            print(
                "APPLE_USE_HEAD_ALA:",
                APPLE_USE_HEAD_ALA,
                "ALA_LR:",
                APPLE_ALA_LR,
                "ALA_RAND_PERCENT:",
                APPLE_ALA_RAND_PERCENT,
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

            run_paths = run_apple(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                dr_lr=APPLE_DR_LR,
                support_pseudo_count=APPLE_SUPPORT_PSEUDO_COUNT,
                device=DEVICE,
                selection_metrics=SELECTION_METRICS,
                client_metadata=client_metadata,
            )

            log_row = build_apple_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=run_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                dr_lr=APPLE_DR_LR,
                support_pseudo_count=APPLE_SUPPORT_PSEUDO_COUNT,
                selection_metrics=SELECTION_METRICS,
            )
            add_subset_metadata(log_row, row)
            log_row["run_type"] = RESULT_RUN_TYPE
            log_row["algorithm"] = RESULT_ALGORITHM
            log_row["display_name"] = DISPLAY_NAME
            log_row["selection_metric"] = "multi"
            log_row["selection_metrics"] = "|".join(SELECTION_METRICS)
            log_row["selection_direction"] = "mixed"
            log_row["mask_splits"] = "|".join(MASK_SPLITS)
            log_row["selection_modes"] = "full|visible"
            log_row["eval_protocols"] = (
                "oracle_full|realistic_visible|realistic_selection_oracle"
            )
            log_row["apple_use_head_ala"] = APPLE_USE_HEAD_ALA
            log_row["apple_ala_lr"] = APPLE_ALA_LR
            log_row["apple_ala_rand_percent"] = APPLE_ALA_RAND_PERCENT
            log_row["apple_ala_convergence_std"] = APPLE_ALA_CONVERGENCE_STD
            log_row["apple_ala_convergence_window"] = APPLE_ALA_CONVERGENCE_WINDOW
            log_row["apple_ala_max_steps"] = APPLE_ALA_MAX_STEPS
            log_row["apple_support_pseudo_count"] = APPLE_SUPPORT_PSEUDO_COUNT
            log_row["apple_routing_mode"] = "adaptive_visible_support_simplex"
            log_row["apple_dr_proximal_regularization"] = False
            log_row["apple_ala_training_order"] = "apple_train_then_ala_filter"
            log_row["apple_ala_materialized_filter"] = True
            log_row["backbone_aggregation_scope"] = (
                "oracle_hard_cluster_only"
            )
            log_row["task_head_donor_scope"] = (
                "oracle_hard_cluster_only"
            )
            log_row["communication"] = (
                "within_oracle_cluster_only"
            )
            log_row["task_head_dr_length"] = len(
                subset_clients
            )
            log_row["oracle_hard_cluster_id"] = int(
                row["oracle_hard_cluster_id"]
            )
            log_row["oracle_hard_base_graph_id"] = int(
                row["oracle_hard_base_graph_id"]
            )
            log_row["oracle_parent_subset_id"] = str(
                row["oracle_parent_subset_id"]
            )
            log_row["oracle_parent_subset_clients"] = str(
                row["oracle_parent_subset_clients"]
            )
            upsert_experiment_rows(experiment_log_csv, [log_row])

            progress.step(
                label=f"OracleHardCluster-APPLE-PostALA done -> {run_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
