import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from andrea.multigraph_generation import TASKS

GRAPH_PARAM_CSV = "./andrea/clustering/cluster_generation_parameters.csv"
OUT_DIR = "./andrea/clustering_q_label_heterogeneity"
SPLIT = "train"

MASKED_GRAPH_PARAM_CSV = os.path.join(OUT_DIR, "cluster_generation_parameters.csv")
MASKED_SELECTED_SUBSET_CSV = os.path.join(OUT_DIR, "selected_subset.csv")

NUM_BASE_GRAPHS = 1

# Low / medium / high designed heterogeneity.
Q_VALUES = [1.0 / len(TASKS), 0.50, 0.80]

FAMILY_PREFIX = "q_task"

BASE_TARGET_POS_RATE = 0.05
BASE_MIN_POS_RATE = 0.01

# Avoid graph_id collisions with old registry.
VIRTUAL_GRAPH_ID_START = 3_000_000

# Mask metadata.
MASK_MODE = "q_task_label_allocation"
MASK_APPLY_SPLIT = "train"
MASK_SEED_BASE = 0

CONTROLLED_BENCHMARK = "q_controlled_task_label_heterogeneity"

# If True, the masked registry overwrites train_*_pos_rate / train_*_pos_nodes
# with the visible-after-mask support. The original values are still preserved in
# true_train_* columns. This is useful because old heterogeneity utilities that
# read train_*_pos_rate will see the effective training profile.
OVERWRITE_TRAIN_SUPPORT_WITH_VISIBLE = True


# =============================================================================
# Basic helpers
# =============================================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def rate_col(task: str, split: str = SPLIT) -> str:
    return f"{split}_{task}_pos_rate"


def pos_nodes_col(task: str, split: str = SPLIT) -> str:
    return f"{split}_{task}_pos_nodes"


def motif_count_col(task: str, split: str = SPLIT) -> str:
    return f"{split}_{task}_motif_count"


def safe_json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def dataset_id_from_row(row: pd.Series) -> str:
    if "dataset_id" in row and pd.notna(row["dataset_id"]):
        return str(row["dataset_id"])
    if "data_dir" in row and pd.notna(row["data_dir"]):
        return os.path.basename(str(row["data_dir"]).rstrip("/"))
    graph_type = str(row["type"]) if "type" in row else "graph"
    rep = row.get("rep", None)
    rep_part = f"_rep{int(rep):02d}" if rep is not None and pd.notna(rep) else ""
    return f"data_{int(row['n'])}_{int(row['d'])}_{row['r']}_{graph_type}{rep_part}"


def profile_slope(v: Sequence[float]) -> float:
    x = np.arange(len(v), dtype=np.float64)
    y = np.asarray(v, dtype=np.float64)
    m, _b = np.polyfit(x, y, deg=1)
    return float(m)


# =============================================================================
# JSD helpers
# =============================================================================
def safe_prob(x: Sequence[float], eps: float = 1e-20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) + eps
    s = float(x.sum())
    if s <= 0:
        return np.ones_like(x, dtype=np.float64) / max(len(x), 1)
    return x / s


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence on normalized task profiles.
    This matches the old selection convention: heterogeneity is profile-shape based.
    """
    p = safe_prob(p, eps)
    q = safe_prob(q, eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a) - np.log(b))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_subset_heterogeneity_logs(selected_df: pd.DataFrame) -> Dict[str, float]:
    """
    Same style as cluster_selection_revised.py:
    - mean/median/max pairwise JSD across all selected clients
    - mean pairwise JSD between family centroids
    - raw per-task std across selected clients
    """
    if len(selected_df) == 0:
        return {}

    p_cols = [f"p_{task}" for task in TASKS]
    P = selected_df[p_cols].to_numpy(dtype=np.float64)

    pairwise_jsds: List[float] = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            pairwise_jsds.append(js_divergence(P[i], P[j]))

    family_centroids: List[np.ndarray] = []
    for family in selected_df["target_family"].drop_duplicates().tolist():
        fam_sub = selected_df[selected_df["target_family"] == family]
        if len(fam_sub) == 0:
            continue
        family_centroids.append(fam_sub[p_cols].to_numpy(dtype=np.float64).mean(axis=0))

    centroid_jsds: List[float] = []
    for i in range(len(family_centroids)):
        for j in range(i + 1, len(family_centroids)):
            centroid_jsds.append(
                js_divergence(family_centroids[i], family_centroids[j])
            )

    logs: Dict[str, float] = {
        "task_profile_jsd_mean": (
            float(np.mean(pairwise_jsds)) if pairwise_jsds else 0.0
        ),
        "task_profile_jsd_median": (
            float(np.median(pairwise_jsds)) if pairwise_jsds else 0.0
        ),
        "task_profile_jsd_max": float(np.max(pairwise_jsds)) if pairwise_jsds else 0.0,
        "between_family_centroid_jsd_mean": (
            float(np.mean(centroid_jsds)) if centroid_jsds else 0.0
        ),
    }

    for task in TASKS:
        logs[f"{task}_pos_rate_std_across_clients"] = float(
            selected_df[f"p_{task}"].std(ddof=0)
        )

    return logs


def add_base_profile_columns(
    df: pd.DataFrame,
    target_pos_rate: float = 0.1,
) -> pd.DataFrame:

    out = df.copy()
    out["dataset_id"] = out.apply(dataset_id_from_row, axis=1)

    p_cols = []

    for task in TASKS:
        col = rate_col(task)

        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

        p_col = f"p_{task}"
        out[p_col] = out[col].astype(np.float64)
        p_cols.append(p_col)

    P = out[p_cols].to_numpy(dtype=np.float64)

    # Basic profile statistics.
    out["p_sum"] = P.sum(axis=1)
    out["p_mean"] = P.mean(axis=1)
    out["p_min"] = P.min(axis=1)
    out["p_max"] = P.max(axis=1)
    out["p_spread"] = out["p_max"] - out["p_min"]
    out["slope"] = [profile_slope(p) for p in P]

    # New simple target-rate scores.
    target = float(target_pos_rate)
    abs_err = np.abs(P - target)

    out["target_pos_rate"] = target

    # Average distance from target across tasks.
    out["target_abs_err_mean"] = abs_err.mean(axis=1)

    # Worst task distance from target.
    out["target_abs_err_max"] = abs_err.max(axis=1)

    # Smooth overall distance from target.
    out["target_abs_err_l2"] = np.sqrt((abs_err**2).mean(axis=1))

    return out


def choose_balanced_base_graphs(
    df: pd.DataFrame,
    n_base: int,
    target_pos_rate: float = 0.1,
) -> pd.DataFrame:
    prof = add_base_profile_columns(df, target_pos_rate=target_pos_rate)

    chosen = prof.sort_values(
        [
            "target_abs_err_max",
            "target_abs_err_mean",
            "p_spread",
            "p_mean",
        ],
        ascending=[True, True, True, False],
    ).head(n_base)

    if len(chosen) < n_base:
        print(f"WARNING: requested {n_base} base graphs but only found {len(chosen)}.")

    return chosen.reset_index(drop=True)


# =============================================================================
# masked client construction
# =============================================================================


def q_tag_from_value(q_value: float) -> str:
    return str(int(round(float(q_value) * 100)))


def q_iid_value() -> float:
    return 1.0 / float(len(TASKS))


def q_other_share(q_value: float) -> float:
    num_tasks = len(TASKS)
    if num_tasks <= 1:
        raise ValueError("q-controlled allocation requires at least 2 tasks.")
    return (1.0 - float(q_value)) / float(num_tasks - 1)


def validate_q_value(q_value: float) -> None:
    q = float(q_value)
    iid_q = q_iid_value()

    if q < 0.0 or q > 1.0:
        raise ValueError(f"q_value must be in [0, 1], got {q}")

    if q + 1e-12 < iid_q:
        raise ValueError(
            f"q_value={q} is below IID q=1/T={iid_q}. "
            "For this benchmark, use q >= 1/T."
        )


def visible_support_from_base(
    base_row: pd.Series,
    *,
    assigned_task: str,
    q_value: float,
) -> Dict[str, float]:
    """
    Expected visible positive support for deterministic q label allocation.

    For each task t:
      - the client assigned to t receives q of t's positive labels;
      - every other client receives (1 - q) / (T - 1) of t's positive labels.

    Example with 5 tasks, 200 positives per task, q=0.5:
      assigned client: 100 positives
      each other client: 25 positives
    """
    validate_q_value(q_value)

    out: Dict[str, float] = {}

    q = float(q_value)
    off = q_other_share(q)
    iid_q = q_iid_value()

    for task in TASKS:
        r_col = rate_col(task)
        n_col = pos_nodes_col(task)

        true_rate = float(base_row[r_col])
        true_pos_nodes = float(base_row[n_col]) if n_col in base_row.index else np.nan

        share = q if task == assigned_task else off

        visible_rate = true_rate * share
        visible_pos_nodes = true_pos_nodes * share

        out[f"visible_{r_col}"] = float(visible_rate)
        out[f"q_share_{task}"] = float(share)

        if not np.isnan(visible_pos_nodes):
            out[f"visible_{n_col}"] = int(round(visible_pos_nodes))

    out["q_value"] = float(q)
    out["q_iid"] = float(iid_q)
    out["q_other_share"] = float(off)
    out["q_assigned_share"] = float(q)

    return out


def make_virtual_client_row(
    base_row: pd.Series,
    *,
    virtual_graph_id: int,
    base_rank: int,
    assigned_task: str,
    q_value: float,
) -> pd.Series:
    row = base_row.copy()

    validate_q_value(q_value)

    base_graph_id = int(base_row["graph_id"])
    base_dataset_id = dataset_id_from_row(base_row)
    q_tag = q_tag_from_value(q_value)

    virtual_dataset_id = f"{base_dataset_id}_q{q_tag}_{assigned_task}"

    # Preserve original train support.
    for task in TASKS:
        r_col = rate_col(task)
        n_col = pos_nodes_col(task)

        if r_col in row.index:
            row[f"true_{r_col}"] = row[r_col]

        if n_col in row.index:
            row[f"true_{n_col}"] = row[n_col]

    # Compute expected visible positive support after q allocation.
    visible = visible_support_from_base(
        base_row,
        assigned_task=assigned_task,
        q_value=q_value,
    )
    for key, value in visible.items():
        row[key] = value

    # Make train_* columns represent the effective visible positive profile.
    # This keeps old plotting / heterogeneity utilities working, but now the
    # values mean q-visible positive support, not old mask-fraction support.
    if OVERWRITE_TRAIN_SUPPORT_WITH_VISIBLE:
        for task in TASKS:
            r_col = rate_col(task)
            n_col = pos_nodes_col(task)

            row[r_col] = row[f"visible_{r_col}"]
            if f"visible_{n_col}" in row.index and n_col in row.index:
                row[n_col] = row[f"visible_{n_col}"]

    row["graph_id"] = int(virtual_graph_id)
    row["dataset_id"] = virtual_dataset_id

    # data_dir stays the base graph directory.
    # So these are virtual clients over the same base graph.
    row["base_graph_id"] = base_graph_id
    row["base_dataset_id"] = base_dataset_id
    row["base_rank"] = int(base_rank)

    other_tasks = [task for task in TASKS if task != assigned_task]

    # Metadata consumed later by load_client_helper.py.
    row["mask_mode"] = MASK_MODE
    row["assigned_task"] = assigned_task

    # Keep mask_task for backward-compatible naming.
    # Here mask_task means the assigned task / q-majority task.
    row["mask_task"] = assigned_task

    row["masked_tasks_json"] = safe_json_dumps(other_tasks)

    # New q metadata.
    row["q_value"] = float(q_value)
    row["q_iid"] = float(q_iid_value())
    row["q_other_share"] = float(q_other_share(q_value))
    row["q_assigned_share"] = float(q_value)
    row["q_allocation_mode"] = "deterministic_disjoint_positive_label_chunks"

    # Backward-compatible alias:
    # old runners expect mask_fraction / designed_heterogeneity.
    # From now on, interpret this as q, not as old masking fraction.
    row["mask_fraction"] = float(q_value)
    row["mask_seed"] = int(MASK_SEED_BASE + virtual_graph_id)
    row["mask_apply_split"] = MASK_APPLY_SPLIT
    row["designed_heterogeneity"] = float(q_value)
    row["controlled_benchmark"] = CONTROLLED_BENCHMARK

    # Family name used by selected_subset manifest.
    row["target_family"] = f"{FAMILY_PREFIX}_{assigned_task}"

    # Effective profile columns for heterogeneity logs.
    for task in TASKS:
        row[f"p_{task}"] = float(row[rate_col(task)])

    p = np.asarray([row[f"p_{task}"] for task in TASKS], dtype=np.float64)

    row["p_sum"] = float(p.sum())
    row["p_mean"] = float(p.mean())
    row["p_min"] = float(p.min())
    row["p_max"] = float(p.max())
    row["p_spread"] = float(p.max() - p.min())
    row["slope"] = profile_slope(p)
    row["dominant_task"] = TASKS[int(np.argmax(p))]

    return row


def next_virtual_graph_id_start(df: pd.DataFrame) -> int:
    max_existing = int(pd.to_numeric(df["graph_id"]).max())
    return max(VIRTUAL_GRAPH_ID_START, max_existing + 1)


def build_masked_registry(
    base_df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    virtual_graph_id = next_virtual_graph_id_start(original_df)

    for base_rank, (_, base_row) in enumerate(base_df.iterrows()):
        for q_value in Q_VALUES:
            for assigned_task in TASKS:
                row = make_virtual_client_row(
                    base_row,
                    virtual_graph_id=virtual_graph_id,
                    base_rank=base_rank,
                    assigned_task=assigned_task,
                    q_value=q_value,
                )
                rows.append(row)
                virtual_graph_id += 1

    masked_df = pd.DataFrame(rows).reset_index(drop=True)
    return masked_df


# =============================================================================
# Manifest construction
# =============================================================================
def build_family_summary_table(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = selected_df["target_family"].drop_duplicates().tolist()

    for family in families:
        sub = selected_df[selected_df["target_family"] == family].copy()
        row = {
            "family": family,
            "n_clients": int(len(sub)),
            "p_mean_mean": float(sub["p_mean"].mean()),
            "p_mean_std": float(sub["p_mean"].std(ddof=0)),
            "slope_mean": float(sub["slope"].mean()),
            "slope_std": float(sub["slope"].std(ddof=0)),
            "p_min_mean": float(sub["p_min"].mean()),
            "p_max_mean": float(sub["p_max"].mean()),
            "p_spread_mean": float(sub["p_spread"].mean()),
        }

        for task in TASKS:
            col = f"p_{task}"
            row[f"{task}_mean"] = float(sub[col].mean())
            row[f"{task}_std"] = float(sub[col].std(ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


def build_manifest_row_for_group(
    selected_df: pd.DataFrame,
    *,
    base_rank: int,
    q_value: float,
) -> Dict:
    sub = selected_df.copy().sort_values(["target_family", "graph_id"])
    graph_ids = [int(x) for x in sub["graph_id"].tolist()]
    dataset_ids = [str(x) for x in sub["dataset_id"].tolist()]
    subset_clients = "|".join(str(x) for x in graph_ids)

    q_tag = q_tag_from_value(q_value)

    subset_id = (
        f"controlled_q_label_base{base_rank}_q{q_tag}_"
        f"{subset_clients.replace('|', '_')}"
    )

    family_order = [f"{FAMILY_PREFIX}_{task}" for task in TASKS]

    family_to_graph_ids: Dict[str, List[int]] = {}
    family_to_dataset_ids: Dict[str, List[str]] = {}
    graph_to_family: Dict[str, str] = {}
    dataset_to_family: Dict[str, str] = {}
    membership_records: List[Dict] = []

    graph_to_task: Dict[str, str] = {}
    dataset_to_task: Dict[str, str] = {}

    for family in family_order:
        fam_sub = sub[sub["target_family"] == family].copy()
        fam_graph_ids = [int(x) for x in fam_sub["graph_id"].tolist()]
        fam_dataset_ids = [str(x) for x in fam_sub["dataset_id"].tolist()]

        family_to_graph_ids[family] = fam_graph_ids
        family_to_dataset_ids[family] = fam_dataset_ids

        for _, row in fam_sub.iterrows():
            gid = int(row["graph_id"])
            did = str(row["dataset_id"])

            graph_to_family[str(gid)] = family
            dataset_to_family[did] = family

            graph_to_task[str(gid)] = str(row["assigned_task"])
            dataset_to_task[did] = str(row["assigned_task"])

            membership_records.append(
                {
                    "graph_id": gid,
                    "dataset_id": did,
                    "family": family,
                    "base_graph_id": int(row["base_graph_id"]),
                    "base_dataset_id": str(row["base_dataset_id"]),
                    "assigned_task": str(row["assigned_task"]),
                    "mask_task": str(row["mask_task"]),
                    "masked_tasks_json": str(row["masked_tasks_json"]),
                    "mask_mode": str(row["mask_mode"]),
                    "mask_fraction": float(row["mask_fraction"]),
                    "q_value": float(row["q_value"]),
                    "q_iid": float(row["q_iid"]),
                    "q_other_share": float(row["q_other_share"]),
                    "q_assigned_share": float(row["q_assigned_share"]),
                    "q_allocation_mode": str(row["q_allocation_mode"]),
                }
            )

    family_counts = (
        sub["target_family"]
        .value_counts()
        .reindex(family_order, fill_value=0)
        .to_dict()
    )

    heterogeneity_logs = compute_subset_heterogeneity_logs(sub)
    family_summary_df = build_family_summary_table(sub)

    family_to_pmean_mean = {}
    family_to_pmean_std = {}
    family_to_slope_mean = {}
    family_to_slope_std = {}
    family_to_task_means = {}
    family_to_task_stds = {}

    for family in family_order:
        fam = family_summary_df[family_summary_df["family"] == family]
        if len(fam) == 0:
            family_to_pmean_mean[family] = None
            family_to_pmean_std[family] = None
            family_to_slope_mean[family] = None
            family_to_slope_std[family] = None
            family_to_task_means[family] = {task: None for task in TASKS}
            family_to_task_stds[family] = {task: None for task in TASKS}
            continue

        fam = fam.iloc[0]
        family_to_pmean_mean[family] = float(fam["p_mean_mean"])
        family_to_pmean_std[family] = float(fam["p_mean_std"])
        family_to_slope_mean[family] = float(fam["slope_mean"])
        family_to_slope_std[family] = float(fam["slope_std"])
        family_to_task_means[family] = {
            task: float(fam[f"{task}_mean"]) for task in TASKS
        }
        family_to_task_stds[family] = {
            task: float(fam[f"{task}_std"]) for task in TASKS
        }

    base_graph_ids = sorted({int(x) for x in sub["base_graph_id"].tolist()})
    base_dataset_ids = sorted({str(x) for x in sub["base_dataset_id"].tolist()})

    num_tasks = len(TASKS)
    q = float(q_value)
    iid_q = q_iid_value()
    off = q_other_share(q)

    global_visible_positive_support_fraction_ideal = 1.0 / float(num_tasks)

    row = {
        "family": "controlled_q_label_heterogeneity_five_client_benchmark",
        "subset_id": subset_id,
        "q_value": float(q),
        "q_iid": float(iid_q),
        "q_other_share": float(off),
        "q_assigned_share": float(q),
        "mask_fraction": float(q),
        "subset_size": int(len(graph_ids)),
        "subset_clients": subset_clients,
        "graph_ids_json": safe_json_dumps(graph_ids),
        "dataset_ids_json": safe_json_dumps(dataset_ids),
        "family_order_json": safe_json_dumps(family_order),
        "family_to_graph_ids_json": safe_json_dumps(family_to_graph_ids),
        "family_to_dataset_ids_json": safe_json_dumps(family_to_dataset_ids),
        "graph_to_family_json": safe_json_dumps(graph_to_family),
        "dataset_to_family_json": safe_json_dumps(dataset_to_family),
        "graph_to_task_json": safe_json_dumps(graph_to_task),
        "dataset_to_task_json": safe_json_dumps(dataset_to_task),
        "membership_json": safe_json_dumps(membership_records),
        "family_counts_json": safe_json_dumps(family_counts),
        "mean_slope": float(sub["slope"].mean()),
        "std_slope": float(sub["slope"].std(ddof=0)),
        "mean_p_mean": float(sub["p_mean"].mean()),
        "std_p_mean": float(sub["p_mean"].std(ddof=0)),
        "mean_p_min": float(sub["p_min"].mean()),
        "mean_p_spread": float(sub["p_spread"].mean()),
        "family_to_pmean_mean_json": safe_json_dumps(family_to_pmean_mean),
        "family_to_pmean_std_json": safe_json_dumps(family_to_pmean_std),
        "family_to_slope_mean_json": safe_json_dumps(family_to_slope_mean),
        "family_to_slope_std_json": safe_json_dumps(family_to_slope_std),
        "family_to_task_means_json": safe_json_dumps(family_to_task_means),
        "family_to_task_stds_json": safe_json_dumps(family_to_task_stds),
        # New controlled-mask fields.
        "controlled_benchmark": CONTROLLED_BENCHMARK,
        "base_rank": int(base_rank),
        "base_graph_ids_json": safe_json_dumps(base_graph_ids),
        "base_dataset_ids_json": safe_json_dumps(base_dataset_ids),
        "mask_mode": MASK_MODE,
        "mask_apply_split": MASK_APPLY_SPLIT,
        "designed_heterogeneity": float(q),
        "specialization_fraction": float(q),
        "q_allocation_mode": "deterministic_disjoint_positive_label_chunks",
        "global_visible_positive_support_fraction_ideal": float(
            global_visible_positive_support_fraction_ideal
        ),
        # Backward-compatible alias. This no longer has the old meaning.
        "global_visible_support_fraction_ideal": float(
            global_visible_positive_support_fraction_ideal
        ),
        **heterogeneity_logs,
    }

    return row


def build_masked_manifest(masked_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    grouped = masked_df.groupby(["base_rank", "q_value"], sort=True)
    for (base_rank, q_value), group in grouped:
        print("q_value:", q_value)
        rows.append(
            build_manifest_row_for_group(
                group,
                base_rank=int(base_rank),
                q_value=float(q_value),
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# Reporting
# =============================================================================
def print_base_graphs(base_df: pd.DataFrame) -> None:
    show_cols = [
        "graph_id",
        "dataset_id",
        "data_dir",
    ] + [f"p_{task}" for task in TASKS]

    existing_cols = [c for c in show_cols if c in base_df.columns]
    print("\n" + "=" * 100)
    print("SELECTED BALANCED BASE GRAPH(S)")
    print("=" * 100)
    print(base_df[existing_cols].to_string(index=False))


def print_manifest_summary(manifest_df: pd.DataFrame) -> None:
    show_cols = [
        "subset_id",
        "subset_size",
        "q_value",
        "q_iid",
        "q_other_share",
        "specialization_fraction",
        "designed_heterogeneity",
        "global_visible_positive_support_fraction_ideal",
    ]
    existing_cols = [c for c in show_cols if c in manifest_df.columns]

    print("\n" + "=" * 100)
    print("Q-CONTROLLED LABEL HETEROGENEITY SELECTED SUBSET SUMMARY")
    print("=" * 100)
    print(manifest_df[existing_cols].to_string(index=False))

    show_cols = [
        "task_profile_jsd_mean",
        "task_profile_jsd_median",
        "task_profile_jsd_max",
        "between_family_centroid_jsd_mean",
        "mean_p_mean",
        "mean_p_min",
        "mean_p_spread",
    ]
    existing_cols = [c for c in show_cols if c in manifest_df.columns]
    print(manifest_df[existing_cols].to_string(index=False))


# =============================================================================
# Q-control correctness checks
# =============================================================================
def print_q_setup() -> None:
    print("\n" + "=" * 100)
    print("Q-CONTROLLED LABEL HETEROGENEITY SETUP")
    print("=" * 100)
    print("TASKS:", TASKS)
    print("num_tasks:", len(TASKS))
    print("IID q = 1 / num_tasks =", q_iid_value())
    print("Q_VALUES:", Q_VALUES)
    print("MASK_MODE:", MASK_MODE)
    print("OUT_DIR:", OUT_DIR)

    for q in Q_VALUES:
        validate_q_value(q)
        print(
            f"q={q:.6f} | assigned_share={q:.6f} | "
            f"off_task_share={(q_other_share(q)):.6f} | "
            f"sum_check={q + (len(TASKS) - 1) * q_other_share(q):.6f}"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_float(x) -> float:
    return float(pd.to_numeric(x))


def _format_matrix(df: pd.DataFrame, digits: int = 6) -> str:
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].map(
            lambda x: "" if pd.isna(x) else f"{float(x):.{digits}g}"
        )
    return tmp.to_string()


def audit_q_group(group: pd.DataFrame, *, base_rank: int, q_value: float) -> None:
    """
    Check one generated 5-client subset for one q value.

    This verifies the CSV/manifest target profiles:
      - exactly one virtual client per task
      - all virtual clients use the same base graph
      - assigned task gets q share
      - non-assigned tasks get (1-q)/(T-1) share
      - per-task visible support approximately sums back to the base support

    Important: this checks the metadata target. The actual tensor label_mask
    will be checked later after implementing q_task_label_allocation in
    load_client_helper.py.
    """
    q = float(q_value)
    off = q_other_share(q)
    num_tasks = len(TASKS)

    group = group.copy()
    group = group.sort_values(["assigned_task", "graph_id"])

    print("\n" + "-" * 100)
    print(f"AUDIT q group | base_rank={base_rank} | q={q:.6f} | off={off:.6f}")
    print("-" * 100)

    _require(
        len(group) == num_tasks,
        f"Expected {num_tasks} clients for q={q}, got {len(group)}",
    )

    assigned = set(group["assigned_task"].astype(str).tolist())
    _require(
        assigned == set(TASKS),
        f"Expected assigned tasks {TASKS}, got {sorted(assigned)}",
    )

    _require(
        group["base_graph_id"].nunique() == 1,
        f"Expected one base_graph_id, got {group['base_graph_id'].unique()}",
    )

    _require(
        group["data_dir"].nunique() == 1,
        f"Expected one shared data_dir, got {group['data_dir'].unique()}",
    )

    _require(
        group["graph_id"].is_unique,
        "Virtual graph_id values are not unique.",
    )

    _require(
        group["dataset_id"].is_unique,
        "Virtual dataset_id values are not unique.",
    )

    _require(
        set(group["mask_mode"].astype(str).unique()) == {MASK_MODE},
        f"Unexpected mask_mode values: {group['mask_mode'].unique()}",
    )

    _require(
        set(group["mask_apply_split"].astype(str).unique()) == {MASK_APPLY_SPLIT},
        f"Unexpected mask_apply_split values: {group['mask_apply_split'].unique()}",
    )

    sub = group.set_index("assigned_task").loc[TASKS]

    rate_matrix = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)
    node_matrix = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)
    share_matrix = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)

    max_rate_error = 0.0
    max_node_error = 0.0

    for client_task in TASKS:
        for task in TASKS:
            expected_share = q if task == client_task else off

            r_col = rate_col(task)
            true_r_col = f"true_{r_col}"

            actual_rate = _as_float(sub.loc[client_task, r_col])
            true_rate = _as_float(sub.loc[client_task, true_r_col])
            expected_rate = true_rate * expected_share

            err = abs(actual_rate - expected_rate)
            max_rate_error = max(max_rate_error, err)

            _require(
                err <= 1e-12,
                (
                    f"Rate mismatch for client_task={client_task}, task={task}: "
                    f"actual={actual_rate}, expected={expected_rate}, "
                    f"true_rate={true_rate}, share={expected_share}"
                ),
            )

            rate_matrix.loc[client_task, task] = actual_rate
            share_matrix.loc[client_task, task] = expected_share

            n_col = pos_nodes_col(task)
            true_n_col = f"true_{n_col}"

            if n_col in sub.columns and true_n_col in sub.columns:
                actual_nodes = int(round(_as_float(sub.loc[client_task, n_col])))
                true_nodes = _as_float(sub.loc[client_task, true_n_col])
                expected_nodes = int(round(true_nodes * expected_share))

                node_err = abs(actual_nodes - expected_nodes)
                max_node_error = max(max_node_error, node_err)

                _require(
                    node_err == 0,
                    (
                        f"Node-count mismatch for client_task={client_task}, task={task}: "
                        f"actual={actual_nodes}, expected={expected_nodes}, "
                        f"true_nodes={true_nodes}, share={expected_share}"
                    ),
                )

                node_matrix.loc[client_task, task] = actual_nodes

    print("\nshare matrix: rows=assigned client task, cols=label task")
    print(_format_matrix(share_matrix, digits=4))

    print("\nvisible positive-rate matrix:")
    print(_format_matrix(rate_matrix, digits=8))

    if not node_matrix.isna().all().all():
        print("\nvisible positive-node-count matrix:")
        print(_format_matrix(node_matrix, digits=8))

    print("\ncolumn-sum checks across the 5 virtual clients:")
    for task in TASKS:
        r_col = rate_col(task)
        true_r_col = f"true_{r_col}"

        true_rate = _as_float(sub.iloc[0][true_r_col])
        visible_rate_sum = float(sub[r_col].astype(float).sum())
        rate_err = abs(visible_rate_sum - true_rate)

        print(
            f"  {task}: visible_rate_sum={visible_rate_sum:.12g} | "
            f"true_rate={true_rate:.12g} | error={rate_err:.3g}"
        )

        _require(
            rate_err <= 1e-12,
            (
                f"Visible rate column for {task} does not sum back to true rate: "
                f"sum={visible_rate_sum}, true={true_rate}"
            ),
        )

        n_col = pos_nodes_col(task)
        true_n_col = f"true_{n_col}"

        if n_col in sub.columns and true_n_col in sub.columns:
            true_nodes = int(round(_as_float(sub.iloc[0][true_n_col])))
            visible_nodes_sum = int(round(sub[n_col].astype(float).sum()))
            node_err = abs(visible_nodes_sum - true_nodes)

            print(
                f"       visible_nodes_sum={visible_nodes_sum} | "
                f"true_nodes={true_nodes} | error={node_err}"
            )

            # Rounding can cause tiny differences if true positives are not nicely
            # divisible by the q/off shares. Keep this as warning-level tolerance.
            _require(
                node_err <= num_tasks,
                (
                    f"Visible node count column for {task} is too far from true count: "
                    f"sum={visible_nodes_sum}, true={true_nodes}, error={node_err}"
                ),
            )

    print(
        f"\nPASS q={q:.6f}: registry target profiles are internally consistent. "
        f"max_rate_error={max_rate_error:.3g}, max_node_error={max_node_error:.3g}"
    )


def audit_manifest(masked_manifest_df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("AUDIT MANIFEST")
    print("=" * 100)

    expected_rows = NUM_BASE_GRAPHS * len(Q_VALUES)
    _require(
        len(masked_manifest_df) == expected_rows,
        f"Expected {expected_rows} manifest rows, got {len(masked_manifest_df)}",
    )

    required_cols = [
        "subset_id",
        "subset_size",
        "subset_clients",
        "q_value",
        "q_iid",
        "q_other_share",
        "q_assigned_share",
        "mask_mode",
        "mask_apply_split",
        "graph_to_family_json",
        "graph_to_task_json",
        "membership_json",
        "family_counts_json",
    ]
    missing = [c for c in required_cols if c not in masked_manifest_df.columns]
    _require(not missing, f"Manifest is missing required columns: {missing}")

    for _, row in masked_manifest_df.iterrows():
        q = float(row["q_value"])
        off = q_other_share(q)

        _require(
            int(row["subset_size"]) == len(TASKS),
            f"subset_size should be {len(TASKS)} for q={q}",
        )
        _require(
            str(row["mask_mode"]) == MASK_MODE,
            f"Wrong mask_mode in manifest: {row['mask_mode']}",
        )
        _require(
            str(row["mask_apply_split"]) == MASK_APPLY_SPLIT,
            f"Wrong mask_apply_split in manifest: {row['mask_apply_split']}",
        )
        _require(
            abs(float(row["q_iid"]) - q_iid_value()) <= 1e-12,
            f"Wrong q_iid for q={q}",
        )
        _require(
            abs(float(row["q_other_share"]) - off) <= 1e-12,
            f"Wrong q_other_share for q={q}",
        )

        graph_to_task = json.loads(row["graph_to_task_json"])
        family_counts = json.loads(row["family_counts_json"])
        membership = json.loads(row["membership_json"])

        _require(
            set(graph_to_task.values()) == set(TASKS),
            f"graph_to_task_json does not cover exactly TASKS for q={q}: {graph_to_task}",
        )

        _require(
            len(membership) == len(TASKS),
            f"membership_json length should be {len(TASKS)} for q={q}",
        )

        _require(
            all(int(v) == 1 for v in family_counts.values()),
            f"Each family should have exactly one client for q={q}: {family_counts}",
        )

        print(
            f"PASS manifest row q={q:.6f} | subset_size={row['subset_size']} | "
            f"subset_clients={row['subset_clients']}"
        )


def audit_q_controlled_outputs(
    masked_registry_df: pd.DataFrame,
    masked_manifest_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 100)
    print("RUNNING Q-CONTROLLED OUTPUT AUDITS")
    print("=" * 100)

    required_registry_cols = [
        "graph_id",
        "dataset_id",
        "data_dir",
        "base_graph_id",
        "base_dataset_id",
        "base_rank",
        "assigned_task",
        "mask_task",
        "mask_mode",
        "mask_apply_split",
        "q_value",
        "q_iid",
        "q_other_share",
        "q_assigned_share",
        "q_allocation_mode",
    ]

    for task in TASKS:
        required_registry_cols.extend(
            [
                rate_col(task),
                f"true_{rate_col(task)}",
                f"visible_{rate_col(task)}",
                f"q_share_{task}",
            ]
        )

    missing = [c for c in required_registry_cols if c not in masked_registry_df.columns]
    _require(not missing, f"Masked registry is missing required columns: {missing}")

    expected_clients = NUM_BASE_GRAPHS * len(Q_VALUES) * len(TASKS)
    _require(
        len(masked_registry_df) == expected_clients,
        f"Expected {expected_clients} registry rows, got {len(masked_registry_df)}",
    )

    _require(
        masked_registry_df["graph_id"].is_unique,
        "graph_id values in masked registry are not unique.",
    )

    _require(
        masked_registry_df["dataset_id"].is_unique,
        "dataset_id values in masked registry are not unique.",
    )

    grouped = masked_registry_df.groupby(["base_rank", "q_value"], sort=True)
    for (base_rank, q_value), group in grouped:
        audit_q_group(group, base_rank=int(base_rank), q_value=float(q_value))

    audit_manifest(masked_manifest_df)

    print("\n" + "=" * 100)
    print("ALL Q-CONTROLLED METADATA AUDITS PASSED")
    print("=" * 100)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ensure_dir(OUT_DIR)

    original_df = pd.read_csv(GRAPH_PARAM_CSV).copy()
    base_df = choose_balanced_base_graphs(
        original_df,
        NUM_BASE_GRAPHS,
        target_pos_rate=BASE_TARGET_POS_RATE,
    )
    print_q_setup()
    print_base_graphs(base_df)

    masked_registry_df = build_masked_registry(base_df, original_df)
    masked_manifest_df = build_masked_manifest(masked_registry_df)

    audit_q_controlled_outputs(masked_registry_df, masked_manifest_df)

    masked_registry_df.to_csv(MASKED_GRAPH_PARAM_CSV, index=False)
    masked_manifest_df.to_csv(MASKED_SELECTED_SUBSET_CSV, index=False)

    print_manifest_summary(masked_manifest_df)

    print("\nWROTE:")
    print(f"  masked registry : {MASKED_GRAPH_PARAM_CSV}")
    print(f"  masked manifest : {MASKED_SELECTED_SUBSET_CSV}")


if __name__ == "__main__":
    main()
