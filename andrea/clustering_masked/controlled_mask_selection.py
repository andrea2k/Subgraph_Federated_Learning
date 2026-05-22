import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from andrea.multigraph_generation import TASKS

GRAPH_PARAM_CSV = "./andrea/clustering_rep/cluster_generation_parameters.csv"
OUT_DIR = "./andrea/clustering_masked"
SPLIT = "train"

MASKED_GRAPH_PARAM_CSV = os.path.join(OUT_DIR, "cluster_generation_parameters.csv")
MASKED_SELECTED_SUBSET_CSV = os.path.join(OUT_DIR, "selected_subset.csv")

NUM_BASE_GRAPHS = 1

# Low / medium / high designed heterogeneity.
MASK_FRACTIONS = [0.20, 0.50, 0.80]

FAMILY_PREFIX = "masked"

BASE_MIN_POS_RATE = 0.01

# Avoid graph_id collisions with old registry.
VIRTUAL_GRAPH_ID_START = 1_000_000

# Mask metadata.
MASK_MODE = "drop_positive_labels"
MASK_APPLY_SPLIT = "train"
MASK_SEED_BASE = 0

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
    target_pos_rate: float = 0.50,
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
    target_pos_rate: float = 0.50,
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
# Virtual masked client construction
# =============================================================================
def visible_support_from_base(
    base_row: pd.Series,
    *,
    mask_task: str,
    mask_fraction: float,
) -> Dict[str, float]:
    """
    Compute effective visible train support after masking positives of one task.
    """
    out: Dict[str, float] = {}

    for task in TASKS:
        r_col = rate_col(task)
        n_col = pos_nodes_col(task)

        true_rate = float(base_row[r_col])
        true_pos_nodes = float(base_row[n_col]) if n_col in base_row.index else np.nan

        if task == mask_task:
            visible_rate = true_rate * (1.0 - mask_fraction)
            visible_pos_nodes = true_pos_nodes * (1.0 - mask_fraction)
        else:
            visible_rate = true_rate
            visible_pos_nodes = true_pos_nodes

        out[f"visible_{r_col}"] = float(visible_rate)
        if not np.isnan(visible_pos_nodes):
            out[f"visible_{n_col}"] = int(round(visible_pos_nodes))
    return out


def make_virtual_client_row(
    base_row: pd.Series,
    *,
    virtual_graph_id: int,
    base_rank: int,
    mask_task: str,
    mask_fraction: float,
) -> pd.Series:
    row = base_row.copy()

    base_graph_id = int(base_row["graph_id"])
    base_dataset_id = dataset_id_from_row(base_row)
    x_tag = str(int(round(mask_fraction * 100)))

    virtual_dataset_id = f"{base_dataset_id}_mask{x_tag}_{mask_task}"
    # Preserve the original train support before optionally overwriting train_* columns.
    for task in TASKS:
        r_col = rate_col(task)
        n_col = pos_nodes_col(task)
        if r_col in row.index:
            row[f"true_{r_col}"] = row[r_col]
        if n_col in row.index:
            row[f"true_{n_col}"] = row[n_col]
    visible = visible_support_from_base(
        base_row,
        mask_task=mask_task,
        mask_fraction=mask_fraction,
    )
    for key, value in visible.items():
        row[key] = value

    if OVERWRITE_TRAIN_SUPPORT_WITH_VISIBLE:
        for task in TASKS:
            r_col = rate_col(task)
            n_col = pos_nodes_col(task)
            row[r_col] = row[f"visible_{r_col}"]
            if f"visible_{n_col}" in row.index and n_col in row.index:
                row[n_col] = row[f"visible_{n_col}"]

    row["graph_id"] = int(virtual_graph_id)
    row["dataset_id"] = virtual_dataset_id

    # data_dir remains the base graph directory.
    # The loader will apply mask metadata after loading train.pt.
    row["base_graph_id"] = base_graph_id
    row["base_dataset_id"] = base_dataset_id
    row["base_rank"] = int(base_rank)

    row["mask_mode"] = MASK_MODE
    row["mask_task"] = mask_task
    row["mask_fraction"] = float(mask_fraction)
    row["mask_seed"] = int(MASK_SEED_BASE + virtual_graph_id)
    row["mask_apply_split"] = MASK_APPLY_SPLIT
    row["designed_heterogeneity"] = float(mask_fraction)
    row["controlled_benchmark"] = "task_positive_label_masking"

    # Family name used by selected_subset manifest.
    row["target_family"] = f"{FAMILY_PREFIX}_{mask_task}"

    # Effective profile columns for manifest heterogeneity logs.
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
    base_df: pd.DataFrame, original_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    virtual_graph_id = next_virtual_graph_id_start(original_df)

    for base_rank, (_, base_row) in enumerate(base_df.iterrows()):
        for mask_fraction in MASK_FRACTIONS:
            for mask_task in TASKS:
                row = make_virtual_client_row(
                    base_row,
                    virtual_graph_id=virtual_graph_id,
                    base_rank=base_rank,
                    mask_task=mask_task,
                    mask_fraction=mask_fraction,
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
    mask_fraction: float,
) -> Dict:
    sub = selected_df.copy().sort_values(["target_family", "graph_id"])
    graph_ids = [int(x) for x in sub["graph_id"].tolist()]
    dataset_ids = [str(x) for x in sub["dataset_id"].tolist()]
    subset_clients = "|".join(str(x) for x in graph_ids)

    subset_id = (
        f"controlled_mask_base{base_rank}_x{mask_fraction}_"
        f"{subset_clients.replace('|', '_')}"
    )

    family_order = [f"{FAMILY_PREFIX}_{task}" for task in TASKS]

    family_to_graph_ids: Dict[str, List[int]] = {}
    family_to_dataset_ids: Dict[str, List[str]] = {}
    graph_to_family: Dict[str, str] = {}
    dataset_to_family: Dict[str, str] = {}
    membership_records: List[Dict] = []

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

            membership_records.append(
                {
                    "graph_id": gid,
                    "dataset_id": did,
                    "family": family,
                    "base_graph_id": int(row["base_graph_id"]),
                    "base_dataset_id": str(row["base_dataset_id"]),
                    "mask_task": str(row["mask_task"]),
                    "mask_fraction": float(row["mask_fraction"]),
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

    row = {
        # Old-style required fields.
        "family": "controlled_masked_five_client_benchmark",
        "subset_id": subset_id,
        "mask_fraction": float(mask_fraction),
        "subset_size": int(len(graph_ids)),
        "subset_clients": subset_clients,
        "graph_ids_json": safe_json_dumps(graph_ids),
        "dataset_ids_json": safe_json_dumps(dataset_ids),
        "family_order_json": safe_json_dumps(family_order),
        "family_to_graph_ids_json": safe_json_dumps(family_to_graph_ids),
        "family_to_dataset_ids_json": safe_json_dumps(family_to_dataset_ids),
        "graph_to_family_json": safe_json_dumps(graph_to_family),
        "dataset_to_family_json": safe_json_dumps(dataset_to_family),
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
        "controlled_benchmark": "task_positive_label_masking",
        "base_rank": int(base_rank),
        "base_graph_ids_json": safe_json_dumps(base_graph_ids),
        "base_dataset_ids_json": safe_json_dumps(base_dataset_ids),
        "mask_mode": MASK_MODE,
        "mask_apply_split": MASK_APPLY_SPLIT,
        "designed_heterogeneity": float(mask_fraction),
        **heterogeneity_logs,
    }

    return row


def build_masked_manifest(masked_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    grouped = masked_df.groupby(["base_rank", "mask_fraction"], sort=True)
    for (base_rank, mask_fraction), group in grouped:
        rows.append(
            build_manifest_row_for_group(
                group,
                base_rank=int(base_rank),
                mask_fraction=float(mask_fraction),
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
        "mask_fraction",
        "designed_heterogeneity",
        "task_profile_jsd_mean",
        "task_profile_jsd_median",
        "task_profile_jsd_max",
        "between_family_centroid_jsd_mean",
        "mean_p_mean",
        "mean_p_min",
        "mean_p_spread",
    ]
    existing_cols = [c for c in show_cols if c in manifest_df.columns]

    print("\n" + "=" * 100)
    print("MASKED SELECTED SUBSET SUMMARY")
    print("=" * 100)
    print(manifest_df[existing_cols].to_string(index=False))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ensure_dir(OUT_DIR)

    original_df = pd.read_csv(GRAPH_PARAM_CSV).copy()
    base_df = choose_balanced_base_graphs(
        original_df,
        NUM_BASE_GRAPHS,
        target_pos_rate=0.5,
    )
    print_base_graphs(base_df)
    masked_registry_df = build_masked_registry(base_df, original_df)
    masked_manifest_df = build_masked_manifest(masked_registry_df)

    masked_registry_df.to_csv(MASKED_GRAPH_PARAM_CSV, index=False)
    masked_manifest_df.to_csv(MASKED_SELECTED_SUBSET_CSV, index=False)

    print_manifest_summary(masked_manifest_df)

    print("\nWROTE:")
    print(f"  masked registry : {MASKED_GRAPH_PARAM_CSV}")
    print(f"  masked manifest : {MASKED_SELECTED_SUBSET_CSV}")


if __name__ == "__main__":
    main()
