import os
import json
from typing import List, Optional, Sequence, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from andrea.multigraph_generation import TASKS


GRAPH_PARAM_CSV = "./andrea/clustering/cluster_generation_parameters.csv"
OUT_DIR = "./andrea/clustering"
SPLIT = "train"

# How many clients to keep per family
CLIENTS_PER_FAMILY = 4

# Support filtering
PMIN_LOW = 0.03

# Family thresholds (fixed once here; do not keep changing them every run)
SLOPE_THRESH = 0.015
STRONG_SLOPE_THRESH = 0.03

FAMILY_ORDER = [
    "strong_decreasing",
    "mild_decreasing",
    "flat_balanced",
    "mild_increasing",
    "strong_increasing",
]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def rate_col(task: str) -> str:
    return f"{SPLIT}_{task}_pos_rate"


def dataset_id_from_row(row: pd.Series) -> str:
    if "dataset_id" in row and pd.notna(row["dataset_id"]):
        return str(row["dataset_id"])
    if "data_dir" in row and pd.notna(row["data_dir"]):
        return os.path.basename(str(row["data_dir"]).rstrip("/"))
    graph_type = str(row["type"]) if "type" in row else "graph"
    return f"data_{int(row['n'])}_{int(row['d'])}_{row['r']}_{graph_type}"


def profile_slope(v: np.ndarray) -> float:
    x = np.arange(len(v), dtype=np.float64)
    y = np.asarray(v, dtype=np.float64)
    m, _b = np.polyfit(x, y, deg=1)
    return float(m)


def rough_shape_family(v: np.ndarray) -> str:

    v = np.asarray(v, dtype=np.float64)
    slope = profile_slope(v)

    if slope <= SLOPE_THRESH and slope >= -SLOPE_THRESH:
        return "flat_balanced"

    if slope > SLOPE_THRESH:
        if slope <= STRONG_SLOPE_THRESH:
            return "mild_increasing"
        if slope > STRONG_SLOPE_THRESH:
            return "strong_increasing"

    if slope < -SLOPE_THRESH:
        if slope > -STRONG_SLOPE_THRESH:
            return "mild_decreasing"
        if slope <= -STRONG_SLOPE_THRESH:
            return "strong_decreasing"

    return "error"


# -----------------------------------------------------------------------------
# Heterogeneity helpers
# -----------------------------------------------------------------------------
def safe_prob(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x + eps
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / max(len(x), 1)
    return x / s


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    p = safe_prob(np.asarray(p, dtype=np.float64), eps)
    q = safe_prob(np.asarray(q, dtype=np.float64), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a) - np.log(b))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_selected_subset_heterogeneity_logs(
    selected_df: pd.DataFrame,
) -> Dict[str, float]:
    if len(selected_df) == 0:
        return {}

    p_cols = [f"p_{task}" for task in TASKS]
    P = selected_df[p_cols].to_numpy(dtype=np.float64)

    # Pairwise JSD across all selected clients
    pairwise_jsds = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            pairwise_jsds.append(js_divergence(P[i], P[j]))

    # Mean centroid profile per family, then pairwise JSD between family centroids
    family_centroids = []
    for family in FAMILY_ORDER:
        fam_sub = selected_df[selected_df["shape_family"] == family]
        if len(fam_sub) == 0:
            continue
        centroid = fam_sub[p_cols].to_numpy(dtype=np.float64).mean(axis=0)
        family_centroids.append(centroid)

    centroid_jsds = []
    for i in range(len(family_centroids)):
        for j in range(i + 1, len(family_centroids)):
            centroid_jsds.append(
                js_divergence(family_centroids[i], family_centroids[j])
            )

    logs: Dict[str, float] = {
        "task_profile_jsd_mean": (
            float(np.mean(pairwise_jsds)) if len(pairwise_jsds) > 0 else 0.0
        ),
        "task_profile_jsd_median": (
            float(np.median(pairwise_jsds)) if len(pairwise_jsds) > 0 else 0.0
        ),
        "task_profile_jsd_max": (
            float(np.max(pairwise_jsds)) if len(pairwise_jsds) > 0 else 0.0
        ),
        "between_family_centroid_jsd_mean": (
            float(np.mean(centroid_jsds)) if len(centroid_jsds) > 0 else 0.0
        ),
    }

    # Raw per-task spread across selected clients
    for task in TASKS:
        logs[f"{task}_pos_rate_std_across_clients"] = float(
            selected_df[f"p_{task}"].std(ddof=0)
        )

    return logs


# -----------------------------------------------------------------------------
# Build the full pool
# -----------------------------------------------------------------------------
def build_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_id"] = out.apply(dataset_id_from_row, axis=1)

    p_cols: List[str] = []
    for task in TASKS:
        pcol = f"p_{task}"
        out[pcol] = out[rate_col(task)].astype(np.float64)
        p_cols.append(pcol)

    P = out[p_cols].to_numpy(dtype=np.float64)

    out["p_sum"] = P.sum(axis=1)
    out["p_mean"] = P.mean(axis=1)
    out["p_min"] = P.min(axis=1)
    out["p_max"] = P.max(axis=1)
    out["p_spread"] = out["p_max"] - out["p_min"]
    out["slope"] = [profile_slope(p) for p in P]
    out["dominant_task"] = [TASKS[int(np.argmax(p))] for p in P]
    out["shape_family"] = [rough_shape_family(p) for p in P]

    return out


# -----------------------------------------------------------------------------
# Reporting / very light plotting
# -----------------------------------------------------------------------------
def print_pool_summary(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"N clients: {len(df)}")

    counts = (
        df["shape_family"]
        .value_counts()
        .reindex(FAMILY_ORDER + ["error"], fill_value=0)
    )
    pct = (counts / max(len(df), 1) * 100.0).round(2)
    print("\nFAMILY COUNTS")
    print(pd.DataFrame({"count": counts, "percent": pct}).to_string())

    desc_cols = ["slope", "p_mean", "p_min", "p_max", "p_spread"]
    print("\nDESCRIPTOR SUMMARY")
    print(df[desc_cols].describe().round(4).to_string())


def print_selected_clients_per_family(selected_df: pd.DataFrame) -> None:
    show_cols = [
        "graph_id",
        "dataset_id",
        "d",
        "r",
        "shape_family",
        "slope",
        "p_mean",
        "p_min",
        "p_max",
        "p_spread",
        "dist_to_family_center",
        "dominant_task",
    ] + [f"p_{task}" for task in TASKS]

    print("\n" + "=" * 80)
    print("SELECTED CLIENTS PER FAMILY")
    print("=" * 80)

    for family in FAMILY_ORDER:
        sub = selected_df[selected_df["shape_family"] == family].sort_values(
            "dist_to_family_center"
        )
        print(f"\nFAMILY: {family}  (n={len(sub)})")
        print("-" * 80)
        if len(sub) == 0:
            print("No selected clients")
            continue
        print(sub[show_cols].to_string(index=False))


# -----------------------------------------------------------------------------
# Filter the pool before selection
# -----------------------------------------------------------------------------
def filter_candidate_pool(profile_df: pd.DataFrame) -> pd.DataFrame:
    out = profile_df.copy()

    print("\n" + "=" * 80)
    print("FILTERING CANDIDATE POOL")
    print("=" * 80)
    print(f"start: {len(out)} clients")

    # Minimum task support filter
    out = out[out["p_min"] >= PMIN_LOW].copy()
    print(f"after p_min >= {PMIN_LOW:.2f}: {len(out)} clients")

    return out


# -----------------------------------------------------------------------------
# Select representative clients from each family
# -----------------------------------------------------------------------------
def select_representatives_per_family(
    candidate_df: pd.DataFrame, per_family: int = CLIENTS_PER_FAMILY
) -> pd.DataFrame:
    """
    Within each family, pick the clients closest to that family's empirical center.

    This avoids choosing weird outliers while keeping the rule simple.
    """
    picks = []

    for family in FAMILY_ORDER:
        sub = candidate_df[candidate_df["shape_family"] == family].copy()
        if len(sub) == 0:
            print(f"WARNING: family {family} has 0 candidates")
            continue

        center_slope = sub["slope"].median()
        center_spread = sub["p_spread"].median()
        center_pmean = sub["p_mean"].median()

        slope_scale = sub["slope"].std(ddof=0) + 1e-12
        spread_scale = sub["p_spread"].std(ddof=0) + 1e-12
        pmean_scale = sub["p_mean"].std(ddof=0) + 1e-12

        sub["dist_to_family_center"] = np.sqrt(
            ((sub["slope"] - center_slope) / slope_scale) ** 2
            + ((sub["p_spread"] - center_spread) / spread_scale) ** 2
            + ((sub["p_mean"] - center_pmean) / pmean_scale) ** 2
        )

        chosen = sub.sort_values("dist_to_family_center").head(per_family)
        picks.append(chosen)

        if len(chosen) < per_family:
            print(
                f"WARNING: family {family} has only {len(chosen)} selected clients "
                f"(< requested {per_family})"
            )

    if len(picks) == 0:
        return candidate_df.iloc[[]].copy()

    return pd.concat(picks, ignore_index=False).reset_index(drop=True)


def build_combined_manifest_from_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    if len(selected_df) == 0:
        return pd.DataFrame()

    sub = selected_df.copy().sort_values(
        ["shape_family", "dist_to_family_center", "graph_id"]
    )

    graph_ids = [int(x) for x in sub["graph_id"].tolist()]
    dataset_ids = [str(x) for x in sub["dataset_id"].tolist()]
    subset_clients = "|".join(str(x) for x in graph_ids)

    # One benchmark row containing all selected families together
    subset_id = f"combined_five_family_benchmark_{subset_clients.replace('|', '_')}"

    family_to_graph_ids = {}
    family_to_dataset_ids = {}
    graph_to_family = {}
    dataset_to_family = {}
    membership_records = []

    for family in FAMILY_ORDER:
        fam_sub = sub[sub["shape_family"] == family].copy()
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
                }
            )

    family_counts = (
        sub["shape_family"].value_counts().reindex(FAMILY_ORDER, fill_value=0).to_dict()
    )

    heterogeneity_logs = compute_selected_subset_heterogeneity_logs(sub)

    row = {
        "family": "combined_five_family_benchmark",
        "subset_id": subset_id,
        "subset_size": int(len(graph_ids)),
        "subset_clients": subset_clients,
        "graph_ids_json": json.dumps(graph_ids),
        "dataset_ids_json": json.dumps(dataset_ids),
        "family_order_json": json.dumps(FAMILY_ORDER),
        "family_to_graph_ids_json": json.dumps(family_to_graph_ids),
        "family_to_dataset_ids_json": json.dumps(family_to_dataset_ids),
        "graph_to_family_json": json.dumps(graph_to_family),
        "dataset_to_family_json": json.dumps(dataset_to_family),
        "membership_json": json.dumps(membership_records),
        "family_counts_json": json.dumps(family_counts),
        "mean_slope": float(sub["slope"].mean()),
        "std_slope": float(sub["slope"].std(ddof=0)),
        "mean_p_mean": float(sub["p_mean"].mean()),
        "std_p_mean": float(sub["p_mean"].std(ddof=0)),
        "mean_p_min": float(sub["p_min"].mean()),
        "mean_p_spread": float(sub["p_spread"].mean()),
        "max_dist_to_family_center": float(sub["dist_to_family_center"].max()),
        **heterogeneity_logs,
    }

    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    # 1) Build the full pool
    graph_df = pd.read_csv(GRAPH_PARAM_CSV)
    profile_df = build_profile_table(graph_df)
    print_pool_summary(profile_df, title="FULL PROFILE POOL")

    # 2) Filter to get a cleaner candidate pool
    candidate_df = filter_candidate_pool(profile_df)
    print_pool_summary(candidate_df, title="FILTERED CANDIDATE POOL")

    # 3) Pick 4 representatives per family
    selected_df = select_representatives_per_family(
        candidate_df, per_family=CLIENTS_PER_FAMILY
    ).copy()
    selected_df["family"] = selected_df["shape_family"]
    print_selected_clients_per_family(selected_df)

    # 4) Build subset manifest (one row per family) for training pipeline
    combined_manifest_df = build_combined_manifest_from_selected(selected_df)
    columns = [
        "task_profile_jsd_mean",
        "task_profile_jsd_median",
        "task_profile_jsd_max",
        "between_family_centroid_jsd_mean",
    ]
    print(combined_manifest_df[columns])
    columns = [f"{task}_pos_rate_std_across_clients" for task in TASKS]
    print(combined_manifest_df[columns])
    combined_manifest_csv = os.path.join(OUT_DIR, "selected_subset.csv")
    combined_manifest_df.to_csv(combined_manifest_csv, index=False)


if __name__ == "__main__":
    main()
