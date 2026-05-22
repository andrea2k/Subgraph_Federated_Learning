import os
import json
from typing import List, Sequence, Dict
import numpy as np
import pandas as pd

from andrea.multigraph_generation import TASKS


GRAPH_PARAM_CSV = "./andrea/clustering_rep/cluster_generation_parameters.csv"
OUT_DIR = "./andrea/clustering_rep"
SPLIT = "train"

# ---------------------------------------------------------------------
# Benchmark size
# ---------------------------------------------------------------------
CLIENTS_PER_FAMILY = 4

# ---------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------
# Remove clients where one task is nearly absent.
PMIN_LOW = 0.01

# ---------------------------------------------------------------------
# Family order
# ---------------------------------------------------------------------
FAMILY_ORDER = [
    "strong_decreasing",
    "mild_decreasing",
    "flat_balanced",
    "mild_increasing",
    "strong_increasing",
]


TARGET_GAMMAS = [0.07, 0.14, 0.21]
TARGET_KAPPA = 0.2


SHORTLIST_PER_FAMILY = 30

# Robust p_mean overlap across gamma-specific shortlists
OVERLAP_Q_LOW = 0.20
OVERLAP_Q_HIGH = 0.80

# Final score weights
PMEAN_WEIGHT = 0.50
SLOPE_WEIGHT = 0.25

# ---------------------------------------------------------------------
# Diversity settings
# ---------------------------------------------------------------------
# A "cell" means one coarse generator setting:
#   (n, d, r, type)
# We ignore rep/seed here on purpose.
#
# Why:
# you now generate many repeated seeds per same (n,d,r,type),
# so without this cap the selector may pick many almost-identical
# clients from the same coarse regime.
MAX_SHORTLIST_PER_CELL = 4
MAX_FINAL_PER_CELL = 4

# ---------------------------------------------------------------------
# Family coding for the target-centroid construction
# ---------------------------------------------------------------------
FAMILY_CODE = {
    "strong_decreasing": -2,
    "mild_decreasing": -1,
    "flat_balanced": 0,
    "mild_increasing": 1,
    "strong_increasing": 2,
}

# Increasing/decreasing direction across tasks
INCREASING_DIR = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)

# Mild center bump for the flat family
FLAT_SHAPE = np.array([-1.0, 1.0, 1.2, 1.0, -1.0], dtype=np.float64)


# =====================================================================
# Basic helpers
# =====================================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def rate_col(task: str) -> str:
    return f"{SPLIT}_{task}_pos_rate"


def gamma_tag(gamma: float) -> str:
    return str(gamma).replace(".", "p")


def profile_slope(v: np.ndarray) -> float:
    """
    Linear slope of the 5-task profile.
    Negative -> decreasing family
    Positive -> increasing family
    """
    x = np.arange(len(v), dtype=np.float64)
    y = np.asarray(v, dtype=np.float64)
    m, _b = np.polyfit(x, y, deg=1)
    return float(m)


def softmax_vec(z: Sequence[float]) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def parameter_cell_id(row: pd.Series) -> str:
    """
    Coarse generation cell used for diversity control.

    Important:
    rep is intentionally excluded.
    So:
        same (n,d,r,type), different reps/seeds
    are treated as the same cell.
    """
    graph_type = str(row["type"]) if "type" in row else "graph"
    return f"{int(row['n'])}|{int(row['d'])}|{float(row['r'])}|{graph_type}"


# =====================================================================
# Target centroids
# =====================================================================
def make_family_target_centroids(
    gamma: float,
    kappa: float,
) -> Dict[str, np.ndarray]:
    """
    Build one normalized 5-task centroid for each family.

    z = kappa * FLAT_SHAPE + gamma * s_family * INCREASING_DIR
    centroid = softmax(z)

    Interpretation:
    - kappa controls the flat baseline shape
    - gamma controls how separated the families become
    """
    out: Dict[str, np.ndarray] = {}
    for family in FAMILY_ORDER:
        s = FAMILY_CODE[family]
        z = (kappa * FLAT_SHAPE) + (gamma * s * INCREASING_DIR)
        out[family] = softmax_vec(z)
    return out


# =====================================================================
# Heterogeneity helpers
# =====================================================================
def safe_prob(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x + eps
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / max(len(x), 1)
    return x / s


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """
    JSD on normalized profiles.
    This means heterogeneity is defined on profile shape.
    """
    p = safe_prob(np.asarray(p, dtype=np.float64), eps)
    q = safe_prob(np.asarray(q, dtype=np.float64), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a) - np.log(b))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_selected_subset_heterogeneity_logs(
    selected_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute the heterogeneity summary that ends up inside selected_subset.csv.

    This mirrors your earlier logic:
    - pairwise JSD across all selected clients
    - JSD between family mean profiles
    - raw per-task std across all selected clients
    """
    if len(selected_df) == 0:
        return {}

    p_cols = [f"p_{task}" for task in TASKS]
    P = selected_df[p_cols].to_numpy(dtype=np.float64)

    pairwise_jsds = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            pairwise_jsds.append(js_divergence(P[i], P[j]))

    family_centroids = []
    for family in FAMILY_ORDER:
        fam_sub = selected_df[selected_df["target_family"] == family]
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

    for task in TASKS:
        logs[f"{task}_pos_rate_std_across_clients"] = float(
            selected_df[f"p_{task}"].std(ddof=0)
        )

    return logs


# =====================================================================
# Build the pool for one gamma
# =====================================================================
def build_profile_table(
    df: pd.DataFrame,
    gamma: float,
    kappa: float,
) -> pd.DataFrame:
    """
    For one gamma:
    - read train split task rates
    - compute task profile stats
    - assign each client to the nearest target family centroid
    """
    out = df.copy()
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
    out["param_cell"] = out.apply(parameter_cell_id, axis=1)

    target_centroids = make_family_target_centroids(gamma=gamma, kappa=kappa)
    centroid_jsds = []
    for i in range(len(target_centroids)):
        for j in range(i + 1, len(target_centroids)):
            centroid_jsds.append(
                js_divergence(
                    target_centroids[FAMILY_ORDER[i]], target_centroids[FAMILY_ORDER[j]]
                )
            )
    assigned_families = []
    assigned_jsds = []

    for p in P:
        jsd_map = {
            family: js_divergence(p, target_centroids[family])
            for family in FAMILY_ORDER
        }
        best_family = min(jsd_map, key=jsd_map.get)
        assigned_families.append(best_family)
        assigned_jsds.append(jsd_map[best_family])

    out["target_family"] = assigned_families
    out["profile_jsd_to_target"] = assigned_jsds

    return out


def filter_candidate_pool(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove low-support clients before any selection.
    """
    out = profile_df.copy()
    out = out[out["p_min"] >= PMIN_LOW].copy()
    return out


# =====================================================================
# Diversity-aware selection helper
# =====================================================================
def greedy_select_with_cell_cap(
    df: pd.DataFrame,
    k: int,
    max_per_cell: int,
    sort_cols: List[str],
    ascending: List[bool],
) -> pd.DataFrame:
    """
    Greedy selector with diversity cap.

    First pass:
    - respect max_per_cell

    Fallback pass:
    - if too few candidates survive, fill remaining slots without the cap

    Why fallback:
    some families may genuinely have too few distinct cells.
    """
    if len(df) == 0:
        return df.iloc[[]].copy()

    ordered = df.sort_values(sort_cols, ascending=ascending).copy()
    target_count = min(k, len(ordered))
    chosen_indices = []
    cell_counts: Dict[str, int] = {}

    for idx, row in ordered.iterrows():
        cell = str(row["param_cell"])
        if cell_counts.get(cell, 0) >= max_per_cell:
            continue
        chosen_indices.append(idx)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        if len(chosen_indices) == target_count:
            break

    # Fallback: if cap was too strict, fill the remaining slots
    if len(chosen_indices) < target_count:
        for idx, _row in ordered.iterrows():
            if idx in chosen_indices:
                continue
            chosen_indices.append(idx)
            if len(chosen_indices) == target_count:
                break

    return ordered.loc[chosen_indices].copy().reset_index(drop=True)


# =====================================================================
# Stage 1: shortlist by shape
# =====================================================================
def build_shape_shortlist(
    candidate_df: pd.DataFrame,
    shortlist_per_family: int = SHORTLIST_PER_FAMILY,
) -> pd.DataFrame:
    """
    For each target family:
    1. restrict to that family
    2. rank by shape match
    3. cap how many repeated-seed copies from the same coarse cell can enter
    """
    picks = []

    for family in FAMILY_ORDER:
        sub = candidate_df[candidate_df["target_family"] == family].copy()
        if len(sub) == 0:
            continue

        chosen = greedy_select_with_cell_cap(
            df=sub,
            k=shortlist_per_family,
            max_per_cell=MAX_SHORTLIST_PER_CELL,
            sort_cols=["profile_jsd_to_target", "p_min", "p_mean"],
            ascending=[True, False, False],
        )
        picks.append(chosen)

    if len(picks) == 0:
        return candidate_df.iloc[[]].copy()

    return pd.concat(picks, ignore_index=False).reset_index(drop=True)


# =====================================================================
# Stage 2: family-wise p_mean matching across gamma
# =====================================================================
def compute_family_pmean_overlap_info(
    shortlists_by_gamma: Dict[float, pd.DataFrame],
    q_low: float = OVERLAP_Q_LOW,
    q_high: float = OVERLAP_Q_HIGH,
) -> pd.DataFrame:
    """
    For each family:
    - collect shortlist p_mean values from all gammas
    - compute a shared overlap band
    - define one common target_pmean

    First try robust quantile overlap:
        [max(q_low), min(q_high)]

    If empty, relax to:
        [min(median), max(median)]
    """
    rows = []

    for family in FAMILY_ORDER:
        per_gamma_vals = {}
        medians = []
        qlows = []
        qhighs = []

        for gamma, df_g in shortlists_by_gamma.items():
            sub = df_g[df_g["target_family"] == family].copy()
            vals = sub["p_mean"].to_numpy(dtype=np.float64)

            if len(vals) == 0:
                continue

            per_gamma_vals[gamma] = vals
            medians.append(float(np.median(vals)))
            qlows.append(float(np.quantile(vals, q_low)))
            qhighs.append(float(np.quantile(vals, q_high)))

        if len(per_gamma_vals) == 0:
            rows.append(
                {
                    "family": family,
                    "target_pmean": np.nan,
                    "overlap_low": np.nan,
                    "overlap_high": np.nan,
                    "band_mode": "empty",
                    "pmean_scale": np.nan,
                }
            )
            continue

        overlap_low = max(qlows)
        overlap_high = min(qhighs)
        band_mode = "quantile_overlap"

        if overlap_low > overlap_high:
            overlap_low = min(medians)
            overlap_high = max(medians)
            band_mode = "median_span"

        target_pmean = float(np.median(medians))
        concat_vals = np.concatenate(list(per_gamma_vals.values()))
        pmean_scale = float(np.std(concat_vals, ddof=0)) + 1e-12

        row = {
            "family": family,
            "target_pmean": target_pmean,
            "overlap_low": float(overlap_low),
            "overlap_high": float(overlap_high),
            "band_mode": band_mode,
            "pmean_scale": pmean_scale,
        }

        for gamma in TARGET_GAMMAS:
            vals = per_gamma_vals.get(gamma, None)
            tag = gamma_tag(gamma)
            if vals is None:
                row[f"gamma_{tag}_shortlist_count"] = 0
                row[f"gamma_{tag}_shortlist_pmean_min"] = np.nan
                row[f"gamma_{tag}_shortlist_pmean_max"] = np.nan
                row[f"gamma_{tag}_shortlist_pmean_median"] = np.nan
            else:
                row[f"gamma_{tag}_shortlist_count"] = int(len(vals))
                row[f"gamma_{tag}_shortlist_pmean_min"] = float(np.min(vals))
                row[f"gamma_{tag}_shortlist_pmean_max"] = float(np.max(vals))
                row[f"gamma_{tag}_shortlist_pmean_median"] = float(np.median(vals))

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Stage 3: final selection for one gamma
# =====================================================================
def select_representatives_with_family_pmean_overlap(
    shortlist_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    gamma: float,
    per_family: int = CLIENTS_PER_FAMILY,
    kappa: float = TARGET_KAPPA,
) -> pd.DataFrame:
    """
    Final selection for one gamma.

    Score = shape fit + p_mean fairness + slope fit

    Then:
    - prefer candidates inside the overlap band
    - also prevent repeated-seed copies from same coarse cell from dominating
    """
    picks = []
    target_centroids = make_family_target_centroids(gamma=gamma, kappa=kappa)

    for family in FAMILY_ORDER:
        sub = shortlist_df[shortlist_df["target_family"] == family].copy()
        if len(sub) == 0:
            continue

        info = overlap_df[overlap_df["family"] == family].iloc[0]

        target = target_centroids[family]
        target_slope = profile_slope(target)

        target_pmean = (
            float(info["target_pmean"])
            if pd.notna(info["target_pmean"])
            else float(sub["p_mean"].median())
        )
        overlap_low = (
            float(info["overlap_low"])
            if pd.notna(info["overlap_low"])
            else float(sub["p_mean"].min())
        )
        overlap_high = (
            float(info["overlap_high"])
            if pd.notna(info["overlap_high"])
            else float(sub["p_mean"].max())
        )
        pmean_scale = (
            float(info["pmean_scale"])
            if pd.notna(info["pmean_scale"])
            else float(sub["p_mean"].std(ddof=0) + 1e-12)
        )

        sub["target_slope"] = target_slope
        sub["target_pmean"] = target_pmean
        sub["overlap_low"] = overlap_low
        sub["overlap_high"] = overlap_high
        sub["pmean_abs_to_target"] = np.abs(sub["p_mean"] - target_pmean)
        sub["slope_abs_to_target"] = np.abs(sub["slope"] - target_slope)

        slope_scale = sub["slope"].std(ddof=0) + 1e-12

        sub["dist_to_family_center"] = (
            sub["profile_jsd_to_target"]
            + PMEAN_WEIGHT * (sub["pmean_abs_to_target"] / pmean_scale)
            + SLOPE_WEIGHT * (sub["slope_abs_to_target"] / slope_scale)
        )

        # Prefer overlap-band candidates if there are enough of them
        in_band = sub[
            (sub["p_mean"] >= overlap_low) & (sub["p_mean"] <= overlap_high)
        ].copy()

        if len(in_band) >= per_family:
            pool = in_band.copy()
            pool["selected_from_overlap_band"] = 1
        else:
            pool = sub.copy()
            pool["selected_from_overlap_band"] = 0

        chosen = greedy_select_with_cell_cap(
            df=pool,
            k=per_family,
            max_per_cell=MAX_FINAL_PER_CELL,
            sort_cols=[
                "dist_to_family_center",
                "profile_jsd_to_target",
                "pmean_abs_to_target",
                "p_min",
            ],
            ascending=[True, True, True, False],
        )

        picks.append(chosen)

    if len(picks) == 0:
        return shortlist_df.iloc[[]].copy()

    out = pd.concat(picks, ignore_index=False).reset_index(drop=True)
    out["selection_gamma"] = float(gamma)
    return out


# =====================================================================
# Family summary for manifest logging
# =====================================================================
def build_family_summary_table(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for family in FAMILY_ORDER:
        sub = selected_df[selected_df["target_family"] == family].copy()
        if len(sub) == 0:
            continue

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
            "profile_jsd_to_target_mean": float(sub["profile_jsd_to_target"].mean()),
            "profile_jsd_to_target_std": float(
                sub["profile_jsd_to_target"].std(ddof=0)
            ),
        }

        for task in TASKS:
            col = f"p_{task}"
            row[f"{task}_mean"] = float(sub[col].mean())
            row[f"{task}_std"] = float(sub[col].std(ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Build one manifest row for one gamma
# =====================================================================
def build_combined_manifest_from_selected(
    selected_df: pd.DataFrame,
    gamma: float,
    kappa: float = TARGET_KAPPA,
) -> pd.DataFrame:
    """
    Output one row in selected_subset.csv.

    Important:
    the core column names stay in the same manifest style as before,
    so your experiment code can still loop over rows.
    """
    if len(selected_df) == 0:
        return pd.DataFrame()

    sub = selected_df.copy().sort_values(
        ["target_family", "dist_to_family_center", "graph_id"]
    )

    graph_ids = [int(x) for x in sub["graph_id"].tolist()]
    dataset_ids = [str(x) for x in sub["dataset_id"].tolist()]
    subset_clients = "|".join(str(x) for x in graph_ids)

    tag = gamma_tag(gamma)
    subset_id = (
        f"combined_five_family_benchmark_gamma_{tag}_{subset_clients.replace('|', '_')}"
    )

    family_to_graph_ids = {}
    family_to_dataset_ids = {}
    graph_to_family = {}
    dataset_to_family = {}
    membership_records = []

    for family in FAMILY_ORDER:
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
                }
            )

    family_counts = (
        sub["target_family"]
        .value_counts()
        .reindex(FAMILY_ORDER, fill_value=0)
        .to_dict()
    )

    heterogeneity_logs = compute_selected_subset_heterogeneity_logs(sub)
    family_summary_df = build_family_summary_table(sub)

    family_to_pmean_mean = {}
    family_to_pmean_std = {}
    family_to_slope_mean = {}
    family_to_slope_std = {}
    family_to_task_means = {}
    family_to_task_stds = {}

    for family in FAMILY_ORDER:
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

    row = {
        "family": "combined_five_family_benchmark",
        "subset_id": subset_id,
        "gamma": gamma,
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
        "family_to_pmean_mean_json": json.dumps(family_to_pmean_mean),
        "family_to_pmean_std_json": json.dumps(family_to_pmean_std),
        "family_to_slope_mean_json": json.dumps(family_to_slope_mean),
        "family_to_slope_std_json": json.dumps(family_to_slope_std),
        "family_to_task_means_json": json.dumps(family_to_task_means),
        "family_to_task_stds_json": json.dumps(family_to_task_stds),
        **heterogeneity_logs,
    }

    return pd.DataFrame([row])


# =====================================================================
# Main pipeline
# =====================================================================
def main() -> None:
    ensure_dir(OUT_DIR)

    graph_df = pd.read_csv(GRAPH_PARAM_CSV)

    # -------------------------------------------------------------
    # Stage A:
    # For each gamma, build the gamma-specific family assignment
    # and shortlist by shape.
    # -------------------------------------------------------------
    shortlists_by_gamma: Dict[float, pd.DataFrame] = {}

    for gamma in TARGET_GAMMAS:
        profile_df = build_profile_table(
            graph_df,
            gamma=gamma,
            kappa=TARGET_KAPPA,
        )
        candidate_df = filter_candidate_pool(profile_df)

        shortlist_df = build_shape_shortlist(
            candidate_df,
            shortlist_per_family=SHORTLIST_PER_FAMILY,
        ).copy()

        shortlist_df["selection_gamma"] = float(gamma)
        shortlists_by_gamma[gamma] = shortlist_df

    # -------------------------------------------------------------
    # Stage B:
    # Match the gammas by a shared within-family p_mean target.
    # -------------------------------------------------------------
    overlap_df = compute_family_pmean_overlap_info(
        shortlists_by_gamma,
        q_low=OVERLAP_Q_LOW,
        q_high=OVERLAP_Q_HIGH,
    )
    # -------------------------------------------------------------
    # Stage C:
    # Final selection and manifest writing.
    # -------------------------------------------------------------
    all_manifest_rows = []

    for gamma in TARGET_GAMMAS:
        shortlist_df = shortlists_by_gamma[gamma]

        selected_df = select_representatives_with_family_pmean_overlap(
            shortlist_df=shortlist_df,
            overlap_df=overlap_df,
            gamma=gamma,
            per_family=CLIENTS_PER_FAMILY,
            kappa=TARGET_KAPPA,
        ).copy()

        selected_df["family"] = selected_df["target_family"]
        selected_df["gamma"] = float(gamma)

        combined_manifest_df = build_combined_manifest_from_selected(
            selected_df,
            gamma=gamma,
            kappa=TARGET_KAPPA,
        )

        all_manifest_rows.append(combined_manifest_df.copy())

    # Final 3-row benchmark manifest
    if len(all_manifest_rows) > 0:
        all_manifest_df = pd.concat(all_manifest_rows, ignore_index=True)
        all_manifest_df.to_csv(
            os.path.join(OUT_DIR, "selected_subset.csv"), index=False
        )
    print("Done!")


if __name__ == "__main__":
    main()
