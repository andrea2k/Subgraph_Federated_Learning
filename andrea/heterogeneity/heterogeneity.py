import os
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Sequence

from andrea.client_stats.extract_stats import OUT_DIR
from andrea.multigraph_generation import TASKS

FEATURE_DIR = OUT_DIR


# Make a valid probability distribution: add eps everywhere, renormalize.
# function used before JS divergence
def _safe_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x + eps
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / s


# Jensen–Shannon divergence (base-e)
def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps)
    q = _safe_prob(q, eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# Make sure the histogram lengths are equal for JSD
def pad_to_length(v: Sequence[float], L: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size >= L:
        return v[:L]
    out = np.zeros(L, dtype=np.float64)
    out[: v.size] = v
    return out


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps)
    return float(-np.sum(p * np.log(p)))


def compute_subset_metrics(
    df_split: pd.DataFrame, graph_ids: List[int]
) -> Dict[str, float]:

    sub = df_split[df_split["graph_id"].isin(graph_ids)].reset_index(drop=True)

    if len(sub) == 0:
        raise ValueError("No clients selected (graph_ids not found in parquet).")

    # (1) node count std/range
    n = sub["num_nodes"].to_numpy(dtype=np.float64)
    node_std = float(n.std(ddof=0))
    node_range = float(n.max() - n.min())

    # (2) density std/range
    dens = sub["density"].to_numpy(dtype=np.float64)
    dens_std = float(dens.std(ddof=0))
    dens_range = float(dens.max() - dens.min())

    # (3) degree JSD in/out (client vs subset-global)
    in_hists = sub["in_deg_hist_counts"].tolist()
    out_hists = sub["out_deg_hist_counts"].tolist()

    Lin = max(len(v) for v in in_hists)
    Lout = max(len(v) for v in out_hists)

    in_mat = np.stack([pad_to_length(v, Lin) for v in in_hists], axis=0)
    out_mat = np.stack([pad_to_length(v, Lout) for v in out_hists], axis=0)

    in_global = _safe_prob(in_mat.sum(axis=0))
    out_global = _safe_prob(out_mat.sum(axis=0))

    in_jsds = [
        js_divergence(_safe_prob(in_mat[i]), in_global) for i in range(in_mat.shape[0])
    ]
    out_jsds = [
        js_divergence(_safe_prob(out_mat[i]), out_global)
        for i in range(out_mat.shape[0])
    ]

    # (4) motif profile JSD (client vs subset-global)
    motif_mat = np.stack(
        [np.asarray(v, dtype=np.float64) for v in sub["motif_counts"].tolist()], axis=0
    )
    motif_global = _safe_prob(motif_mat.sum(axis=0))
    motif_jsds = [
        js_divergence(_safe_prob(motif_mat[i]), motif_global)
        for i in range(motif_mat.shape[0])
    ]

    # (5) label prevalence JSD (normalize label_counts to sum=1 per client; global from pooled label_counts)
    label_counts_mat = np.stack(
        [np.asarray(v, dtype=np.float64) for v in sub["label_counts"].tolist()], axis=0
    )
    label_global = _safe_prob(label_counts_mat.sum(axis=0))
    label_jsds = [
        js_divergence(_safe_prob(label_counts_mat[i]), label_global)
        for i in range(label_counts_mat.shape[0])
    ]

    # (6) labelset-size histogram JSD
    lss_mat = np.stack(
        [np.asarray(v, dtype=np.float64) for v in sub["labelset_size_counts"].tolist()],
        axis=0,
    )
    lss_global = _safe_prob(lss_mat.sum(axis=0))
    lss_jsds = [
        js_divergence(_safe_prob(lss_mat[i]), lss_global)
        for i in range(lss_mat.shape[0])
    ]

    # (7) label mixing matrix JSD
    mix_mat = np.stack(
        [np.asarray(v, dtype=np.float64) for v in sub["label_mixing_counts"].tolist()],
        axis=0,
    )
    mix_global = _safe_prob(mix_mat.sum(axis=0))
    mix_jsds = [
        js_divergence(_safe_prob(mix_mat[i]), mix_global)
        for i in range(mix_mat.shape[0])
    ]

    # (8) generator mix entropy (over subset)
    gen_counts = sub["type"].value_counts().to_numpy(dtype=np.float64)
    gen_p = _safe_prob(gen_counts)
    gen_ent = entropy(gen_p)
    gen_ent_norm = gen_ent / np.log(len(gen_p)) if len(gen_p) > 1 else 0.0

    # (9) param spread (n,d,r) from registry params stored in parquet
    params = sub[["n_param", "d_param", "r_param"]].to_numpy(dtype=np.float64)
    eps = 1e-12
    means = params.mean(axis=0)
    vars_ = params.var(axis=0)
    cv2 = vars_ / (means**2 + eps)
    cv2_mean = float(cv2.mean())

    # avg pairwise distance after z-score
    z = (params - params.mean(axis=0)) / (params.std(axis=0) + eps)
    dists = []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            dists.append(float(np.linalg.norm(z[i] - z[j])))
    avg_pairwise_zdist = float(np.mean(dists)) if dists else 0.0

    return {
        # node std / range (1)
        "node_count_std": node_std,
        "node_count_range": node_range,
        # density std / range (2)
        "density_std": dens_std,
        "density_range": dens_range,
        # in / out degree jsd (3)
        "in_degree_jsd_mean": float(np.mean(in_jsds)),
        "out_degree_jsd_mean": float(np.mean(out_jsds)),
        # motif count jsd (4)
        "motif_profile_jsd_mean": float(np.mean(motif_jsds)),
        # label prevalence jsd (5)
        "label_prev_jsd_mean": float(np.mean(label_jsds)),
        # labelset-size histogram jsd (6)
        "labelset_size_jsd_mean": float(np.mean(lss_jsds)),
        # label mixing matrix jsd (7)
        "label_mixing_jsd_mean": float(np.mean(mix_jsds)),
        # generator mix entropy (8)
        "generator_entropy": float(gen_ent),
        "generator_entropy_norm": float(gen_ent_norm),
        # param spread (n,d,r) (9)
        "param_cv2_mean": float(cv2_mean),
        "param_avg_pairwise_zdist": float(avg_pairwise_zdist),
        "param_cv2_n": float(cv2[0]),
        "param_cv2_d": float(cv2[1]),
        "param_cv2_r": float(cv2[2]),
    }


def sample_candidate_subsets(
    graph_ids: List[str], subset_size: int, num_candidates: int, rng: random.Random
):
    seen = set()
    subsets = []
    max_possible = (
        math.comb(len(graph_ids), subset_size) if len(graph_ids) >= subset_size else 0
    )
    target = min(num_candidates, max_possible)
    while len(subsets) < target:
        cand = tuple(sorted(rng.sample(graph_ids, subset_size)))
        if cand not in seen:
            seen.add(cand)
            subsets.append(list(cand))
    return subsets


SEED = 0
NUM_CANDIDATE_SUBSETS = 50
DIR = "andrea"


def main():
    split = "train"
    df = pd.read_parquet(f"{FEATURE_DIR}/client_features_{split}.parquet")
    graph_ids = df["graph_id"].unique().tolist()
    rows = []

    rng = random.Random(SEED)

    for subset_size in range(4, 9):  # 4,5,6,7,8
        subsets = sample_candidate_subsets(
            graph_ids=graph_ids,
            subset_size=subset_size,
            num_candidates=NUM_CANDIDATE_SUBSETS,
            rng=rng,
        )
        print(f"subset_size={subset_size}, sampled={len(subsets)}")

        for i, subset in enumerate(subsets):
            metrics = compute_subset_metrics(df, subset)
            return
            row = {
                "subset_id": f"s{subset_size}_{i:03d}",
                "subset_size": subset_size,
                "subset_clients": "|".join(map(str, subset)),
                **metrics,
            }
            rows.append(row)

    subset_df = pd.DataFrame(rows)
    subset_df.to_csv(os.path.join(DIR, "heterogeneity.csv"), index=False)
    print(
        f"Saved heterogeneity metrics for different subsets to:",
        os.path.join(DIR, "heterogeneity.csv"),
    )


if __name__ == "__main__":
    main()
