import os
import math
import json
import itertools
import numpy as np
import pandas as pd
from typing import Sequence
from typing import Any, Dict

from andrea.client_stats.extract_stats import OUT_DIR
from andrea.multigraph_generation import TASKS

FEATURE_DIR = OUT_DIR
DIR = "andrea/heterogeneity"
SPLIT = "train"
DEFAULT_MIN_POS_COUNT = 100


def as_float_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, tuple):
        return np.asarray(list(value), dtype=np.float64)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            return np.asarray(json.loads(value), dtype=np.float64)
    raise TypeError(f"Unsupported array value type: {type(value)!r}")


def dump_json_vec(v: Sequence[float]) -> str:
    return json.dumps(np.asarray(v, dtype=np.float64).tolist())


def dump_json_task_gap_rows(v_i: Sequence[float], v_j: Sequence[float]) -> str:
    vi = np.asarray(v_i, dtype=np.float64)
    vj = np.asarray(v_j, dtype=np.float64)
    gaps = np.abs(vi - vj)
    rows = []
    for task_idx, task_name in enumerate(TASKS):
        rows.append(
            {
                "task": str(task_name),
                "value_i": float(vi[task_idx]),
                "value_j": float(vj[task_idx]),
                "gap": float(gaps[task_idx]),
            }
        )
    return json.dumps(rows)


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


def pad_to_length(v: Sequence[float], length: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size >= length:
        return v[:length]
    out = np.zeros(length, dtype=np.float64)
    out[: v.size] = v
    return out


def l1_mean_abs_gap(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def max_abs_gap(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def top_gap_task(v_i: Sequence[float], v_j: Sequence[float]) -> tuple[str, float]:
    vi = np.asarray(v_i, dtype=np.float64)
    vj = np.asarray(v_j, dtype=np.float64)
    gaps = np.abs(vi - vj)
    top_idx = int(np.argmax(gaps))
    return str(TASKS[top_idx]), float(gaps[top_idx])


def structured_complementarity_score(
    rate_i: Sequence[float],
    rate_j: Sequence[float],
    count_i: Sequence[float],
    count_j: Sequence[float],
    top_k: int = 1,
    min_presence_rate: float = 0.05,
    min_positive_count: int = DEFAULT_MIN_POS_COUNT,
    min_active_tasks: int = 3,
) -> float:
    """
    High score means:
    - one or a few tasks differ clearly (specialization)
    - the remaining tasks are still reasonably similar (shared background)
    - both clients have enough support on the remaining tasks to make collaboration meaningful
    """
    vi = np.asarray(rate_i, dtype=np.float64)
    vj = np.asarray(rate_j, dtype=np.float64)
    ci = np.asarray(count_i, dtype=np.float64)
    cj = np.asarray(count_j, dtype=np.float64)

    if vi.size == 0:
        return 0.0

    gaps = np.abs(vi - vj)
    order = np.argsort(gaps)[::-1]
    k = min(top_k, gaps.size)
    top_idx = order[:k]
    rest_idx = order[k:]
    focused_gap = float(np.mean(gaps[top_idx])) if top_idx.size > 0 else 0.0
    if rest_idx.size > 0:
        rest_gap = float(np.mean(gaps[rest_idx]))
        rest_similarity = float(np.clip(1.0 - rest_gap, 0.0, 1.0))
    else:
        rest_similarity = 1.0

    min_rate = (vi >= min_presence_rate) & (vj >= min_presence_rate)
    min_count = (ci >= min_positive_count) & (cj >= min_positive_count)
    if rest_idx.size > 0:
        rest_presence = float(np.mean((min_rate[rest_idx]) & min_count[rest_idx]))
    else:
        rest_presence = 1.0

    active_i = int(np.sum((vi >= min_presence_rate) & (ci >= min_positive_count)))
    active_j = int(np.sum((vj >= min_presence_rate) & (cj >= min_positive_count)))
    coverage_i = min(1.0, active_i / float(max(min_active_tasks, 1)))
    coverage_j = min(1.0, active_j / float(max(min_active_tasks, 1)))
    coverage_bonus = coverage_i * coverage_j

    score = focused_gap * rest_similarity * rest_presence * coverage_bonus

    return float(score)


def compute_pair_metrics(
    row_i: pd.Series,
    row_j: pd.Series,
    min_positive_count: int = DEFAULT_MIN_POS_COUNT,
) -> Dict[str, Any]:
    num_nodes_i = float(row_i["num_nodes"])
    num_nodes_j = float(row_j["num_nodes"])
    num_edges_i = float(row_i["num_edges"])
    num_edges_j = float(row_j["num_edges"])

    in_i = as_float_array(row_i["in_deg_hist_counts"])
    in_j = as_float_array(row_j["in_deg_hist_counts"])
    out_i = as_float_array(row_i["out_deg_hist_counts"])
    out_j = as_float_array(row_j["out_deg_hist_counts"])

    rate_i = as_float_array(row_i["label_rates"])
    rate_j = as_float_array(row_j["label_rates"])
    count_i = as_float_array(row_i["label_counts"])
    count_j = as_float_array(row_j["label_counts"])
    lss_i = as_float_array(row_i["labelset_size_counts"])
    lss_j = as_float_array(row_j["labelset_size_counts"])
    mix_i = as_float_array(row_i["label_mixing_counts"])
    mix_j = as_float_array(row_j["label_mixing_counts"])

    motif_i = as_float_array(row_i["motif_density"])
    motif_j = as_float_array(row_j["motif_density"])
    task_h_i = as_float_array(row_i["taskwise_homophily"])
    task_h_j = as_float_array(row_j["taskwise_homophily"])

    density_gap = abs(float(row_i["density"]) - float(row_j["density"]))
    size_ratio = max(num_nodes_i, num_nodes_j) / max(1.0, min(num_nodes_i, num_nodes_j))
    size_mismatch = abs(math.log(max(size_ratio, 1e-12)))

    edge_ratio = max(num_edges_i, num_edges_j) / max(1.0, min(num_edges_i, num_edges_j))

    in_degree_jsd = js_divergence(
        pad_to_length(in_i, max(len(in_i), len(in_j))),
        pad_to_length(in_j, max(len(in_i), len(in_j))),
    )
    out_degree_jsd = js_divergence(
        pad_to_length(out_i, max(len(out_i), len(out_j))),
        pad_to_length(out_j, max(len(out_i), len(out_j))),
    )

    motif_density_jsd = js_divergence(motif_i, motif_j)
    motif_mean_abs_gap = l1_mean_abs_gap(motif_i, motif_j)
    motif_max_gap = max_abs_gap(motif_i, motif_j)
    motif_top_gap_task, motif_top_gap_value = top_gap_task(motif_i, motif_j)

    label_rate_jsd = js_divergence(rate_i, rate_j)
    label_rate_mean_abs_gap = l1_mean_abs_gap(rate_i, rate_j)
    label_rate_max_gap = max_abs_gap(rate_i, rate_j)
    label_top_gap_task, label_top_gap_value = top_gap_task(rate_i, rate_j)

    labelset_size_jsd = js_divergence(
        pad_to_length(lss_i, max(len(lss_i), len(lss_j))),
        pad_to_length(lss_j, max(len(lss_i), len(lss_j))),
    )
    label_mixing_jsd = js_divergence(mix_i, mix_j)

    homophily_gap = abs(
        float(row_i["homophily_jaccard"]) - float(row_j["homophily_jaccard"])
    )
    task_homophily_mean_abs_gap = l1_mean_abs_gap(task_h_i, task_h_j)
    task_homophily_max_gap = max_abs_gap(task_h_i, task_h_j)
    task_h_top_gap_task, task_h_top_gap_value = top_gap_task(task_h_i, task_h_j)

    support_count_log_gap = float(
        np.mean(np.abs(np.log1p(count_i) - np.log1p(count_j)))
    )
    support_regime_gap = float(
        np.mean((count_i >= min_positive_count) != (count_j >= min_positive_count))
    )
    shared_support_frac = float(
        np.mean((count_i >= min_positive_count) & (count_j >= min_positive_count))
    )
    low_support_task_frac = float(
        np.mean((count_i < min_positive_count) | (count_j < min_positive_count))
    )
    top_k = 1
    complementarity_score = structured_complementarity_score(
        rate_i=rate_i,
        rate_j=rate_j,
        count_i=count_i,
        count_j=count_j,
        top_k=1,
        min_positive_count=min_positive_count,
        min_active_tasks=len(TASKS) - top_k,
    )
    structure_gap = float(
        np.mean([in_degree_jsd, out_degree_jsd, motif_mean_abs_gap, density_gap])
    )
    task_profile_gap = float(label_rate_mean_abs_gap)

    return {
        "graph_id_i": int(row_i["graph_id"]),
        "graph_id_j": int(row_j["graph_id"]),
        "dataset_i": str(row_i["dataset_id"]),
        "dataset_j": str(row_j["dataset_id"]),
        "type_i": row_i.get("type", None),
        "type_j": row_j.get("type", None),
        "num_nodes_i": int(num_nodes_i),
        "num_nodes_j": int(num_nodes_j),
        "num_edges_i": int(num_edges_i),
        "num_edges_j": int(num_edges_j),
        "density_gap": float(density_gap),
        "size_ratio": float(size_ratio),
        "size_mismatch": float(size_mismatch),
        "edge_ratio": float(edge_ratio),
        "in_degree_jsd": float(in_degree_jsd),
        "out_degree_jsd": float(out_degree_jsd),
        "motif_density_jsd": float(motif_density_jsd),
        "motif_mean_abs_gap": float(motif_mean_abs_gap),
        "motif_max_gap": float(motif_max_gap),
        "motif_top_gap_task": motif_top_gap_task,
        "motif_top_gap_value": float(motif_top_gap_value),
        "label_rate_jsd": float(label_rate_jsd),
        "label_rate_mean_abs_gap": float(label_rate_mean_abs_gap),
        "label_rate_max_gap": float(label_rate_max_gap),
        "label_top_gap_task": label_top_gap_task,
        "label_top_gap_value": float(label_top_gap_value),
        "labelset_size_jsd": float(labelset_size_jsd),
        "label_mixing_jsd": float(label_mixing_jsd),
        "homophily_gap": float(homophily_gap),
        "homophily_i": float(row_i["homophily_jaccard"]),
        "homophily_j": float(row_j["homophily_jaccard"]),
        "task_homophily_mean_abs_gap": float(task_homophily_mean_abs_gap),
        "task_homophily_max_gap": float(task_homophily_max_gap),
        "task_homophily_top_gap_task": task_h_top_gap_task,
        "task_homophily_top_gap_value": float(task_h_top_gap_value),
        "support_count_log_gap": float(support_count_log_gap),
        "support_regime_gap": float(support_regime_gap),
        "shared_support_frac": float(shared_support_frac),
        "low_support_task_frac": float(low_support_task_frac),
        "complementarity_score": float(complementarity_score),
        "task_profile_gap": float(task_profile_gap),
        "structure_gap": float(structure_gap),
        "label_rates_i_json": dump_json_vec(rate_i),
        "label_rates_j_json": dump_json_vec(rate_j),
        "label_counts_i_json": dump_json_vec(count_i),
        "label_counts_j_json": dump_json_vec(count_j),
        "label_gap_json": dump_json_task_gap_rows(rate_i, rate_j),
        "taskwise_homophily_i_json": dump_json_vec(task_h_i),
        "taskwise_homophily_j_json": dump_json_vec(task_h_j),
        "task_homophily_gap_json": dump_json_task_gap_rows(task_h_i, task_h_j),
    }


def build_all_pairs_df(
    feature_df: pd.DataFrame,
    min_positive_count: int = DEFAULT_MIN_POS_COUNT,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    total = math.comb(len(feature_df), 2)
    done = 0

    for i_idx, j_idx in itertools.combinations(range(len(feature_df)), 2):
        row_i = feature_df.iloc[i_idx]
        row_j = feature_df.iloc[j_idx]

        pair_id = f"p_{int(row_i['graph_id'])}_{int(row_j['graph_id'])}"
        metrics = compute_pair_metrics(
            row_i=row_i,
            row_j=row_j,
            min_positive_count=min_positive_count,
        )
        rows.append(
            {
                "subset_id": pair_id,
                "subset_size": 2,
                "subset_clients": f"{int(row_i['graph_id'])}|{int(row_j['graph_id'])}",
                **metrics,
            }
        )
        done += 1
        if done % 1000 == 0 or done == total:
            print(f"built {done}/{total} pairs")

    pair_df = pd.DataFrame(rows)
    pair_df["incompatibility_score"] = pair_df[
        [
            "task_profile_gap",
            "homophily_gap",
            "structure_gap",
            "size_mismatch",
            "low_support_task_frac",
        ]
    ].mean(axis=1)
    return pair_df


def main():
    feature_df = pd.read_parquet(f"{FEATURE_DIR}/client_features_{SPLIT}.parquet")

    all_pairs_df = build_all_pairs_df(
        feature_df=feature_df,
        min_positive_count=DEFAULT_MIN_POS_COUNT,
    )

    all_pairs_path = os.path.join(DIR, "heterogeneity_all_pairs.csv")
    all_pairs_df.to_csv(all_pairs_path, index=False)
    print("saved full pair table -> {all_pairs_path}")
    print(len(all_pairs_df))


if __name__ == "__main__":
    main()
