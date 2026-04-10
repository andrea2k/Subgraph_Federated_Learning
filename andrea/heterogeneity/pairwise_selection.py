import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Iterable
from dataclasses import dataclass
from andrea.heterogeneity.heterogeneity_pairwise import DIR

DEFAULT_NUM_PER_FAMILY = 1
DEFAULT_MAX_LABEL_RATE = 0.6
DEFAULT_MIN_LABEL_RATE = 0.05


@dataclass(frozen=True)
class FamilySpec:
    name: str
    maximize: Dict[str, float]
    minimize: Dict[str, float]
    constraints: Dict[str, tuple[str, float]]


def filter_pairs_all_task_rates_below_threshold(
    df: pd.DataFrame,
    max_label_rate: float = DEFAULT_MAX_LABEL_RATE,
    min_label_rate: float = DEFAULT_MIN_LABEL_RATE,
) -> pd.DataFrame:
    out = df.copy()

    def all_tasks_ok(x) -> bool:
        if pd.isna(x):
            return False
        vals = json.loads(x) if isinstance(x, str) else x
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return False
        return bool(np.all((arr >= min_label_rate) & (arr <= max_label_rate)))

    keep_mask = out["label_rates_i_json"].apply(all_tasks_ok) & out[
        "label_rates_j_json"
    ].apply(all_tasks_ok)
    return out[keep_mask].copy()


def zscore_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-12
    for col in cols:
        mu = float(out[col].mean())
        sd = float(out[col].std(ddof=0))
        out[f"{col}_z"] = (out[col] - mu) / (sd + eps)
    return out


def apply_constraints(
    df: pd.DataFrame, constraints: Dict[str, tuple[str, float]]
) -> pd.DataFrame:
    out = df.copy()
    for col, (op, value) in constraints.items():
        if op == ">=":
            out = out[out[col] >= value]
        elif op == ">":
            out = out[out[col] > value]
        elif op == "<=":
            out = out[out[col] <= value]
        elif op == "<":
            out = out[out[col] < value]
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return out


def family_score(df: pd.DataFrame, spec: FamilySpec) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    for col, weight in spec.maximize.items():
        score = score + weight * df[f"{col}_z"]
    for col, weight in spec.minimize.items():
        score = score - weight * df[f"{col}_z"]
    return score


def pick_top_unique_rows(
    df: pd.DataFrame,
    spec: FamilySpec,
    num_rows: int,
    used_subset_ids: set[str],
) -> pd.DataFrame:
    cand = apply_constraints(df, spec.constraints).copy()
    if cand.empty:
        return cand

    cand["selection_score"] = family_score(cand, spec)
    cand = cand.sort_values("selection_score", ascending=False).reset_index(drop=True)

    chosen = []
    for _, row in cand.iterrows():
        subset_id = str(row["subset_id"])
        if subset_id in used_subset_ids:
            continue
        used_subset_ids.add(subset_id)
        row = row.copy()
        row["family"] = spec.name
        row["rank_within_family"] = len(chosen) + 1
        chosen.append(row)
        if len(chosen) >= num_rows:
            break

    if not chosen:
        return pd.DataFrame(
            columns=list(cand.columns) + ["family", "rank_within_family"]
        )
    print(f"{len(chosen)} chosen for {spec.name}")
    return pd.DataFrame(chosen)


def build_family_specs() -> list[FamilySpec]:
    return [
        FamilySpec(
            name="similar",
            maximize={"shared_support_frac": 1.0},
            minimize={
                "task_profile_gap": 1.0,
                "homophily_gap": 1.0,
                "structure_gap": 1.0,
                "size_mismatch": 0.5,
            },
            constraints={"shared_support_frac": (">=", 0.60)},
        ),
        FamilySpec(
            name="task_profile_only",
            maximize={"task_profile_gap": 1.0},
            minimize={
                "homophily_gap": 1.0,
                "structure_gap": 1.0,
                "size_mismatch": 0.5,
            },
            constraints={"shared_support_frac": (">=", 0.20)},
        ),
        FamilySpec(
            name="homophily_only",
            maximize={"homophily_gap": 1.0},
            minimize={
                "task_profile_gap": 1.0,
                "structure_gap": 1.0,
                "size_mismatch": 0.5,
            },
            constraints={"shared_support_frac": (">=", 0.40)},
        ),
        FamilySpec(
            name="structure_only",
            maximize={"structure_gap": 1.0},
            minimize={
                "task_profile_gap": 1.0,
                "homophily_gap": 1.0,
                "size_mismatch": 0.5,
            },
            constraints={"shared_support_frac": (">=", 0.40)},
        ),
        FamilySpec(
            name="support_proxy_only",
            maximize={"support_regime_gap": 1.0, "low_support_task_frac": 0.2},
            minimize={
                "task_profile_gap": 1.0,
                "homophily_gap": 1.0,
                "structure_gap": 1.0,
                "size_mismatch": 0.5,
            },
            constraints={},
        ),
        FamilySpec(
            name="complementary",
            maximize={"complementarity_score": 1.2, "shared_support_frac": 0.8},
            minimize={
                "homophily_gap": 0.8,
                "structure_gap": 0.8,
                "size_mismatch": 0.5,
            },
            constraints={
                "shared_support_frac": (">=", 0.60),
                "task_profile_gap": (">=", 0.05),
            },
        ),
        FamilySpec(
            name="incompatible",
            maximize={
                "task_profile_gap": 1.0,
                "homophily_gap": 1.0,
                "structure_gap": 1.0,
                "size_mismatch": 0.5,
                "low_support_task_frac": 0.5,
            },
            minimize={"shared_support_frac": 0.5},
            constraints={},
        ),
    ]


def select_stage1_pairs(
    all_pairs_df: pd.DataFrame,
    num_per_family: int = DEFAULT_NUM_PER_FAMILY,
    max_label_rate: float = DEFAULT_MAX_LABEL_RATE,
    min_label_rate: float = DEFAULT_MIN_LABEL_RATE,
) -> pd.DataFrame:

    score_cols = [
        "shared_support_frac",
        "task_profile_gap",
        "homophily_gap",
        "structure_gap",
        "size_mismatch",
        "support_regime_gap",
        "low_support_task_frac",
        "complementarity_score",
        "incompatibility_score",
    ]

    work = zscore_cols(all_pairs_df, score_cols)

    work = filter_pairs_all_task_rates_below_threshold(
        work, max_label_rate=max_label_rate, min_label_rate=min_label_rate
    )

    families = build_family_specs()

    selected_frames = []
    used_subset_ids: set[str] = set()

    for spec in families:
        chosen = pick_top_unique_rows(
            df=work,
            spec=spec,
            num_rows=num_per_family,
            used_subset_ids=used_subset_ids,
        )
        selected_frames.append(chosen)

    out = pd.concat(selected_frames, axis=0, ignore_index=True)
    keep_cols = [
        "family",
        "rank_within_family",
        "selection_score",
        "subset_id",
        "subset_size",
        "subset_clients",
        "graph_id_i",
        "graph_id_j",
        "dataset_i",
        "dataset_j",
        "type_i",
        "type_j",
        "shared_support_frac",
        "support_regime_gap",
        "low_support_task_frac",
        "task_profile_gap",
        "structure_gap",
        "homophily_gap",
        "size_mismatch",
        "complementarity_score",
        "incompatibility_score",
        "label_top_gap_task",
        "label_top_gap_value",
        "motif_top_gap_task",
        "motif_top_gap_value",
        "task_homophily_top_gap_task",
        "task_homophily_top_gap_value",
        "label_rates_i_json",
        "label_rates_j_json",
        "label_counts_i_json",
        "label_counts_j_json",
        "label_gap_json",
        "num_nodes_i",
        "num_nodes_j",
        "num_edges_i",
        "num_edges_j",
        "homophily_i",
        "homophily_j",
        "taskwise_homophily_i_json",
        "taskwise_homophily_j_json",
        "task_homophily_gap_json",
    ]
    existing = [c for c in keep_cols if c in out.columns]
    out = out[existing].copy()
    out = out.sort_values(["family", "rank_within_family"]).reset_index(drop=True)
    return out


def main():
    all_pairs_path = os.path.join(DIR, "heterogeneity_all_pairs.csv")
    all_pairs_df = pd.read_csv(all_pairs_path)

    selected_df = select_stage1_pairs(
        all_pairs_df=all_pairs_df,
        num_per_family=DEFAULT_NUM_PER_FAMILY,
        max_label_rate=DEFAULT_MAX_LABEL_RATE,
    )
    out_path = os.path.join(DIR, "selected_pairs.csv")
    selected_df.to_csv(out_path, index=False)
    print("Saved selected similar/different pairs to:", out_path)


if __name__ == "__main__":
    main()
