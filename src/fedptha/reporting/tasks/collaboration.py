#!/usr/bin/env python3

from __future__ import annotations

from fedptha.paths import frozen_root, generated_root

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd

SUITE = Path(str(frozen_root()))
OUT_DIR = Path(str(generated_root() / "tables"))

DIR_PHYSICAL = SUITE / "additions/twenty_client/oracle_physical_community"
DIR_REP = SUITE / "additions/twenty_client/specialist_rep_transfer"
DIR_HIER = SUITE / "additions/twenty_client/oracle_hierarchical_taskfedavg_ala"
DIR_GCFL = SUITE / "main/twenty_client/gcflplus"

EXPECTED_Q = {0.20, 0.50, 0.80, 0.95}
EXPECTED_SEEDS = {0, 1, 2}
TRUE_PRAUC = "trapezoidal_precision_recall_curve_auc"

OUT_SEED = OUT_DIR / "collaboration_diagnostic_vo_seed_results.csv"
OUT_SUMMARY = OUT_DIR / "collaboration_diagnostic_vo_summary.csv"
OUT_REGIONS = OUT_DIR / "collaboration_physical_regions.csv"
OUT_REPS = OUT_DIR / "collaboration_specialist_representatives.csv"
OUT_REP_SETS = OUT_DIR / "collaboration_specialist_representative_sets.csv"
OUT_HIER = OUT_DIR / "collaboration_hierarchical_structure.csv"
OUT_GCFL_SUM = OUT_DIR / "collaboration_gcflplus_cluster_summary.csv"
OUT_GCFL_CLUSTERS = OUT_DIR / "collaboration_gcflplus_final_clusters.csv"


def abort(message: str) -> None:
    raise SystemExit(f"\nABORT: {message}\n")


def as_int_seed(values, context: str) -> int:
    xs = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().unique()
    if len(xs) != 1:
        abort(f"{context}: expected one seed, found {xs.tolist()}")
    seed = int(round(float(xs[0])))
    if seed not in EXPECTED_SEEDS:
        abort(f"{context}: unexpected seed {seed}")
    return seed


def q_from_dataset_ids(values, context: str) -> float:
    tokens = set()
    for value in values:
        if pd.isna(value):
            continue
        for hit in re.findall(r"q(20|50|80|95)", str(value), flags=re.I):
            tokens.add(hit)

    if len(tokens) != 1:
        abort(f"{context}: expected one q token in dataset_id, found {sorted(tokens)}")

    q = {
        "20": 0.20,
        "50": 0.50,
        "80": 0.80,
        "95": 0.95,
    }[next(iter(tokens))]

    if q not in EXPECTED_Q:
        abort(f"{context}: unexpected q={q}")

    return q


def split_members(value) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()
    return tuple(x.strip() for x in str(value).split("|") if x.strip())


def numeric_graph_sort_key(graph_id: str):
    try:
        return int(graph_id)
    except Exception:
        return graph_id


def main_result_files(directory: Path, family: str) -> list[Path]:
    files = sorted(directory.glob("*.csv"))

    def keep(path: Path) -> bool:
        name = path.name
        return not (
            name.endswith("_dr.csv")
            or name.endswith("_representatives.csv")
            or name.endswith("_hierarchy_weights.csv")
            or name.endswith("_inter_ala.csv")
            or name.endswith("_clusters.csv")
        )

    kept = [p for p in files if keep(p)]

    expected = {
        "physical": 48,
        "representative": 12,
        "hierarchical": 12,
        "gcflplus": 12,
    }[family]

    if len(kept) != expected:
        abort(
            f"{family}: expected {expected} main result CSVs, found {len(kept)} "
            f"in {directory}"
        )

    return kept


def read_vo_clients(path: Path, expected_n_clients: int):
    df = pd.read_csv(path, low_memory=False)
    context = path.name

    required = {
        "phase",
        "graph_id",
        "dataset_id",
        "seed",
        "split",
        "eval_mask_mode",
        "selection_protocol",
        "macro_pr_auc",
        "pr_auc_definition",
        "subset_clients",
    }
    missing = required - set(df.columns)
    if missing:
        abort(f"{context}: missing required columns {sorted(missing)}")

    defs = set(
        df["pr_auc_definition"].dropna().astype(str).str.strip().unique().tolist()
    )
    if defs != {TRUE_PRAUC}:
        abort(f"{context}: unexpected PR-AUC definition(s): {sorted(defs)}")

    q = q_from_dataset_ids(df["dataset_id"], context)
    seed = as_int_seed(df["seed"].dropna().unique(), context)

    phase = df["phase"].astype(str)
    mask = (
        phase.str.contains("best_realistic_selection_oracle", na=False)
        & ~phase.str.contains("_task", na=False)
        & ~phase.str.endswith("_mean", na=False)
        & (df["split"].astype(str) == "test")
        & (df["eval_mask_mode"].astype(str) == "full")
        & (df["selection_protocol"].astype(str) == "realistic_selection_oracle")
    )

    vo = df.loc[mask].copy()
    vo["macro_pr_auc"] = pd.to_numeric(vo["macro_pr_auc"], errors="coerce")
    vo = vo[vo["macro_pr_auc"].notna()].copy()

    if len(vo) != expected_n_clients:
        abort(
            f"{context}: expected {expected_n_clients} VO client rows, "
            f"found {len(vo)}"
        )

    graph_ids = vo["graph_id"].astype(str).tolist()
    if len(set(graph_ids)) != expected_n_clients:
        abort(
            f"{context}: expected {expected_n_clients} unique VO graph_ids, "
            f"found {len(set(graph_ids))}"
        )

    mean_mask = (
        phase.str.contains("best_realistic_selection_oracle", na=False)
        & phase.str.endswith("_mean", na=False)
        & (df["selection_protocol"].astype(str) == "realistic_selection_oracle")
    )
    mean_rows = df.loc[mean_mask].copy()

    stored_mean = None
    if len(mean_rows) == 1:
        x = pd.to_numeric(mean_rows["macro_pr_auc"], errors="coerce").iloc[0]
        if pd.notna(x):
            stored_mean = float(x)
            calc = float(vo["macro_pr_auc"].mean())
            if not math.isclose(calc, stored_mean, rel_tol=0.0, abs_tol=5e-10):
                abort(
                    f"{context}: client mean {calc:.12f} != stored mean "
                    f"{stored_mean:.12f}"
                )

    subset_sets = {
        split_members(v) for v in df["subset_clients"].dropna().astype(str).unique()
    }
    subset_sets = {s for s in subset_sets if s}

    return {
        "df": df,
        "vo": vo,
        "q": q,
        "seed": seed,
        "score": float(vo["macro_pr_auc"].mean()),
        "graph_ids": tuple(sorted(set(graph_ids), key=numeric_graph_sort_key)),
        "subset_sets": subset_sets,
        "stored_mean": stored_mean,
    }


print("=" * 108)
print("COLLABORATION-STRUCTURE DIAGNOSTICS: FINAL VO + STRUCTURE AUDIT")
print("=" * 108)

# ---------------------------------------------------------------------
# A. Oracle physical-community isolation
#
# IMPORTANT: the 20-client benchmark uses different client-ID blocks
# at different q values. Therefore there are 16 distinct 5-client
# subsets globally (4 communities x 4 q values), not 4 globally.
# The invariant is exactly FOUR physical communities WITHIN EACH q.
# ---------------------------------------------------------------------

physical_files = main_result_files(DIR_PHYSICAL, "physical")
physical_runs = []

for path in physical_files:
    run = read_vo_clients(path, expected_n_clients=5)

    if len(run["subset_sets"]) != 1:
        abort(
            f"{path.name}: expected one 5-client physical subset, "
            f"found {len(run['subset_sets'])}"
        )

    region = next(iter(run["subset_sets"]))
    if len(region) != 5:
        abort(f"{path.name}: physical subset has {len(region)} clients, expected 5")

    if set(run["graph_ids"]) != set(region):
        abort(f"{path.name}: VO graph IDs do not match physical subset_clients")

    physical_runs.append(
        {
            "path": path,
            "q": run["q"],
            "seed": run["seed"],
            "region_members": region,
            "score": run["score"],
        }
    )

physical_runs_df = pd.DataFrame(physical_runs)

regions_by_q = {}
graph_to_region_by_q = {}
region_rows = []

for q in sorted(EXPECTED_Q):
    qpart = physical_runs_df[physical_runs_df["q"] == q].copy()

    if len(qpart) != 12:
        abort(f"Physical isolation q={q}: expected 12 files, found {len(qpart)}")

    all_regions_q = set(qpart["region_members"].tolist())
    if len(all_regions_q) != 4:
        abort(
            f"Physical isolation q={q}: expected exactly 4 physical communities, "
            f"found {len(all_regions_q)}"
        )

    regions_sorted = sorted(
        all_regions_q,
        key=lambda members: min(numeric_graph_sort_key(x) for x in members),
    )

    # Every seed must contain exactly the same four communities at this q.
    for seed in sorted(EXPECTED_SEEDS):
        spart = qpart[qpart["seed"] == seed]
        if len(spart) != 4:
            abort(
                f"Physical isolation q={q} seed={seed}: expected 4 community runs, "
                f"found {len(spart)}"
            )
        if set(spart["region_members"].tolist()) != set(regions_sorted):
            abort(f"Physical isolation q={q} seed={seed}: physical partition mismatch")

    regions_by_q[q] = regions_sorted

    mapping = {}
    for region_id, members in enumerate(regions_sorted):
        for graph_id in sorted(members, key=numeric_graph_sort_key):
            if graph_id in mapping:
                abort(
                    f"Physical isolation q={q}: client {graph_id} is in multiple regions"
                )
            mapping[graph_id] = region_id
            region_rows.append(
                {
                    "q": q,
                    "physical_region": region_id,
                    "graph_id": graph_id,
                }
            )

    if len(mapping) != 20:
        abort(
            f"Physical isolation q={q}: regions cover {len(mapping)} clients, expected 20"
        )

    graph_to_region_by_q[q] = mapping

physical_seed_rows = []

for (q, seed), group in physical_runs_df.groupby(["q", "seed"], sort=True):
    if len(group) != 4:
        abort(f"Physical isolation q={q} seed={seed}: expected 4 community runs")

    # All four communities have size 5, so the mean of four community means
    # is exactly the 20-client macro mean.
    combined = float(group["score"].mean())

    physical_seed_rows.append(
        {
            "setup": "twenty_client",
            "diagnostic": "oracle_physical_community",
            "diagnostic_display": "Oracle Physical-Community Isolation",
            "q": q,
            "seed": int(seed),
            "n_eval_clients": 20,
            "vo_macro_pr_auc": combined,
            "vo_macro_pr_auc_percent": 100.0 * combined,
            "source_result_paths": json.dumps(sorted(str(p) for p in group["path"])),
        }
    )

# ---------------------------------------------------------------------
# B. Specialist representative transfer
# ---------------------------------------------------------------------

rep_result_files = main_result_files(DIR_REP, "representative")
rep_seed_rows = []

for path in rep_result_files:
    run = read_vo_clients(path, expected_n_clients=20)

    if len(run["subset_sets"]) != 1:
        abort(f"{path.name}: representative-transfer subset metadata ambiguous")

    subset = next(iter(run["subset_sets"]))
    if len(subset) != 20:
        abort(f"{path.name}: representative-transfer subset has {len(subset)} clients")

    if set(run["graph_ids"]) != set(subset):
        abort(f"{path.name}: representative transfer did not evaluate its full subset")

    expected_q_clients = set(graph_to_region_by_q[run["q"]])
    if set(subset) != expected_q_clients:
        abort(
            f"{path.name}: representative-transfer 20-client subset does not match "
            f"the q={run['q']:.2f} benchmark client set"
        )

    rep_seed_rows.append(
        {
            "setup": "twenty_client",
            "diagnostic": "specialist_rep_transfer",
            "diagnostic_display": "Specialist Representative Transfer",
            "q": run["q"],
            "seed": run["seed"],
            "n_eval_clients": 20,
            "vo_macro_pr_auc": run["score"],
            "vo_macro_pr_auc_percent": 100.0 * run["score"],
            "source_result_paths": json.dumps([str(path)]),
        }
    )

rep_sidecars = sorted(DIR_REP.glob("*_representatives.csv"))
if len(rep_sidecars) != 12:
    abort(f"Expected 12 representative sidecars, found {len(rep_sidecars)}")

rep_rows = []
rep_set_records = []

for path in rep_sidecars:
    df = pd.read_csv(path, low_memory=False)

    required = {
        "task_idx",
        "candidate_graph_id",
        "candidate_dataset_id",
        "candidate_physical_community",
        "visible_train_positive_support",
        "is_selected_representative",
    }
    missing = required - set(df.columns)
    if missing:
        abort(f"{path.name}: representative sidecar missing {sorted(missing)}")

    q = q_from_dataset_ids(df["candidate_dataset_id"], path.name)

    selected_flag = pd.to_numeric(
        df["is_selected_representative"], errors="coerce"
    ).fillna(0)
    selected = df[selected_flag == 1].copy()

    if len(selected) != 5:
        abort(
            f"{path.name}: expected 5 selected representatives, found {len(selected)}"
        )

    task_ids = sorted(
        pd.to_numeric(selected["task_idx"], errors="raise").astype(int).tolist()
    )
    if task_ids != [0, 1, 2, 3, 4]:
        abort(f"{path.name}: expected one representative per task 0..4, got {task_ids}")

    selected_graphs = tuple(
        selected.sort_values("task_idx")["candidate_graph_id"].astype(str).tolist()
    )

    expected_q_clients = set(graph_to_region_by_q[q])
    if not set(selected_graphs).issubset(expected_q_clients):
        abort(f"{path.name}: selected representative outside q={q:.2f} client set")

    for _, row in selected.sort_values("task_idx").iterrows():
        graph_id = str(row["candidate_graph_id"])
        sidecar_region = int(row["candidate_physical_community"])
        derived_region = graph_to_region_by_q[q][graph_id]

        # We record both. If labels are aligned, they should match exactly.
        rep_rows.append(
            {
                "q": q,
                "sidecar": str(path),
                "task_idx": int(row["task_idx"]),
                "representative_graph_id": graph_id,
                "sidecar_physical_region": sidecar_region,
                "derived_physical_region": derived_region,
                "region_label_matches": sidecar_region == derived_region,
                "visible_train_positive_support": float(
                    row["visible_train_positive_support"]
                ),
            }
        )

    rep_set_records.append(
        {
            "q": q,
            "sidecar": str(path),
            "selected_graph_ids_by_task": "|".join(selected_graphs),
        }
    )

rep_sets_df = pd.DataFrame(rep_set_records)
rep_set_summary_rows = []

for q, group in rep_sets_df.groupby("q", sort=True):
    if len(group) != 3:
        abort(f"Representative sidecars q={q}: expected 3 files, found {len(group)}")

    sets = sorted(group["selected_graph_ids_by_task"].unique().tolist())

    rep_set_summary_rows.append(
        {
            "q": q,
            "n_sidecars": len(group),
            "n_unique_selected_sets": len(sets),
            "selected_graph_ids_by_task": " || ".join(sets),
        }
    )

rep_rows_df = pd.DataFrame(rep_rows)
if not rep_rows_df["region_label_matches"].all():
    bad = rep_rows_df[~rep_rows_df["region_label_matches"]]
    abort(
        "Representative sidecar physical-community labels do not match "
        f"the oracle physical partitions. First mismatches:\n{bad.head().to_string(index=False)}"
    )

# ---------------------------------------------------------------------
# C. Oracle hierarchical TaskFedAvg + ALA
# ---------------------------------------------------------------------

hier_files = main_result_files(DIR_HIER, "hierarchical")
hier_seed_rows = []
hier_structure_rows = []

for path in hier_files:
    run = read_vo_clients(path, expected_n_clients=20)
    vo = run["vo"].copy()

    expected_q_clients = set(graph_to_region_by_q[run["q"]])
    if set(run["graph_ids"]) != expected_q_clients:
        abort(f"{path.name}: hierarchical result client set does not match q benchmark")

    if "oracle_hard_cluster_id" not in vo.columns:
        abort(f"{path.name}: missing oracle_hard_cluster_id")

    cluster_ids = pd.to_numeric(vo["oracle_hard_cluster_id"], errors="coerce")
    if cluster_ids.isna().any():
        abort(f"{path.name}: missing oracle_hard_cluster_id on VO client row")

    vo["_cluster_id"] = cluster_ids.astype(int)

    cluster_groups = {}
    for cluster_id, g in vo.groupby("_cluster_id", sort=True):
        members = tuple(
            sorted(g["graph_id"].astype(str).tolist(), key=numeric_graph_sort_key)
        )
        cluster_groups[int(cluster_id)] = members

        expected_same_id = tuple(
            sorted(
                [
                    graph_id
                    for graph_id, rid in graph_to_region_by_q[run["q"]].items()
                    if rid == int(cluster_id)
                ],
                key=numeric_graph_sort_key,
            )
        )

        hier_structure_rows.append(
            {
                "q": run["q"],
                "seed": run["seed"],
                "oracle_cluster_id": int(cluster_id),
                "cluster_size": len(members),
                "members": "|".join(members),
                "matches_physical_region_same_id": set(members)
                == set(expected_same_id),
                "source_result_path": str(path),
            }
        )

    if set(cluster_groups) != {0, 1, 2, 3}:
        abort(
            f"{path.name}: expected hierarchical clusters 0,1,2,3; "
            f"found {sorted(cluster_groups)}"
        )

    if any(len(v) != 5 for v in cluster_groups.values()):
        abort(f"{path.name}: hierarchical clusters are not all size 5")

    physical_sets = {frozenset(x) for x in regions_by_q[run["q"]]}
    hierarchical_sets = {frozenset(x) for x in cluster_groups.values()}
    if hierarchical_sets != physical_sets:
        abort(
            f"{path.name}: hierarchical oracle partition does not match "
            f"the physical communities at q={run['q']:.2f}"
        )

    intra = set(
        run["df"]
        .get("intra_method", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    inter = set(
        run["df"]
        .get("inter_method", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if intra and intra != {"oracle_physical_community_support_simplex_postala"}:
        abort(f"{path.name}: unexpected intra_method {sorted(intra)}")

    if inter and inter != {"leave_one_cluster_out_support_weighted_task_fedavg"}:
        abort(f"{path.name}: unexpected inter_method {sorted(inter)}")

    hier_seed_rows.append(
        {
            "setup": "twenty_client",
            "diagnostic": "oracle_hierarchical_taskfedavg_ala",
            "diagnostic_display": "Oracle Hierarchical TaskFedAvg + ALA",
            "q": run["q"],
            "seed": run["seed"],
            "n_eval_clients": 20,
            "vo_macro_pr_auc": run["score"],
            "vo_macro_pr_auc_percent": 100.0 * run["score"],
            "source_result_paths": json.dumps([str(path)]),
        }
    )

hier_structure_df = pd.DataFrame(hier_structure_rows)

# Same-ID labels should normally align too; if not, the partition is still
# structurally the same but labels were permuted. Record rather than abort.
same_id_label_alignment = bool(
    hier_structure_df["matches_physical_region_same_id"].all()
)

# ---------------------------------------------------------------------
# D. GCFL+ performance and cluster structure
# ---------------------------------------------------------------------

gcfl_files = main_result_files(DIR_GCFL, "gcflplus")
gcfl_seed_rows = []
gcfl_run_lookup = {}

for path in gcfl_files:
    run = read_vo_clients(path, expected_n_clients=20)

    expected_q_clients = set(graph_to_region_by_q[run["q"]])
    if set(run["graph_ids"]) != expected_q_clients:
        abort(f"{path.name}: GCFL+ client set does not match q benchmark")

    gcfl_seed_rows.append(
        {
            "setup": "twenty_client",
            "diagnostic": "gcflplus",
            "diagnostic_display": "GCFL+",
            "q": run["q"],
            "seed": run["seed"],
            "n_eval_clients": 20,
            "vo_macro_pr_auc": run["score"],
            "vo_macro_pr_auc_percent": 100.0 * run["score"],
            "source_result_paths": json.dumps([str(path)]),
        }
    )

    gcfl_run_lookup[path.stem] = (run["q"], run["seed"])

cluster_sidecars = sorted(DIR_GCFL.glob("*_clusters.csv"))
if len(cluster_sidecars) != 12:
    abort(f"Expected 12 GCFL+ cluster sidecars, found {len(cluster_sidecars)}")

gcfl_summary_rows = []
gcfl_cluster_rows = []

for path in cluster_sidecars:
    base_stem = path.stem.removesuffix("_clusters")
    if base_stem not in gcfl_run_lookup:
        abort(f"{path.name}: cannot match cluster sidecar to main GCFL+ result")

    q, seed = gcfl_run_lookup[base_stem]
    df = pd.read_csv(path, low_memory=False)

    required = {"round", "cluster_id", "cluster_size", "members"}
    missing = required - set(df.columns)
    if missing:
        abort(f"{path.name}: cluster sidecar missing {sorted(missing)}")

    df["_round_num"] = pd.to_numeric(df["round"], errors="coerce")
    if df["_round_num"].isna().any():
        abort(f"{path.name}: nonnumeric round in cluster sidecar")

    min_round = int(df["_round_num"].min())
    max_round = int(df["_round_num"].max())

    if min_round != 1 or max_round != 80:
        abort(f"{path.name}: expected rounds 1..80, observed {min_round}..{max_round}")

    counts = df.groupby("_round_num").size()
    max_active_clusters = int(counts.max())
    final = df[df["_round_num"] == max_round].copy()
    final_cluster_count = len(final)

    final_member_sets = []
    exact_region_matches = 0
    weighted_best_overlap = 0
    physical_regions = regions_by_q[q]
    physical_region_sets = {frozenset(x) for x in physical_regions}

    for _, row in final.iterrows():
        members = split_members(row["members"])
        size = int(float(row["cluster_size"]))

        if len(members) != size:
            abort(
                f"{path.name}: cluster {row['cluster_id']} size metadata {size} "
                f"!= parsed members {len(members)}"
            )

        final_member_sets.append(set(members))

        overlaps = [len(set(members) & set(region)) for region in physical_regions]

        best_overlap = max(overlaps) if overlaps else 0
        best_regions = [i for i, x in enumerate(overlaps) if x == best_overlap]

        if frozenset(members) in physical_region_sets:
            exact_region_matches += 1

        weighted_best_overlap += best_overlap

        gcfl_cluster_rows.append(
            {
                "q": q,
                "seed": seed,
                "final_round": max_round,
                "cluster_id": int(float(row["cluster_id"])),
                "cluster_size": size,
                "members": "|".join(members),
                "overlap_physical_region_0": overlaps[0],
                "overlap_physical_region_1": overlaps[1],
                "overlap_physical_region_2": overlaps[2],
                "overlap_physical_region_3": overlaps[3],
                "best_physical_region_overlap": best_overlap,
                "best_physical_region_ids": "|".join(map(str, best_regions)),
                "cluster_purity_against_physical_regions": (
                    best_overlap / size if size else np.nan
                ),
                "is_exact_physical_region": frozenset(members) in physical_region_sets,
                "source_cluster_sidecar": str(path),
            }
        )

    union = set().union(*final_member_sets) if final_member_sets else set()
    total_members = sum(len(s) for s in final_member_sets)

    expected_q_clients = set(graph_to_region_by_q[q])

    if union != expected_q_clients or total_members != 20:
        abort(
            f"{path.name}: final GCFL+ clusters do not form a disjoint partition "
            f"of the 20 q={q:.2f} clients"
        )

    partition_purity = weighted_best_overlap / 20.0

    gcfl_summary_rows.append(
        {
            "q": q,
            "seed": seed,
            "first_round": min_round,
            "final_round": max_round,
            "max_active_cluster_count": max_active_clusters,
            "final_cluster_count": final_cluster_count,
            "ever_split": max_active_clusters > 1,
            "final_cluster_sizes": "|".join(
                str(int(float(x)))
                for x in final.sort_values("cluster_id")["cluster_size"].tolist()
            ),
            "exact_physical_region_matches": exact_region_matches,
            "partition_purity_against_physical_regions": partition_purity,
            "source_cluster_sidecar": str(path),
        }
    )

# ---------------------------------------------------------------------
# E. Grid checks and seed summaries
# ---------------------------------------------------------------------

seed_rows = physical_seed_rows + rep_seed_rows + hier_seed_rows + gcfl_seed_rows

seed_df = pd.DataFrame(seed_rows)

diagnostics = {
    "oracle_physical_community",
    "specialist_rep_transfer",
    "oracle_hierarchical_taskfedavg_ala",
    "gcflplus",
}

expected_keys = {(q, s) for q in EXPECTED_Q for s in EXPECTED_SEEDS}

for diagnostic in diagnostics:
    part = seed_df[seed_df["diagnostic"] == diagnostic]
    keys = {(round(float(r.q), 2), int(r.seed)) for r in part.itertuples()}

    if keys != expected_keys:
        abort(
            f"{diagnostic}: q x seed coverage mismatch; "
            f"missing={sorted(expected_keys - keys)}, "
            f"extra={sorted(keys - expected_keys)}"
        )

summary_rows = []

for (diagnostic, display, q), group in seed_df.groupby(
    ["diagnostic", "diagnostic_display", "q"],
    sort=True,
):
    values = group.sort_values("seed")["vo_macro_pr_auc"].to_numpy(dtype=float)

    if len(values) != 3:
        abort(f"{diagnostic} q={q}: expected 3 seeds, found {len(values)}")

    summary_rows.append(
        {
            "setup": "twenty_client",
            "diagnostic": diagnostic,
            "diagnostic_display": display,
            "q": q,
            "n_seeds": 3,
            "vo_macro_pr_auc_mean": float(values.mean()),
            "vo_macro_pr_auc_sample_sd": float(values.std(ddof=1)),
            "vo_macro_pr_auc_mean_percent": float(100.0 * values.mean()),
            "vo_macro_pr_auc_sample_sd_pp": float(100.0 * values.std(ddof=1)),
            "seed_values_json": json.dumps([float(x) for x in values]),
        }
    )

summary_df = pd.DataFrame(summary_rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)

seed_df.sort_values(["diagnostic", "q", "seed"]).to_csv(OUT_SEED, index=False)
summary_df.sort_values(["diagnostic", "q"]).to_csv(OUT_SUMMARY, index=False)
pd.DataFrame(region_rows).sort_values(["q", "physical_region", "graph_id"]).to_csv(
    OUT_REGIONS, index=False
)
rep_rows_df.sort_values(["q", "sidecar", "task_idx"]).to_csv(OUT_REPS, index=False)
pd.DataFrame(rep_set_summary_rows).sort_values(["q"]).to_csv(OUT_REP_SETS, index=False)
hier_structure_df.sort_values(["q", "seed", "oracle_cluster_id"]).to_csv(
    OUT_HIER, index=False
)
pd.DataFrame(gcfl_summary_rows).sort_values(["q", "seed"]).to_csv(
    OUT_GCFL_SUM, index=False
)
pd.DataFrame(gcfl_cluster_rows).sort_values(["q", "seed", "cluster_id"]).to_csv(
    OUT_GCFL_CLUSTERS, index=False
)

# ---------------------------------------------------------------------
# F. Compact report
# ---------------------------------------------------------------------

print()
print("=" * 108)
print("VO MACRO PR-AUC SUMMARY (mean +/- sample SD over seeds; %)")
print("=" * 108)

display_order = [
    "oracle_physical_community",
    "specialist_rep_transfer",
    "oracle_hierarchical_taskfedavg_ala",
    "gcflplus",
]

for diagnostic in display_order:
    part = summary_df[summary_df["diagnostic"] == diagnostic].sort_values("q")
    display = part["diagnostic_display"].iloc[0]

    print()
    print(display)

    for row in part.itertuples():
        print(
            f"  q={row.q:.2f}: "
            f"{row.vo_macro_pr_auc_mean_percent:.2f} +/- "
            f"{row.vo_macro_pr_auc_sample_sd_pp:.2f}"
        )

print()
print("=" * 108)
print("PHYSICAL COMMUNITIES (q-specific client IDs)")
print("=" * 108)

for q in sorted(EXPECTED_Q):
    print(f"  q={q:.2f}")
    for region_id, members in enumerate(regions_by_q[q]):
        print(f"    region {region_id}: {'|'.join(members)}")

print()
print("=" * 108)
print("SPECIALIST REPRESENTATIVES")
print("=" * 108)

rep_set_summary_df = pd.DataFrame(rep_set_summary_rows).sort_values("q")
for row in rep_set_summary_df.itertuples():
    print(
        f"  q={row.q:.2f}: unique sets={row.n_unique_selected_sets}; "
        f"{row.selected_graph_ids_by_task}"
    )

print("  Evaluation coverage: 20/20 clients for every q x seed run.")
print("  Sidecar physical-region labels match oracle physical partitions: PASS.")

print()
print("=" * 108)
print("HIERARCHICAL ORACLE STRUCTURE")
print("=" * 108)
print("  4 clusters x 5 clients in every q x seed run.")
print("  Cluster partition matches the four q-specific physical communities: PASS.")
print(
    "  Same-ID region labels align: "
    + (
        "PASS"
        if same_id_label_alignment
        else "LABEL-PERMUTED (partition still matches)"
    )
)
print("  Intra method: oracle_physical_community_support_simplex_postala")
print("  Inter method: leave_one_cluster_out_support_weighted_task_fedavg")

print()
print("=" * 108)
print("GCFL+ CLUSTER STRUCTURE")
print("=" * 108)

gcfl_sum_df = pd.DataFrame(gcfl_summary_rows).sort_values(["q", "seed"])

for row in gcfl_sum_df.itertuples():
    print(
        f"  q={row.q:.2f} seed={row.seed}: "
        f"max_clusters={row.max_active_cluster_count}, "
        f"final_clusters={row.final_cluster_count}, "
        f"final_sizes={row.final_cluster_sizes}, "
        f"exact_region_matches={row.exact_physical_region_matches}, "
        f"physical_partition_purity={row.partition_purity_against_physical_regions:.3f}"
    )

print()
print("=" * 108)
print("PASS - COLLABORATION-STRUCTURE DIAGNOSTIC AUDIT")
print("=" * 108)
print(f"Seed results          : {OUT_SEED}")
print(f"VO summary            : {OUT_SUMMARY}")
print(f"Physical regions      : {OUT_REGIONS}")
print(f"Representatives       : {OUT_REPS}")
print(f"Representative sets   : {OUT_REP_SETS}")
print(f"Hierarchy structure   : {OUT_HIER}")
print(f"GCFL+ cluster summary : {OUT_GCFL_SUM}")
print(f"GCFL+ final clusters  : {OUT_GCFL_CLUSTERS}")
print()
print(
    "No model, raw result, frozen summary, checkpoint, or source artifact was modified."
)
