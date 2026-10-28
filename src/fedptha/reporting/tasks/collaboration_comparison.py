#!/usr/bin/env python3

from __future__ import annotations

from fedptha.paths import generated_root

from pathlib import Path
import json

import pandas as pd

OUT_DIR = Path(str(generated_root() / "tables"))

DIAG_SEED = OUT_DIR / "collaboration_diagnostic_vo_seed_results.csv"
MAIN_SEED = OUT_DIR / "ablation_seed_results.csv"
REP_SETS = OUT_DIR / "collaboration_specialist_representative_sets.csv"
GCFL_STRUCT = OUT_DIR / "collaboration_gcflplus_cluster_summary.csv"

OUT_PAIRED = OUT_DIR / "collaboration_diagnostic_vs_full_paired_seed_deltas.csv"
OUT_SUMMARY = OUT_DIR / "collaboration_diagnostic_vs_full_summary.csv"
OUT_REP_TRAIN = OUT_DIR / "collaboration_specialist_training_coverage.csv"

Q_ORDER = [0.20, 0.50, 0.80, 0.95]

DIAG_ORDER = [
    "oracle_physical_community",
    "specialist_rep_transfer",
    "oracle_hierarchical_taskfedavg_ala",
    "gcflplus",
]

DISPLAY = {
    "oracle_physical_community": "Oracle Physical-Community Isolation",
    "specialist_rep_transfer": "Specialist Representative Transfer",
    "oracle_hierarchical_taskfedavg_ala": "Oracle Hierarchical TaskFedAvg + ALA",
    "gcflplus": "GCFL+",
}


def abort(message):
    raise SystemExit(f"\nABORT: {message}\n")


def choose_score_column(df):
    preferred = [
        "vo_macro_pr_auc",
        "macro_pr_auc",
        "metric_value",
        "score",
        "value",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    candidates = [
        col
        for col in df.columns
        if "pr_auc" in col.lower()
        and pd.api.types.is_numeric_dtype(df[col])
        and "sd" not in col.lower()
        and "std" not in col.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    abort(
        "Could not uniquely identify the Figure-5 seed score column.\n"
        f"Candidate PR-AUC columns: {candidates}\n"
        f"All columns: {list(df.columns)}"
    )


def parse_json_paths(value):
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [Path(str(x)) for x in obj]
        if isinstance(obj, str):
            return [Path(obj)]
    except Exception:
        pass
    return []


def q_from_path(path):
    text = str(path)
    for token, q in [
        ("q20", 0.20),
        ("q50", 0.50),
        ("q80", 0.80),
        ("q95", 0.95),
    ]:
        if token in text:
            return q
    return None


print("=" * 104)
print("COLLABORATION-DIAGNOSTIC DECISION AUDIT")
print("=" * 104)

for path in [DIAG_SEED, MAIN_SEED, REP_SETS, GCFL_STRUCT]:
    if not path.is_file():
        abort(f"Missing required file: {path}")

diag = pd.read_csv(DIAG_SEED)
main = pd.read_csv(MAIN_SEED)

required_diag = {"diagnostic", "q", "seed", "vo_macro_pr_auc", "source_result_paths"}
missing = required_diag - set(diag.columns)
if missing:
    abort(f"Diagnostic seed table missing columns: {sorted(missing)}")

for col in ["setup", "method", "q", "seed", "protocol"]:
    if col not in main.columns:
        abort(f"Figure-5 seed table missing required column: {col}")

score_col = choose_score_column(main)
print("Figure-5 seed score column:", score_col)

main["q"] = pd.to_numeric(main["q"], errors="raise").round(2)
diag["q"] = pd.to_numeric(diag["q"], errors="raise").round(2)

main_vo = main[
    (main["setup"].astype(str) == "twenty_client")
    & (main["protocol"].astype(str).str.upper() == "VO")
    & main["method"].isin(["support_simplex", "support_simplex_postala"])
].copy()

main_vo[score_col] = pd.to_numeric(main_vo[score_col], errors="raise")

if len(main_vo) != 24:
    abort(f"Expected 24 main 20C VO seed rows for two methods, found {len(main_vo)}")

pivot = main_vo.pivot(
    index=["q", "seed"],
    columns="method",
    values=score_col,
).reset_index()

if len(pivot) != 12:
    abort(f"Expected 12 q x seed main-reference rows, found {len(pivot)}")

merged = diag.merge(
    pivot,
    on=["q", "seed"],
    how="left",
    validate="many_to_one",
)

if merged[["support_simplex", "support_simplex_postala"]].isna().any().any():
    abort("Missing main-method reference scores after merge")

merged["delta_vs_support_simplex_pp"] = 100.0 * (
    merged["vo_macro_pr_auc"] - merged["support_simplex"]
)
merged["delta_vs_support_simplex_postala_pp"] = 100.0 * (
    merged["vo_macro_pr_auc"] - merged["support_simplex_postala"]
)

paired_cols = [
    "diagnostic",
    "diagnostic_display",
    "q",
    "seed",
    "vo_macro_pr_auc",
    "support_simplex",
    "support_simplex_postala",
    "delta_vs_support_simplex_pp",
    "delta_vs_support_simplex_postala_pp",
    "n_eval_clients",
    "source_result_paths",
]
merged[paired_cols].sort_values(["diagnostic", "q", "seed"]).to_csv(
    OUT_PAIRED, index=False
)

summary_rows = []

for (diagnostic, display, q), g in merged.groupby(
    ["diagnostic", "diagnostic_display", "q"],
    sort=True,
):
    vals = pd.to_numeric(g["vo_macro_pr_auc"], errors="raise").to_numpy()
    d_plain = pd.to_numeric(g["delta_vs_support_simplex_pp"], errors="raise").to_numpy()
    d_full = pd.to_numeric(
        g["delta_vs_support_simplex_postala_pp"], errors="raise"
    ).to_numpy()

    summary_rows.append(
        {
            "diagnostic": diagnostic,
            "diagnostic_display": display,
            "q": q,
            "vo_macro_pr_auc_mean_percent": 100.0 * vals.mean(),
            "vo_macro_pr_auc_sample_sd_pp": 100.0 * vals.std(ddof=1),
            "delta_vs_support_simplex_mean_pp": d_plain.mean(),
            "delta_vs_support_simplex_sample_sd_pp": d_plain.std(ddof=1),
            "delta_vs_support_simplex_postala_mean_pp": d_full.mean(),
            "delta_vs_support_simplex_postala_sample_sd_pp": d_full.std(ddof=1),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUMMARY, index=False)

# ------------------------------------------------------------------
# Verify that Specialist Representative Transfer actually trains only
# the five selected representatives, while its selected VO evaluation
# covers all 20 clients (the latter was already checked upstream).
# ------------------------------------------------------------------

rep_sets_df = pd.read_csv(REP_SETS)
rep_expected = {}

for row in rep_sets_df.itertuples(index=False):
    q = round(float(row.q), 2)

    if int(row.n_unique_selected_sets) != 1:
        abort(
            f"q={q:.2f}: expected one representative set across seeds, "
            f"found {row.n_unique_selected_sets}"
        )

    text = str(row.selected_graph_ids_by_task)
    if " || " in text:
        abort(f"q={q:.2f}: more than one representative set encoded")

    reps = tuple(x for x in text.split("|") if x)
    if len(reps) != 5:
        abort(f"q={q:.2f}: representative set has {len(reps)} clients")

    rep_expected[q] = set(reps)

rep_diag = diag[diag["diagnostic"] == "specialist_rep_transfer"].copy()
coverage_rows = []

for row in rep_diag.itertuples(index=False):
    q = round(float(row.q), 2)
    seed = int(row.seed)

    paths = parse_json_paths(row.source_result_paths)
    if len(paths) != 1:
        abort(f"Representative q={q:.2f} seed={seed}: expected one source result path")

    path = paths[0]
    if not path.is_file():
        abort(f"Representative q={q:.2f} seed={seed}: missing source file {path}")

    df = pd.read_csv(path, low_memory=False)

    if "phase" not in df.columns or "graph_id" not in df.columns:
        abort(f"{path.name}: cannot audit training participants")

    phase = df["phase"].astype(str)
    train = df[phase == "apple_train_diag"].copy()

    if train.empty:
        # Fallback: identify a unique train-diagnostic phase if naming changed.
        train_like = [
            p
            for p in phase.dropna().unique().tolist()
            if "train" in str(p).lower() and "diag" in str(p).lower()
        ]
        if len(train_like) == 1:
            train = df[phase == train_like[0]].copy()
        else:
            coverage_rows.append(
                {
                    "q": q,
                    "seed": seed,
                    "expected_representatives": "|".join(sorted(rep_expected[q])),
                    "observed_training_clients": "",
                    "n_unique_training_clients": pd.NA,
                    "training_rounds_seen": pd.NA,
                    "rounds_with_exact_five_representatives": pd.NA,
                    "evaluation_clients": int(row.n_eval_clients),
                    "status": "NOT_DIRECTLY_AUDITABLE",
                    "audit_note": (
                        "Final result CSV contains no per-round training diagnostic phase; "
                        "representative selection and 20-client evaluation are verified, "
                        "but per-round training participation is not reconstructed here."
                    ),
                    "source_result_path": str(path),
                }
            )
            continue

    train["round_num"] = pd.to_numeric(train["round"], errors="coerce")
    train = train[train["round_num"].notna()].copy()

    if train.empty:
        abort(f"{path.name}: training diagnostic rows have no numeric rounds")

    unique_trainers = set(train["graph_id"].dropna().astype(str))

    per_round = train.groupby("round_num")["graph_id"].apply(
        lambda x: set(x.dropna().astype(str))
    )

    rounds_seen = len(per_round)
    exact_rounds = sum(ids == rep_expected[q] for ids in per_round)

    status = (
        "PASS"
        if unique_trainers == rep_expected[q]
        and rounds_seen == 80
        and exact_rounds == 80
        else "FAIL"
    )

    coverage_rows.append(
        {
            "q": q,
            "seed": seed,
            "expected_representatives": "|".join(sorted(rep_expected[q])),
            "observed_training_clients": "|".join(sorted(unique_trainers)),
            "n_unique_training_clients": len(unique_trainers),
            "training_rounds_seen": rounds_seen,
            "rounds_with_exact_five_representatives": exact_rounds,
            "evaluation_clients": int(row.n_eval_clients),
            "status": status,
            "source_result_path": str(path),
        }
    )

coverage = pd.DataFrame(coverage_rows).sort_values(["q", "seed"])
coverage.to_csv(OUT_REP_TRAIN, index=False)

if (coverage["status"] == "FAIL").any():
    print()
    print("Representative training-coverage failures:")
    print(coverage[coverage["status"] == "FAIL"].to_string(index=False))
    abort("Specialist representative training coverage contains a direct contradiction")

gcfl = pd.read_csv(GCFL_STRUCT)
gcfl["q"] = pd.to_numeric(gcfl["q"], errors="raise").round(2)

print()
print("=" * 104)
print("PERFORMANCE RELATIVE TO FULL SUPPORT-SIMPLEX + POSTALA")
print("(paired by seed; diagnostic minus full method, percentage points)")
print("=" * 104)

for diagnostic in DIAG_ORDER:
    part = summary[summary["diagnostic"] == diagnostic].sort_values("q")
    print()
    print(DISPLAY[diagnostic])

    for r in part.itertuples():
        print(
            f"  q={r.q:.2f}: "
            f"VO={r.vo_macro_pr_auc_mean_percent:.2f} +/- "
            f"{r.vo_macro_pr_auc_sample_sd_pp:.2f}; "
            f"delta_full={r.delta_vs_support_simplex_postala_mean_pp:+.2f} +/- "
            f"{r.delta_vs_support_simplex_postala_sample_sd_pp:.2f} pp"
        )

print()
print("=" * 104)
print("SPECIALIST REPRESENTATIVE TRAINING COVERAGE")
print("=" * 104)

for r in coverage.itertuples():
    print(
        f"  q={r.q:.2f} seed={r.seed}: "
        f"train={r.n_unique_training_clients} clients, "
        f"exact rounds={r.rounds_with_exact_five_representatives}/80, "
        f"eval={r.evaluation_clients}/20, {r.status}"
    )

print()
print("=" * 104)
print("GCFL+ STRUCTURE BY q")
print("=" * 104)

for q in Q_ORDER:
    g = gcfl[gcfl["q"] == q]

    print(
        f"  q={q:.2f}: "
        f"final clusters={int(g['final_cluster_count'].min())}-"
        f"{int(g['final_cluster_count'].max())}; "
        f"max clusters ever={int(g['max_active_cluster_count'].min())}-"
        f"{int(g['max_active_cluster_count'].max())}; "
        f"physical purity={g['partition_purity_against_physical_regions'].min():.2f}-"
        f"{g['partition_purity_against_physical_regions'].max():.2f}; "
        f"exact physical-region matches total="
        f"{int(g['exact_physical_region_matches'].sum())}"
    )

print()
print("=" * 104)
print("COMPLETE - COLLABORATION-DIAGNOSTIC DECISION AUDIT")
print("=" * 104)
print(f"Paired seed deltas      : {OUT_PAIRED}")
print(f"Decision summary        : {OUT_SUMMARY}")
print(f"Representative coverage : {OUT_REP_TRAIN}")
print()
print("No figure was created and no scientific source artifact was modified.")
