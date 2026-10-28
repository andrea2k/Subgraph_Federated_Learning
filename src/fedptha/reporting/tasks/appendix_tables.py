from fedptha.paths import generated_root

#!/usr/bin/env python3

from pathlib import Path
import math
import pandas as pd

ROOT = Path(str(generated_root()))
OUT = ROOT / "tables"
TABLES = ROOT / "tables"

DIAG_SEED = OUT / "collaboration_diagnostic_vo_seed_results.csv"
DIAG_SUMMARY = OUT / "collaboration_diagnostic_vo_summary.csv"
FIG5_SEED = OUT / "ablation_seed_results.csv"
GCFL_FINAL = OUT / "collaboration_gcflplus_final_clusters.csv"

MAIN_TEX = TABLES / "table_collaboration_diagnostics_20c.tex"
GCFL_TEX = TABLES / "table_gcflplus_clusters_appendix_d.tex"
AUDIT_TXT = OUT / "collaboration_tables_audit.txt"

Q_ORDER = [0.20, 0.50, 0.80, 0.95]
SEEDS = [0, 1, 2]

ROWS = [
    (
        "support_simplex_postala",
        "Support-Simplex + PostALA",
        "main",
    ),
    (
        "specialist_rep_transfer",
        "Task-Specialist Representatives",
        "diagnostic",
    ),
    (
        "oracle_physical_community",
        "Oracle Physical-Community Isolation",
        "diagnostic",
    ),
    (
        "oracle_hierarchical_taskfedavg_ala",
        "Oracle Hierarchical Collaboration + ALA",
        "diagnostic",
    ),
]


def abort(message):
    raise SystemExit(f"\nABORT: {message}\n")


def fmt(mean, sd):
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * sd:.2f}"


def pick_score_column(df):
    for col in [
        "macro_pr_auc",
        "vo_macro_pr_auc",
        "score",
        "value",
    ]:
        if col in df.columns:
            return col

    candidates = [
        col
        for col in df.columns
        if "pr_auc" in col.lower()
        and "mean" not in col.lower()
        and "sd" not in col.lower()
        and "std" not in col.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    abort(
        "Could not identify Figure-5 seed score column.\n"
        f"Columns: {list(df.columns)}"
    )


def parse_members(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]


for path in [
    DIAG_SEED,
    FIG5_SEED,
    GCFL_FINAL,
]:
    if not path.is_file():
        abort(f"Missing required input: {path}")

TABLES.mkdir(parents=True, exist_ok=True)

diag = pd.read_csv(DIAG_SEED)
fig5 = pd.read_csv(FIG5_SEED)
gcfl = pd.read_csv(GCFL_FINAL)

# ================================================================
# 1. MAIN-TEXT COLLABORATION TABLE
# ================================================================

required_diag = {
    "diagnostic",
    "q",
    "seed",
    "vo_macro_pr_auc",
}

missing = required_diag - set(diag.columns)
if missing:
    abort("Diagnostic seed table missing columns: " f"{sorted(missing)}")

required_fig5 = {
    "setup",
    "method",
    "q",
    "seed",
    "protocol",
}

missing = required_fig5 - set(fig5.columns)
if missing:
    abort("Figure-5 seed table missing columns: " f"{sorted(missing)}")

score_col = pick_score_column(fig5)

diag["q"] = pd.to_numeric(
    diag["q"],
    errors="raise",
).round(2)

diag["seed"] = pd.to_numeric(
    diag["seed"],
    errors="raise",
).astype(int)

diag["vo_macro_pr_auc"] = pd.to_numeric(
    diag["vo_macro_pr_auc"],
    errors="raise",
)

fig5["q"] = pd.to_numeric(
    fig5["q"],
    errors="raise",
).round(2)

fig5["seed"] = pd.to_numeric(
    fig5["seed"],
    errors="raise",
).astype(int)

fig5[score_col] = pd.to_numeric(
    fig5[score_col],
    errors="raise",
)

main_source = fig5[
    (fig5["setup"].astype(str) == "twenty_client")
    & (fig5["protocol"].astype(str).str.upper() == "VO")
    & (fig5["method"].astype(str) == "support_simplex_postala")
][
    [
        "q",
        "seed",
        score_col,
    ]
].copy()

main_source = main_source.rename(columns={score_col: "score"})

expected_keys = {(q, seed) for q in Q_ORDER for seed in SEEDS}

main_keys = {
    (float(row.q), int(row.seed)) for row in main_source.itertuples(index=False)
}

if main_keys != expected_keys or len(main_source) != 12:
    abort("Support-Simplex + PostALA " "q x seed coverage mismatch")

summary = {}

audit = [
    "COLLABORATION TABLES AUDIT",
    "=" * 88,
    f"Figure-5 seed score column: {score_col}",
]

for method_key, display, source_kind in ROWS:

    if source_kind == "main":
        source = main_source.copy()

    else:
        source = diag[diag["diagnostic"].astype(str) == method_key][
            [
                "q",
                "seed",
                "vo_macro_pr_auc",
            ]
        ].copy()

        source = source.rename(columns={"vo_macro_pr_auc": "score"})

    keys = {(float(row.q), int(row.seed)) for row in source.itertuples(index=False)}

    if keys != expected_keys or len(source) != 12:
        abort(f"{display}: q x seed coverage mismatch")

    summary[method_key] = {}

    for q in Q_ORDER:

        values = (
            source[source["q"] == q].sort_values("seed")["score"].to_numpy(dtype=float)
        )

        if len(values) != 3:
            abort(f"{display} q={q:.2f}: " f"expected 3 seeds, found {len(values)}")

        mean = float(values.mean())
        sd = float(values.std(ddof=1))

        summary[method_key][q] = (
            mean,
            sd,
        )

        audit.append(
            f"{display} q={q:.2f}: " f"{100.0 * mean:.2f} +/- " f"{100.0 * sd:.2f}"
        )

# Cross-check diagnostic aggregates against the
# already-audited collaboration summary.

if not DIAG_SUMMARY.is_file():
    abort(f"Missing audited diagnostic summary: " f"{DIAG_SUMMARY}")

frozen = pd.read_csv(DIAG_SUMMARY)

required_frozen = {
    "diagnostic",
    "q",
    "vo_macro_pr_auc_mean",
    "vo_macro_pr_auc_sample_sd",
}

missing = required_frozen - set(frozen.columns)

if missing:
    abort("Diagnostic summary missing columns: " f"{sorted(missing)}")

frozen["q"] = pd.to_numeric(
    frozen["q"],
    errors="raise",
).round(2)

for method_key, display, source_kind in ROWS:

    if source_kind != "diagnostic":
        continue

    for q in Q_ORDER:

        row = frozen[
            (frozen["diagnostic"].astype(str) == method_key) & (frozen["q"] == q)
        ]

        if len(row) != 1:
            abort(
                f"{display} q={q:.2f}: "
                "audited summary row count "
                f"is {len(row)}, expected 1"
            )

        calc_mean, calc_sd = summary[method_key][q]

        ref_mean = float(row["vo_macro_pr_auc_mean"].iloc[0])

        ref_sd = float(row["vo_macro_pr_auc_sample_sd"].iloc[0])

        if not math.isclose(
            calc_mean,
            ref_mean,
            rel_tol=0.0,
            abs_tol=5e-12,
        ):
            abort(f"{display} q={q:.2f}: " "mean mismatch against " "audited summary")

        if not math.isclose(
            calc_sd,
            ref_sd,
            rel_tol=0.0,
            abs_tol=5e-12,
        ):
            abort(f"{display} q={q:.2f}: " "SD mismatch against " "audited summary")

audit.append("Diagnostic summary cross-check: PASS")

main_lines = [
    "% Auto-generated from audited seed-level results.",
    "% Do not edit numeric cells by hand.",
    "% Set the caption and label when including this table.",
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    (r"Method / diagnostic" r" & $q=.20$" r" & $q=.50$" r" & $q=.80$" r" & $q=.95$ \\"),
    r"\midrule",
]

for method_key, display, _ in ROWS:

    cells = [fmt(*summary[method_key][q]) for q in Q_ORDER]

    main_lines.append(display + " & " + " & ".join(cells) + r" \\")

main_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"% \caption{Collaboration diagnostics: AC Macro PR-AUC.}",
    r"% \label{tab:collaboration_diagnostics}",
    r"\end{table}",
    "",
]

MAIN_TEX.write_text("\n".join(main_lines))

# ================================================================
# 2. APPENDIX-D GCFL+ STRUCTURE TABLE
# ================================================================

required_gcfl = {
    "q",
    "seed",
    "final_round",
    "cluster_id",
    "cluster_size",
    "members",
}

missing = required_gcfl - set(gcfl.columns)

if missing:
    abort("GCFL+ cluster table missing columns: " f"{sorted(missing)}")

gcfl["q"] = pd.to_numeric(
    gcfl["q"],
    errors="raise",
).round(2)

gcfl["seed"] = pd.to_numeric(
    gcfl["seed"],
    errors="raise",
).astype(int)

gcfl["final_round"] = pd.to_numeric(
    gcfl["final_round"],
    errors="raise",
).astype(int)

gcfl["cluster_id"] = pd.to_numeric(
    gcfl["cluster_id"],
    errors="raise",
).astype(int)

gcfl["cluster_size"] = pd.to_numeric(
    gcfl["cluster_size"],
    errors="raise",
).astype(int)

appendix_rows = []

for q in Q_ORDER:

    for seed in SEEDS:

        group = gcfl[(gcfl["q"] == q) & (gcfl["seed"] == seed)].sort_values(
            "cluster_id"
        )

        if group.empty:
            abort(f"GCFL+ q={q:.2f} seed={seed}: " "no final cluster rows")

        if set(group["final_round"]) != {80}:
            abort(f"GCFL+ q={q:.2f} seed={seed}: " "final round is not exactly 80")

        if group["cluster_id"].duplicated().any():
            abort(f"GCFL+ q={q:.2f} seed={seed}: " "duplicate cluster IDs")

        all_members = []
        cluster_cells = []
        sizes = []

        for row in group.itertuples(index=False):

            members = parse_members(row.members)

            if len(members) != int(row.cluster_size):
                abort(
                    f"GCFL+ q={q:.2f} "
                    f"seed={seed} "
                    f"cluster={row.cluster_id}: "
                    "cluster size does not match "
                    "membership count"
                )

            all_members.extend(members)
            sizes.append(int(row.cluster_size))

            cluster_cells.append(
                f"C{int(row.cluster_id)}: " + r"\{" + ", ".join(members) + r"\}"
            )

        if len(all_members) != 20 or len(set(all_members)) != 20:
            abort(
                f"GCFL+ q={q:.2f} seed={seed}: "
                "final memberships are not a "
                "20-client disjoint partition"
            )

        appendix_rows.append(
            {
                "q": q,
                "seed": seed,
                "n_clusters": len(group),
                "sizes": ", ".join(str(x) for x in sizes),
                "memberships": "; ".join(cluster_cells),
            }
        )

        audit.append(
            f"GCFL+ q={q:.2f} seed={seed}: "
            f"clusters={len(group)}, "
            f"sizes={sizes}, "
            "20 unique clients, "
            "final round=80: PASS"
        )

if len(appendix_rows) != 12:
    abort("Expected 12 Appendix-D rows, " f"found {len(appendix_rows)}")

# Cross-check the already-audited exact-region
# recovery result if that column is available.

if "is_exact_physical_region" in gcfl.columns:

    flags = gcfl["is_exact_physical_region"].astype(str).str.strip().str.lower()

    n_exact = int(
        flags.isin(
            [
                "true",
                "1",
                "yes",
            ]
        ).sum()
    )

    if n_exact != 0:
        abort(
            "Expected zero exact "
            "physical-region cluster matches, "
            f"found {n_exact}"
        )

    audit.append(
        "GCFL+ exact physical-region " "matches across final partitions: " "0 PASS"
    )

appendix_lines = [
    "% Auto-generated from audited GCFL+ final-cluster results.",
    "% Requires booktabs and longtable.",
    "% Set the caption and label when including this table.",
    (r"\begin{longtable}" r"{ccclp{0.58\textwidth}}"),
    r"\toprule",
    (r"$q$ & Seed & Final clusters" r" & Sizes & Memberships \\"),
    r"\midrule",
    r"\endfirsthead",
    r"\toprule",
    (r"$q$ & Seed & Final clusters" r" & Sizes & Memberships \\"),
    r"\midrule",
    r"\endhead",
]

for row in appendix_rows:

    appendix_lines.append(
        f"{row['q']:.2f}"
        f" & {row['seed']}"
        f" & {row['n_clusters']}"
        f" & {row['sizes']}"
        f" & {row['memberships']} " + r"\\"
    )

appendix_lines += [
    r"\bottomrule",
    r"\end{longtable}",
    "",
]

GCFL_TEX.write_text("\n".join(appendix_lines))

# ================================================================
# 3. FINAL AUDIT REPORT
# ================================================================

audit += [
    "",
    "OUTPUTS",
    f"Main table: {MAIN_TEX}",
    f"Appendix table: {GCFL_TEX}",
    "No scientific source CSV was modified.",
    "FINAL STATUS: PASS",
]

AUDIT_TXT.write_text("\n".join(audit) + "\n")

print("=" * 88)
print("COLLABORATION LATEX TABLE BUILD + AUDIT")
print("=" * 88)

print(f"Main table     : {MAIN_TEX}")
print(f"Appendix table : {GCFL_TEX}")
print(f"Audit report   : {AUDIT_TXT}")

print()
print("MAIN-TABLE VALUES")

for method_key, display, _ in ROWS:

    values = " | ".join(
        fmt(*summary[method_key][q]).replace(
            " $\\pm$ ",
            " +/- ",
        )
        for q in Q_ORDER
    )

    print(f"  {display}: {values}")

print()
print(
    "Appendix-D GCFL+ rows:",
    len(appendix_rows),
    "/ 12",
)

print()
print(
    "PASS - tables generated directly "
    "from audited CSVs; "
    "no source artifact modified."
)
