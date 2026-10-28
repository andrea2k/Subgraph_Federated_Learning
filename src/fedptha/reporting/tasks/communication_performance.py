from fedptha.paths import generated_root

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(str(generated_root()))
PERF_PATH = ROOT / "tables/main_ac_summary.csv"
COMM_PATH = ROOT / "tables/communication_audit.csv"
OUT = ROOT / "tables"

POINTS_OUT = OUT / "communication_performance.csv"
DOM_OUT = OUT / "communication_frontier.csv"
SUMMARY_OUT = OUT / "communication_frontier.txt"

# ---------------------------------------------------------------------
# Main federated comparison only.
# Fully Local has zero communication by definition and would trivially
# distort a federated communication frontier.
# Global-Centralized has no comparable federated model-exchange cost.
# Collaboration-structure diagnostics are audited separately later.
# ---------------------------------------------------------------------

PERF_TO_COMM = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "gcflplus": "GCFL+",
    "fedala": "FedALA",
    "apple": "APPLE",
    "cwfedavg_task": "cwFedAvg-Task",
    "support_simplex": "Support-Simplex",
    "support_simplex_postala": "Support-Simplex + PostALA",
}

EXPECTED_SETUPS = {"five_client", "twenty_client"}
EXPECTED_Q = {0.20, 0.50, 0.80, 0.95}

perf = pd.read_csv(PERF_PATH)
comm = pd.read_csv(COMM_PATH)

# -----------------------------
# Validate frozen performance.
# -----------------------------
main = perf[perf["method"].isin(PERF_TO_COMM)].copy()

if len(main) != 64:
    raise RuntimeError(
        f"Expected 64 main-method VO rows "
        f"(2 setups x 4 q x 8 methods), got {len(main)}"
    )

if set(main["setup"]) != EXPECTED_SETUPS:
    raise RuntimeError(
        f"Unexpected performance setups: {sorted(main['setup'].unique())}"
    )

q_values = {round(float(x), 2) for x in main["q"].unique()}
if q_values != EXPECTED_Q:
    raise RuntimeError(f"Unexpected q values: {sorted(q_values)}")

if not (main["n_model_seeds"].astype(int) == 3).all():
    raise RuntimeError("Expected exactly 3 model seeds for every main point.")

main["communication_method"] = main["method"].map(PERF_TO_COMM)

# -----------------------------
# Validate communication table.
# -----------------------------
comm_main = comm[comm["method"].isin(set(PERF_TO_COMM.values()))].copy()

if len(comm_main) != 16:
    raise RuntimeError(
        f"Expected 16 main communication rows "
        f"(2 setups x 8 methods), got {len(comm_main)}"
    )

if comm_main["total_algorithmic_GiB"].isna().any():
    bad = comm_main.loc[
        comm_main["total_algorithmic_GiB"].isna(),
        ["setup", "method"],
    ]
    raise RuntimeError(
        "Missing algorithmic communication for:\n" + bad.to_string(index=False)
    )

# One communication value per setup/method.
dup = comm_main.duplicated(["setup", "method"], keep=False)
if dup.any():
    raise RuntimeError(
        "Duplicate communication rows:\n"
        + comm_main.loc[dup, ["setup", "method"]].to_string(index=False)
    )

# -----------------------------
# Merge frozen sources.
# -----------------------------
comm_cols = [
    "setup",
    "method",
    "total_algorithmic_bytes",
    "total_algorithmic_GiB",
    "training_80_bytes",
    "training_80_GiB",
    "setup_metadata_min_bytes",
    "algorithmic_one_shot_or_deploy_bytes",
    "total_algorithmic_definition",
]

points = main.merge(
    comm_main[comm_cols],
    left_on=["setup", "communication_method"],
    right_on=["setup", "method"],
    how="left",
    validate="many_to_one",
    suffixes=("_perf", "_comm"),
)

if points["total_algorithmic_GiB"].isna().any():
    bad = points.loc[
        points["total_algorithmic_GiB"].isna(),
        ["setup", "method_perf", "communication_method"],
    ]
    raise RuntimeError(
        "Performance/communication merge failed:\n" + bad.to_string(index=False)
    )

points = points.rename(
    columns={
        "method_perf": "method_id",
        "method_display": "method",
        "macro_pr_auc_mean": "vo_macro_pr_auc_mean",
        "macro_pr_auc_sample_sd": "vo_macro_pr_auc_sample_sd",
        "macro_pr_auc_mean_percent": "vo_macro_pr_auc_mean_percent",
        "macro_pr_auc_sample_sd_percent": "vo_macro_pr_auc_sample_sd_percent",
    }
)

keep_cols = [
    "setup",
    "q",
    "method_id",
    "method",
    "n_model_seeds",
    "vo_macro_pr_auc_mean",
    "vo_macro_pr_auc_sample_sd",
    "vo_macro_pr_auc_mean_percent",
    "vo_macro_pr_auc_sample_sd_percent",
    "macro_seed_values_json",
    "aggregation_rule",
    "total_algorithmic_bytes",
    "total_algorithmic_GiB",
    "training_80_bytes",
    "training_80_GiB",
    "setup_metadata_min_bytes",
    "algorithmic_one_shot_or_deploy_bytes",
    "total_algorithmic_definition",
]

points = points[keep_cols].copy()
points = points.sort_values(["setup", "q", "total_algorithmic_GiB", "method"])

# ---------------------------------------------------------------------
# Mean-based deterministic dominance.
#
# A dominates B iff:
#   communication_A <= communication_B
#   performance_A   >= performance_B
# with at least one strict inequality.
#
# SD is retained as uncertainty information but does not define dominance.
# ---------------------------------------------------------------------

TOL_C = 1e-12
TOL_P = 1e-12

audit_rows = []

for (setup, q), group in points.groupby(["setup", "q"], sort=True):
    records = group.to_dict("records")

    for target in records:
        dominators = []
        dominated_targets = []

        ct = float(target["total_algorithmic_GiB"])
        pt = float(target["vo_macro_pr_auc_mean_percent"])

        for other in records:
            if other["method_id"] == target["method_id"]:
                continue

            co = float(other["total_algorithmic_GiB"])
            po = float(other["vo_macro_pr_auc_mean_percent"])

            no_more_comm = co <= ct + TOL_C
            no_less_perf = po >= pt - TOL_P
            strictly_better = (co < ct - TOL_C) or (po > pt + TOL_P)

            if no_more_comm and no_less_perf and strictly_better:
                dominators.append(other["method"])

            # Does target dominate other?
            no_more_comm_t = ct <= co + TOL_C
            no_less_perf_t = pt >= po - TOL_P
            strictly_better_t = (ct < co - TOL_C) or (pt > po + TOL_P)

            if no_more_comm_t and no_less_perf_t and strictly_better_t:
                dominated_targets.append(other["method"])

        audit_rows.append(
            {
                **target,
                "n_dominators": len(dominators),
                "dominated": len(dominators) > 0,
                "pareto_nondominated": len(dominators) == 0,
                "dominated_by": " | ".join(sorted(dominators)),
                "n_methods_dominated": len(dominated_targets),
                "dominates": " | ".join(sorted(dominated_targets)),
                "dominance_definition": "mean-based: no more communication and no lower VO mean, "
                "with at least one strict inequality; SD not used",
            }
        )

audit = pd.DataFrame(audit_rows)
audit = audit.sort_values(
    ["setup", "q", "pareto_nondominated", "total_algorithmic_GiB", "method"],
    ascending=[True, True, False, True, True],
).reset_index(drop=True)

points.to_csv(POINTS_OUT, index=False)
audit.to_csv(DOM_OUT, index=False)

# -----------------------------
# Human-readable audit summary.
# -----------------------------
with SUMMARY_OUT.open("w") as f:
    f.write("COMMUNICATION-PERFORMANCE DOMINANCE AUDIT\n")
    f.write("=" * 88 + "\n\n")
    f.write(
        "Population: 8 main federated methods only. "
        "Fully Local and Global-Centralized excluded.\n"
    )
    f.write("Performance: frozen Figure-3 AC Macro PR-AUC mean over seeds 0,1,2.\n")
    f.write(
        "Communication: frozen total_algorithmic_GiB from final communication audit.\n"
    )
    f.write(
        "Dominance: A dominates B if A has no more communication and no lower "
        "VO mean, with at least one strict inequality.\n"
    )
    f.write(
        "Sample SD is retained for uncertainty visualization but does not alter "
        "mean-based dominance classification.\n\n"
    )

    for (setup, q), group in audit.groupby(["setup", "q"], sort=True):
        f.write("-" * 88 + "\n")
        f.write(f"{setup} | q={q:.2f}\n")
        f.write("-" * 88 + "\n")

        nd = group[group["pareto_nondominated"]]
        f.write("Nondominated methods:\n")
        for _, r in nd.sort_values(
            ["total_algorithmic_GiB", "vo_macro_pr_auc_mean_percent"]
        ).iterrows():
            f.write(
                f"  {r['method']}: "
                f"{r['total_algorithmic_GiB']:.6f} GiB, "
                f"VO={r['vo_macro_pr_auc_mean_percent']:.3f}"
                f" +/- {r['vo_macro_pr_auc_sample_sd_percent']:.3f}%\n"
            )

        f.write("\nDominated methods:\n")
        dom = group[group["dominated"]]
        if dom.empty:
            f.write("  none\n")
        else:
            for _, r in dom.sort_values(["total_algorithmic_GiB", "method"]).iterrows():
                f.write(
                    f"  {r['method']}: "
                    f"{r['total_algorithmic_GiB']:.6f} GiB, "
                    f"VO={r['vo_macro_pr_auc_mean_percent']:.3f}% "
                    f"| dominated by: {r['dominated_by']}\n"
                )
        f.write("\n")

print("=" * 88)
print("COMMUNICATION-PERFORMANCE AUDIT BUILT")
print("=" * 88)

for (setup, q), group in audit.groupby(["setup", "q"], sort=True):
    print()
    print(f"{setup} | q={q:.2f}")
    print(
        group[
            [
                "method",
                "total_algorithmic_GiB",
                "vo_macro_pr_auc_mean_percent",
                "vo_macro_pr_auc_sample_sd_percent",
                "pareto_nondominated",
                "dominated_by",
            ]
        ].to_string(index=False)
    )

print()
print("Wrote:")
print(" ", POINTS_OUT)
print(" ", DOM_OUT)
print(" ", SUMMARY_OUT)
