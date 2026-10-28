from fedptha.paths import generated_root

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

OUT = Path(str(generated_root() / "tables"))
COMM = OUT / "communication_audit.csv"

# Match the historical table's two-decimal display precision using regenerated scores.
summary = pd.read_csv(OUT / "collaboration_diagnostic_vo_summary.csv")
ablations = pd.read_csv(OUT / "ablation_ac_summary.csv")
labels = {
    "specialist_rep_transfer": "Task-Specialist Representatives",
    "oracle_physical_community": "Oracle Physical-Community Isolation",
    "oracle_hierarchical_taskfedavg_ala": "Oracle Hierarchical Collaboration + ALA",
}
records = []
for row in summary.itertuples():
    if row.diagnostic in labels:
        records.append(
            (
                labels[row.diagnostic],
                row.q,
                round(100 * row.vo_macro_pr_auc_mean, 2),
                round(100 * row.vo_macro_pr_auc_sample_sd, 2),
            )
        )
for row in ablations.itertuples():
    if row.setup == "twenty_client" and row.method == "support_simplex_postala":
        records.append(
            (
                "Support-Simplex + PostALA",
                row.q,
                round(100 * row.macro_pr_auc_mean, 2),
                round(100 * row.macro_pr_auc_sample_sd, 2),
            )
        )
perf = pd.DataFrame(
    records, columns=["method", "q", "vo_mean_percent", "vo_sd_percent"]
)
perf = perf.sort_values(["method", "q"]).reset_index(drop=True)

comm = pd.read_csv(COMM)
comm = comm[
    (comm["setup"] == "twenty_client") & comm["method"].isin(perf["method"].unique())
][["method", "total_algorithmic_bytes", "total_algorithmic_GiB"]].copy()

df = perf.merge(comm, on="method", how="left", validate="many_to_one")

if df["total_algorithmic_GiB"].isna().any():
    raise RuntimeError(
        "Missing communication rows:\n"
        + df[df["total_algorithmic_GiB"].isna()].to_string(index=False)
    )

rows = []

for q, group in df.groupby("q", sort=True):
    recs = group.to_dict("records")

    for target in recs:
        dominators = []

        for other in recs:
            if other["method"] == target["method"]:
                continue

            if (
                other["total_algorithmic_GiB"]
                <= target["total_algorithmic_GiB"] + 1e-12
                and other["vo_mean_percent"] >= target["vo_mean_percent"] - 1e-12
                and (
                    other["total_algorithmic_GiB"]
                    < target["total_algorithmic_GiB"] - 1e-12
                    or other["vo_mean_percent"] > target["vo_mean_percent"] + 1e-12
                )
            ):
                dominators.append(other["method"])

        rows.append(
            {
                **target,
                "pareto_nondominated_within_diagnostics": len(dominators) == 0,
                "dominated_by": " | ".join(sorted(dominators)),
            }
        )

audit = pd.DataFrame(rows).sort_values(
    ["q", "total_algorithmic_GiB", "vo_mean_percent"],
    ascending=[True, True, False],
)

csv_path = OUT / "communication_diagnostic_tradeoff_20c.csv"
txt_path = OUT / "communication_diagnostic_tradeoff_20c.txt"

audit.to_csv(csv_path, index=False)

with txt_path.open("w") as f:
    f.write("20C COMMUNICATION-PERFORMANCE DIAGNOSTIC CHECK\n")
    f.write("=" * 76 + "\n\n")
    f.write(
        "These are controlled/oracle collaboration diagnostics, not the main "
        "deployable-method frontier.\n\n"
    )

    for q, g in audit.groupby("q"):
        f.write(f"q={q:.2f}\n")
        for _, r in g.iterrows():
            flag = (
                "NONDOMINATED"
                if r["pareto_nondominated_within_diagnostics"]
                else "dominated"
            )
            f.write(
                f"  {r['method']}: "
                f"{r['total_algorithmic_GiB']:.6f} GiB, "
                f"{r['vo_mean_percent']:.2f} +/- {r['vo_sd_percent']:.2f}% "
                f"[{flag}]"
            )
            if r["dominated_by"]:
                f.write(f" by {r['dominated_by']}")
            f.write("\n")
        f.write("\n")

print(audit.to_string(index=False))
print()
print("Wrote:")
print(" ", csv_path)
print(" ", txt_path)
