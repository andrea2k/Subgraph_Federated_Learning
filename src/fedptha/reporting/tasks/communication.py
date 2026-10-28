from fedptha.paths import generated_root

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(str(generated_root() / "communication_payload"))
OUT = Path(str(generated_root() / "tables"))
OUT.mkdir(parents=True, exist_ok=True)

Q = 80
T = 5
J = 5
FP32_BYTES = 4


def gib(x):
    if x is None:
        return None
    return float(x) / (1024**3)


def mib(x):
    if x is None:
        return None
    return float(x) / (1024**2)


def load_payload(setup):
    path = ROOT / setup / "payload_by_subset.csv"
    df = pd.read_csv(path)

    cols = [
        "state_full_bytes",
        "param_full_bytes",
        "state_nonhead_bytes",
        "param_nonhead_bytes",
        "state_head_bytes",
        "param_head_bytes",
    ]

    unique = df[cols].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError(
            f"{setup}: expected one invariant payload row, got {len(unique)}"
        )

    r = unique.iloc[0]
    return {c: int(r[c]) for c in cols}


rows = []


def add(
    *,
    setup,
    method,
    K,
    symbolic,
    per_round,
    setup_min=0,
    extra=0,
    eval_round=None,
    eval_80=None,
    status,
    manuscript_action,
    note="",
):
    training = None if per_round is None else int(Q * per_round)

    rows.append(
        {
            "setup": setup,
            "method": method,
            "K": K,
            "rounds": Q,
            "symbolic_expression": symbolic,
            "per_round_bytes": per_round,
            "per_round_MiB": None if per_round is None else mib(per_round),
            "training_80_bytes": training,
            "training_80_GiB": None if training is None else gib(training),
            "setup_metadata_min_bytes": setup_min,
            "algorithmic_one_shot_or_deploy_bytes": extra,
            "eval_only_per_round_bytes": eval_round,
            "eval_only_80_bytes": eval_80,
            "status": status,
            "manuscript_action": manuscript_action,
            "note": note,
        }
    )


for setup, K in [("five_client", 5), ("twenty_client", 20)]:
    p = load_payload(setup)

    P_state = p["state_full_bytes"]
    P_param = p["param_full_bytes"]
    B_state = p["state_nonhead_bytes"]
    H_state = p["state_head_bytes"]
    H_param = p["param_head_bytes"]

    support_upload = K * T * FP32_BYTES

    add(
        setup=setup,
        method="Fully Local",
        K=K,
        symbolic="0",
        per_round=0,
        status="PASS",
        manuscript_action="none",
    )

    for method in ["FedAvg", "FedProx", "FedALA", "GCFL+"]:
        add(
            setup=setup,
            method=method,
            K=K,
            symbolic="2 K P_state",
            per_round=2 * K * P_state,
            status="PASS",
            manuscript_action="keep asymptotic O(KP)",
            note=(
                "Complete state_dict download/upload. "
                "For GCFL+, clustering histories are derived coordinator-side "
                "from already available uploaded states."
                if method == "GCFL+"
                else "Complete state_dict download/upload."
            ),
        )

    add(
        setup=setup,
        method="cwFedAvg-Task",
        K=K,
        symbolic="2 K P_state",
        per_round=2 * K * P_state,
        setup_min=support_upload,
        status="PASS",
        manuscript_action=(
            "keep O(KP); report fixed support-count setup metadata separately"
        ),
        note=(
            f"Minimum one-time support collection: {K*T} FP32 scalars "
            f"= {support_upload} bytes."
        ),
    )

    add(
        setup=setup,
        method="APPLE",
        K=K,
        symbolic="K^2 P_param",
        per_round=K * K * P_param,
        status="PASS_WITH_NOTATION_CLARIFICATION",
        manuscript_action=(
            "keep asymptotic O(K^2 P), but distinguish P_param from full "
            "state_dict payload in exact-byte accounting"
        ),
        note="External donor mixing uses learnable parameters; receiver buffers remain local.",
    )

    for method in ["Support-Simplex", "Support-Simplex + PostALA"]:
        add(
            setup=setup,
            method=method,
            K=K,
            symbolic="2 K B_state + K^2 H_param",
            per_round=2 * K * B_state + K * K * H_param,
            setup_min=support_upload,
            status="PASS_WITH_NOTATION_CLARIFICATION",
            manuscript_action=(
                "keep asymptotic O(KB + K^2H); use B_state and H_param "
                "for exact bytes; report support metadata separately"
            ),
            note=(
                "PostALA is local and adds no repeated donor transfer."
                if method.endswith("PostALA")
                else "Backbone/non-head state uses FedAvg; donor routing uses task-head parameters."
            ),
        )

    add(
        setup=setup,
        method="Global-Centralized",
        K=K,
        symbolic="NOT_APPLICABLE",
        per_round=None,
        status="NOT_APPLICABLE",
        manuscript_action="exclude from federated communication comparison",
    )


# -------------------------------------------------------------------------
# Exact 20C regional diagnostics
# -------------------------------------------------------------------------

p20 = load_payload("twenty_client")
P20_state = p20["state_full_bytes"]
B20_state = p20["state_nonhead_bytes"]
H20_param = p20["param_head_bytes"]

regional_path = (
    ROOT / "twenty_client" / "oracle_regional" / "oracle_regional_communication.csv"
)
regional = pd.read_csv(regional_path)

if regional["physical_per_round_bytes"].nunique() != 1:
    raise RuntimeError("Physical regional per-round bytes vary across q.")
if regional["hierarchy_inter_bytes"].nunique() != 1:
    raise RuntimeError("Hierarchy one-shot bytes vary across q.")

physical_round = int(regional.iloc[0]["physical_per_round_bytes"])
hier_inter = int(regional.iloc[0]["hierarchy_inter_bytes"])

support_upload_20 = 20 * T * FP32_BYTES

add(
    setup="twenty_client",
    method="Oracle Physical-Community Isolation",
    K=20,
    symbolic="sum_r [2 n_r B_state,r + n_r^2 H_param,r]",
    per_round=physical_round,
    setup_min=support_upload_20,
    status="PASS_WITH_EXACT_BYTE_CORRECTION",
    manuscript_action=(
        "keep asymptotic O(KB + K^2H/R); use exact region-specific "
        "payload sum for concrete bytes"
    ),
    note="Region-specific port vocabularies cause very small B_state differences.",
)

add(
    setup="twenty_client",
    method="Oracle Hierarchical Collaboration + ALA",
    K=20,
    symbolic=(
        "sum_r [2 n_r B_state,r + n_r^2 H_param,r] " "+ one-shot sum_r n_r H_state,r"
    ),
    per_round=physical_round,
    setup_min=support_upload_20,
    extra=hier_inter,
    status="PASS_WITH_EXACT_BYTE_CORRECTION",
    manuscript_action=(
        "keep repeated regional asymptotic term plus one-shot O(KH); "
        "use exact regional bytes"
    ),
    note="Head-only ALA after cross-region delivery is local.",
)

# -------------------------------------------------------------------------
# Task-Specialist Representatives
# -------------------------------------------------------------------------

rep_round = 2 * J * B20_state + J * J * H20_param
rep_training = Q * rep_round

# Round-level validation/model-selection broadcast to the 20-client population.
rep_eval_round = 20 * P20_state
rep_eval_80 = Q * rep_eval_round

# Canonical deployment: five representative owners retain their own states,
# so only the 15 non-training recipients require an external transfer.
rep_deploy_external = (20 - J) * P20_state

add(
    setup="twenty_client",
    method="Task-Specialist Representatives",
    K=20,
    symbolic="2 J B_state + J^2 H_param, J=5",
    per_round=rep_round,
    setup_min=support_upload_20,
    extra=rep_deploy_external,
    eval_round=rep_eval_round,
    eval_80=rep_eval_80,
    status="PASS_WITH_SEPARATE_EXTRA_COSTS",
    manuscript_action=(
        "training formula PASS; keep round-level validation traffic excluded "
        "from optimization; report deployment separately; no 75% saving claim"
    ),
    note=(
        "Models are instantiated using the 20-client context, but only five "
        "selected representatives optimize. Deployment column counts transfers "
        "to the 15 non-training recipients. Literal server rebroadcast to all "
        "20 would instead cost 20*P_state."
    ),
)


df = pd.DataFrame(rows)

# Stable ordering.
order = [
    "Fully Local",
    "FedAvg",
    "FedProx",
    "FedALA",
    "GCFL+",
    "cwFedAvg-Task",
    "APPLE",
    "Support-Simplex",
    "Support-Simplex + PostALA",
    "Oracle Physical-Community Isolation",
    "Oracle Hierarchical Collaboration + ALA",
    "Task-Specialist Representatives",
    "Global-Centralized",
]

df["method_order"] = df["method"].map({m: i for i, m in enumerate(order)})
df["setup_order"] = df["setup"].map({"five_client": 0, "twenty_client": 1})
df = (
    df.sort_values(["setup_order", "method_order"])
    .drop(columns=["setup_order", "method_order"])
    .reset_index(drop=True)
)

# -------------------------------------------------------------------------
# Technical-review freeze fields.
#
# setup_metadata_min_bytes is intentionally NOT folded into the exact
# algorithmic total because it represents only the minimum client->coordinator
# support-count collection reconstructed from source. A complete metadata
# dissemination protocol is not explicitly materialized by the simulator.
#
# total_algorithmic_bytes therefore means:
#   repeated optimization communication
#   + required algorithmic one-shot/deployment communication
#
# Benchmark validation/test traffic is excluded.
# -------------------------------------------------------------------------

df["setup_metadata_label"] = df["setup_metadata_min_bytes"].apply(
    lambda x: ("minimum_one_time_support_count_collection" if int(x) > 0 else "none")
)

df["total_algorithmic_bytes"] = df.apply(
    lambda r: (
        None
        if pd.isna(r["training_80_bytes"])
        else int(r["training_80_bytes"])
        + int(r["algorithmic_one_shot_or_deploy_bytes"])
    ),
    axis=1,
)

df["total_algorithmic_GiB"] = df["total_algorithmic_bytes"].apply(
    lambda x: None if pd.isna(x) else gib(int(x))
)

df["total_algorithmic_definition"] = (
    "80-round optimization tensor payload + required algorithmic "
    "one-shot/deployment payload; validation/test traffic excluded; "
    "minimum support-count collection reported separately"
)

csv_path = OUT / "communication_audit.csv"
df.to_csv(csv_path, index=False)

payload_rows = []
for setup in ["five_client", "twenty_client"]:
    p = load_payload(setup)
    payload_rows.append(
        {
            "setup": setup,
            **p,
            "state_partition_ok": int(
                p["state_full_bytes"]
                == p["state_nonhead_bytes"] + p["state_head_bytes"]
            ),
            "head_state_equals_head_params": int(
                p["state_head_bytes"] == p["param_head_bytes"]
            ),
        }
    )

payload_df = pd.DataFrame(payload_rows)
payload_path = OUT / "payload_constants.csv"
payload_df.to_csv(payload_path, index=False)

summary_path = OUT / "communication_audit.txt"
with summary_path.open("w") as f:
    f.write("FINAL COMMUNICATION AUDIT\n")
    f.write("=" * 80 + "\n\n")

    for _, r in payload_df.iterrows():
        f.write(
            f"{r['setup']}: "
            f"P_state={r['state_full_bytes']} B, "
            f"P_param={r['param_full_bytes']} B, "
            f"B_state={r['state_nonhead_bytes']} B, "
            f"H_state={r['state_head_bytes']} B, "
            f"H_param={r['param_head_bytes']} B\n"
        )

    f.write("\n20C exact regional correction:\n")
    f.write(f"  Oracle Physical per round = {physical_round} B\n")
    f.write(f"  Oracle Physical 80 rounds = {Q * physical_round} B\n")
    f.write(f"  Hierarchy one-shot = {hier_inter} B\n")

    f.write("\nTask-Specialist Representatives:\n")
    f.write(f"  training per round = {rep_round} B\n")
    f.write(f"  training 80 rounds = {rep_training} B\n")
    f.write(f"  training 80 rounds = {gib(rep_training):.6f} GiB\n")
    f.write(f"  eval-only broadcast per round = {rep_eval_round} B\n")
    f.write(f"  eval-only broadcasts over 80 rounds = {gib(rep_eval_80):.6f} GiB\n")
    f.write(f"  deployment to 15 non-training recipients = {rep_deploy_external} B\n")
    f.write(f"  deployment = {mib(rep_deploy_external):.6f} MiB\n")

    f.write("\nTECHNICAL-REVIEW ACCOUNTING CONVENTION\n")
    f.write(
        "  setup_metadata_min_bytes = minimum one-time support-count "
        "collection only; complete metadata dissemination is not claimed.\n"
    )
    f.write(
        "  total_algorithmic_bytes = 80-round optimization communication "
        "+ required one-shot/deployment communication.\n"
    )
    f.write(
        "  Validation/test/model-selection traffic is excluded from "
        "total_algorithmic_bytes.\n"
    )

    f.write("\nFINAL ALGORITHMIC TOTALS\n")
    for _, row in df.iterrows():
        value = row["total_algorithmic_bytes"]
        if pd.isna(value):
            continue
        f.write(
            f"  {row['setup']} | {row['method']}: "
            f"{int(value)} B = {float(row['total_algorithmic_GiB']):.6f} GiB\n"
        )

    f.write("\nSTATUS\n")
    f.write("  Main asymptotic communication classes: PASS\n")
    f.write("  Main repeated exact-byte accounting: PASS\n")
    f.write("  APPLE exact-byte notation: clarification required in manuscript\n")
    f.write(
        "  Support-Simplex exact-byte notation: clarification required in manuscript\n"
    )
    f.write("  Oracle regional concrete bytes: corrected by exact regional audit\n")
    f.write("  Hierarchical one-shot accounting: PASS\n")
    f.write("  Specialist training accounting: PASS\n")
    f.write("  Support metadata: minimum collection reported separately\n")
    f.write("  Figure 6: NOT BUILT\n")
    f.write("  Pareto claim: NOT ASSESSED\n")

print("=" * 80)
print("FINAL COMMUNICATION AUDIT BUILT")
print("=" * 80)
print(payload_df.to_string(index=False))
print()
print(
    df[
        [
            "setup",
            "method",
            "per_round_bytes",
            "training_80_GiB",
            "setup_metadata_min_bytes",
            "algorithmic_one_shot_or_deploy_bytes",
            "status",
        ]
    ].to_string(index=False)
)
print()
print("Wrote:")
print(" ", csv_path)
print(" ", payload_path)
print(" ", summary_path)
