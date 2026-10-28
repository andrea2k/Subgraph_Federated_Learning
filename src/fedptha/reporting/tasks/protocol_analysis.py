from __future__ import annotations

from fedptha.paths import generated_root

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(generated_root()))
CANON = ROOT / "tables" / "main_protocol_results.csv"
STEP3 = ROOT / "tables" / "protocol_summary.csv"
OUT = ROOT / "tables" / "protocol_comparison.csv"

SETUPS = ("five_client", "twenty_client")
Q_VALUES = (0.20, 0.50, 0.80, 0.95)
SEEDS = (0, 1, 2)
PROTOCOLS = ("VV", "VO", "OO")

METHODS = (
    "fully_local",
    "centralized",
    "fedavg",
    "fedprox",
    "gcflplus",
    "fedala",
    "apple",
    "cwfedavg_task",
    "support_simplex",
    "support_simplex_postala",
)

DISPLAY = {
    "fully_local": "Fully Local",
    "centralized": "Global-Centralized",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "gcflplus": "GCFL+",
    "fedala": "FedALA",
    "apple": "APPLE",
    "cwfedavg_task": "cwFedAvg-Task",
    "support_simplex": "Support-Simplex",
    "support_simplex_postala": "Support-Simplex + PostALA",
}

FEDERATED = (
    "fedavg",
    "fedprox",
    "gcflplus",
    "fedala",
    "apple",
    "cwfedavg_task",
    "support_simplex",
    "support_simplex_postala",
)

LITERATURE_FIVE = (
    "fedavg",
    "fedprox",
    "gcflplus",
    "fedala",
    "apple",
)

METRICS = (
    "macro_pr_auc",
    "cycle2_pr_auc",
    "cycle3_pr_auc",
    "cycle4_pr_auc",
    "cycle5_pr_auc",
    "cycle6_pr_auc",
)

ATOL = 5e-10


def abort(message: str) -> None:
    raise SystemExit(f"\nABORT: {message}\n")


def sample_sd(values) -> float:
    return float(pd.Series(values, dtype=float).std(ddof=1))


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def norm_text(value):
    return None if is_missing(value) else str(value).strip()


def norm_pipe(value):
    if is_missing(value):
        return None
    parts = [x.strip() for x in str(value).split("|") if x.strip()]
    return tuple(sorted(parts))


def canonicalize_json(obj):
    if isinstance(obj, dict):
        return {
            str(k): canonicalize_json(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, list):
        items = [canonicalize_json(v) for v in obj]
        return sorted(
            items,
            key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def norm_json(value):
    if is_missing(value):
        return None
    try:
        obj = json.loads(str(value))
    except Exception as exc:
        abort(f"invalid selection_provenance_json: {exc}")
    obj = canonicalize_json(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def state_signature(row) -> dict:
    return {
        "selection_scope": norm_text(row["selection_scope"]),
        "selected_round_values": norm_pipe(row["selected_round_values"]),
        "selected_epoch_values": norm_pipe(row["selected_epoch_values"]),
        "selection_provenance_json": norm_json(row["selection_provenance_json"]),
        "source_result_paths": norm_pipe(row["source_result_paths"]),
        "selector_metric": norm_text(row["selector_metric"]),
    }


def has_direct_state_evidence(signature: dict) -> bool:
    provenance = signature["selection_provenance_json"]
    return (
        signature["selected_round_values"] is not None
        or signature["selected_epoch_values"] is not None
        or provenance not in {None, "[]", "{}"}
    )


def pct(value: float) -> float:
    return 100.0 * float(value)


def fmt_mean_sd(mean: float, sd: float) -> str:
    return f"{pct(mean):6.2f} +/- {pct(sd):5.2f}"


def seed_json_from_group(group: pd.DataFrame, column: str) -> str:
    values = {
        str(int(row.seed)): float(getattr(row, column))
        for row in group.sort_values("seed").itertuples(index=False)
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def parse_seed_json(text: str) -> dict:
    try:
        obj = json.loads(text)
    except Exception as exc:
        abort(f"invalid seed JSON: {exc}")
    if set(map(str, obj.keys())) != {"0", "1", "2"}:
        abort(f"seed JSON keys are not exactly 0,1,2: {obj.keys()}")
    return {str(k): float(v) for k, v in obj.items()}


def check_registry(df: pd.DataFrame, name: str, require_seed: bool) -> None:
    if set(df["setup"]) != set(SETUPS):
        abort(f"{name}: setup registry mismatch")
    if set(df["method"]) != set(METHODS):
        abort(f"{name}: method registry mismatch")
    if set(df["q"]) != set(Q_VALUES):
        abort(f"{name}: q grid mismatch")
    if set(df["protocol"]) != set(PROTOCOLS):
        abort(f"{name}: protocol registry mismatch")
    if "task_weighted_fedavg" in set(df["method"]):
        abort(f"{name}: forbidden task_weighted_fedavg detected")
    if require_seed and set(df["seed"]) != set(SEEDS):
        abort(f"{name}: seed registry mismatch")


print("=" * 92)
print("FIGURE 4 - PROTOCOL AUDIT + 80-ROW ANALYSIS TABLE")
print("=" * 92)
print()
print("READ-ONLY INPUTS")
print(" ", CANON)
print(" ", STEP3)
print()
print("DERIVED OUTPUT")
print(" ", OUT)

for path in (CANON, STEP3):
    if not path.is_file():
        abort(f"missing input: {path}")

canonical = pd.read_csv(CANON)
summary = pd.read_csv(STEP3)

canonical["q"] = pd.to_numeric(canonical["q"], errors="raise").round(2)
summary["q"] = pd.to_numeric(summary["q"], errors="raise").round(2)
canonical["seed"] = pd.to_numeric(canonical["seed"], errors="raise").astype(int)

required_canonical = {
    "setup",
    "method",
    "q",
    "seed",
    "protocol",
    *METRICS,
    "selection_scope",
    "selected_round_values",
    "selected_epoch_values",
    "selection_provenance_json",
    "source_result_paths",
    "pr_auc_definition",
    "selector_metric",
    "paper_q_treatment",
    "centralized_q_pooling_applied",
}
required_summary = {
    "setup",
    "method",
    "method_display",
    "q",
    "protocol",
    "n_model_seeds",
    "macro_seed_values_json",
    "aggregation_rule",
    "centralized_q_pooled",
}
for metric in METRICS:
    required_summary.add(f"{metric}_mean")
    required_summary.add(f"{metric}_sample_sd")

missing = sorted(required_canonical - set(canonical.columns))
if missing:
    abort("canonical missing columns: " + ", ".join(missing))
missing = sorted(required_summary - set(summary.columns))
if missing:
    abort("Step-3 summary missing columns: " + ", ".join(missing))

print()
print("STEP A1 - GRID / REGISTRY / METRIC AUDIT")
print("-" * 92)

if len(canonical) != 720:
    abort(f"canonical rows = {len(canonical)}, expected 720")
if len(summary) != 240:
    abort(f"Step-3 summary rows = {len(summary)}, expected 240")

check_registry(canonical, "canonical", require_seed=True)
check_registry(summary, "Step-3 summary", require_seed=False)

seed_key = ["setup", "method", "q", "seed", "protocol"]
summary_key = ["setup", "method", "q", "protocol"]

if canonical.duplicated(seed_key).any():
    abort("duplicate canonical logical keys")
if summary.duplicated(summary_key).any():
    abort("duplicate Step-3 summary logical keys")

for key, group in canonical.groupby(summary_key, sort=False):
    seeds = tuple(sorted(group["seed"].tolist()))
    if seeds != SEEDS:
        abort(f"bad seed grid at {key}: {seeds}")

for metric in METRICS:
    values = pd.to_numeric(canonical[metric], errors="coerce")
    if not np.isfinite(values).all():
        abort(f"canonical {metric} contains non-finite values")
    if ((values < 0) | (values > 1)).any():
        abort(f"canonical {metric} outside [0,1]")

    means = pd.to_numeric(summary[f"{metric}_mean"], errors="coerce")
    sds = pd.to_numeric(summary[f"{metric}_sample_sd"], errors="coerce")
    if not np.isfinite(means).all() or not np.isfinite(sds).all():
        abort(f"Step-3 {metric} summary contains non-finite values")
    if ((means < 0) | (means > 1)).any() or (sds < 0).any():
        abort(f"Step-3 {metric} summary invalid")

if set(canonical["pr_auc_definition"].astype(str)) != {
    "trapezoidal_precision_recall_curve_auc"
}:
    abort("unexpected PR-AUC definition")

if set(canonical["selector_metric"].astype(str)) != {"macro_pr_auc"}:
    abort("unexpected selector metric")

if not (pd.to_numeric(summary["n_model_seeds"], errors="raise") == 3).all():
    abort("Step-3 summary contains n_model_seeds != 3")

print("Canonical rows          : 720 / 720")
print("Step-3 summary rows     : 240 / 240")
print("Methods                 : 10 / 10")
print("Setups                  : 2 / 2")
print("q values                : 0.20, 0.50, 0.80, 0.95")
print("Protocols               : VV, VO, OO")
print("Seeds                   : 0, 1, 2")
print("Duplicate logical keys  : NONE")
print("Task-Weighted FedAvg    : ABSENT")
print("True trapezoidal PR-AUC : CONFIRMED")

# Build a paper-facing seed-level numeric table.
# All ordinary methods retain their q-specific canonical values.
# Global-Centralized is different by frozen design: q does not define
# a different centralized learning problem, so its four q-tagged values
# are pooled within each setup/seed/protocol and then repeated across q.
paper_seed = canonical.copy()
central_mask = paper_seed["method"] == "centralized"

if set(canonical.loc[central_mask, "paper_q_treatment"].astype(str)) != {
    "pool_q_within_seed_then_repeat"
}:
    abort("unexpected Global-Centralized paper_q_treatment")

for metric in METRICS:
    pooled = (
        canonical.loc[central_mask]
        .groupby(["setup", "seed", "protocol"])[metric]
        .mean()
        .rename("_pooled_value")
        .reset_index()
    )

    idx = paper_seed.index[central_mask]
    central_rows = paper_seed.loc[idx, ["setup", "seed", "protocol"]].merge(
        pooled,
        on=["setup", "seed", "protocol"],
        how="left",
        validate="many_to_one",
    )

    if central_rows["_pooled_value"].isna().any():
        abort(f"failed to pool Global-Centralized {metric} within seed")

    paper_seed.loc[idx, metric] = central_rows["_pooled_value"].to_numpy()

print()
print("STEP A2 - REPRODUCE FROZEN STEP-3 SUMMARIES")
print("-" * 92)

reproduced_rows = []
for key, group in paper_seed.groupby(summary_key, sort=False):
    record = dict(zip(summary_key, key))
    for metric in METRICS:
        record[f"{metric}_re_mean"] = float(group[metric].mean())
        record[f"{metric}_re_sd"] = sample_sd(group[metric])
    reproduced_rows.append(record)

reproduced = pd.DataFrame(reproduced_rows)
check = summary.merge(
    reproduced,
    on=summary_key,
    how="outer",
    validate="one_to_one",
    indicator=True,
)
if not (check["_merge"] == "both").all():
    abort("canonical and Step-3 summary logical keys do not match")

max_mean_error = 0.0
max_sd_error = 0.0
for metric in METRICS:
    mean_error = (check[f"{metric}_mean"] - check[f"{metric}_re_mean"]).abs()
    sd_error = (check[f"{metric}_sample_sd"] - check[f"{metric}_re_sd"]).abs()

    max_mean_error = max(max_mean_error, float(mean_error.max()))
    max_sd_error = max(max_sd_error, float(sd_error.max()))

    if (mean_error > ATOL).any():
        abort(f"{metric} means do not reproduce frozen Step 3")
    if (sd_error > ATOL).any():
        abort(f"{metric} sample SDs do not reproduce frozen Step 3")

macro_lookup = paper_seed.set_index(seed_key)["macro_pr_auc"]
for row in summary.itertuples(index=False):
    stored = parse_seed_json(row.macro_seed_values_json)
    for seed_text, stored_value in stored.items():
        logical_key = (
            row.setup,
            row.method,
            round(float(row.q), 2),
            int(seed_text),
            row.protocol,
        )
        canonical_value = float(macro_lookup.loc[logical_key])
        if not np.isclose(stored_value, canonical_value, rtol=0.0, atol=ATOL):
            abort(f"stored macro seed value mismatch at {logical_key}")

print("Means reproduced         : YES")
print("Sample SDs reproduced    : YES")
print("Stored macro seed values : YES")
print(f"Maximum mean abs error   : {max_mean_error:.3e}")
print(f"Maximum SD abs error     : {max_sd_error:.3e}")

print()
print("STEP A3 - GLOBAL-CENTRALIZED q-POOLING AUDIT")
print("-" * 92)

raw_central = canonical[canonical["method"] == "centralized"].copy()
paper_central = paper_seed[paper_seed["method"] == "centralized"].copy()
summary_central = summary[summary["method"] == "centralized"].copy()

if not (
    pd.to_numeric(summary_central["centralized_q_pooled"], errors="raise") == 1
).all():
    abort("Step-3 Centralized rows are not marked centralized_q_pooled=1")

expected_rule = "pool_q_within_seed_then_mean_sample_sd_across_seeds"
if set(summary_central["aggregation_rule"].astype(str)) != {expected_rule}:
    abort("unexpected Step-3 Global-Centralized aggregation_rule")

# The raw canonical rows are q-tagged inputs. They may differ across q.
# The paper-facing numeric values must equal the within-seed q average
# repeated across all four q positions.
for key, raw_group in raw_central.groupby(["setup", "seed", "protocol"], sort=False):
    q_found = tuple(sorted(raw_group["q"].tolist()))
    if q_found != Q_VALUES:
        abort(f"Global-Centralized bad raw q grid at {key}: {q_found}")

    paper_group = paper_central[
        (paper_central["setup"] == key[0])
        & (paper_central["seed"] == key[1])
        & (paper_central["protocol"] == key[2])
    ].sort_values("q")

    if tuple(paper_group["q"].tolist()) != Q_VALUES:
        abort(f"Global-Centralized bad paper q grid at {key}")

    for metric in METRICS:
        expected = float(raw_group[metric].mean())
        repeated = paper_group[metric].to_numpy(dtype=float)
        if not np.allclose(repeated, expected, rtol=0.0, atol=ATOL):
            abort(f"Global-Centralized {metric} pooling/repeat mismatch at {key}")

# The frozen summary must repeat the same mean/SD at every q within setup/protocol.
for key, group in summary_central.groupby(["setup", "protocol"], sort=False):
    if tuple(sorted(group["q"].tolist())) != Q_VALUES:
        abort(f"Global-Centralized bad Step-3 q grid at {key}")
    for metric in METRICS:
        if group[f"{metric}_mean"].nunique(dropna=False) != 1:
            abort(f"Global-Centralized {metric} Step-3 mean changes across q at {key}")
        if group[f"{metric}_sample_sd"].nunique(dropna=False) != 1:
            abort(f"Global-Centralized {metric} Step-3 SD changes across q at {key}")

print("Raw q-tagged Centralized values : allowed to differ across q")
print("Within-seed q pooling           : CONFIRMED")
print("Repeated paper-facing q values  : CONFIRMED")
print("Frozen Step-3 pooling rule      : CONFIRMED")
print("Step-3 Centralized q repeat     : CONFIRMED")
print()
print("STEP A4 - VV / VO SAME-SELECTED-STATE AUDIT")
print("-" * 92)

mismatches = []
insufficient = []
pair_count = 0
scope_counts = {}

for key, group in canonical.groupby(["setup", "method", "q", "seed"], sort=False):
    if set(group["protocol"]) != set(PROTOCOLS):
        abort(f"protocol triplet missing at {key}")

    vv = group[group["protocol"] == "VV"].iloc[0]
    vo = group[group["protocol"] == "VO"].iloc[0]

    sig_vv = state_signature(vv)
    sig_vo = state_signature(vo)

    scope = sig_vv["selection_scope"]
    scope_counts[scope] = scope_counts.get(scope, 0) + 1

    if not has_direct_state_evidence(sig_vv) or not has_direct_state_evidence(sig_vo):
        insufficient.append(key)

    differing = [field for field in sig_vv if sig_vv[field] != sig_vo[field]]
    if differing:
        mismatches.append((key, differing, sig_vv, sig_vo))

    pair_count += 1

if mismatches:
    key, fields, sig_vv, sig_vo = mismatches[0]
    print("First mismatch:")
    print("  key:", key)
    for field in fields:
        print(" ", field)
        print("    VV:", sig_vv[field])
        print("    VO:", sig_vo[field])
    abort(f"{len(mismatches)} VV/VO pairs have different selection metadata")

if insufficient:
    print("First keys lacking direct selection evidence:")
    for key in insufficient[:20]:
        print(" ", key)
    abort(
        f"{len(insufficient)} VV/VO pairs lack selected-round/epoch/provenance evidence"
    )

if pair_count != 240:
    abort(f"VV/VO pair count = {pair_count}, expected 240")

print("VV/VO pairs checked      : 240 / 240")
print("Selection metadata       : IDENTICAL")
print("Direct state evidence    : PRESENT")
print("Selection-scope counts   :", scope_counts)
print(
    "Meaning                  : VV and VO use the same visible-selected state/state collection"
)
print(
    "OO note                  : OO may select a different state because full validation is allowed"
)

print()
print("STEP B - BUILD 80-ROW FIGURE-4 ANALYSIS TABLE")
print("-" * 92)

wide = paper_seed.pivot(
    index=["setup", "method", "q", "seed"],
    columns="protocol",
    values="macro_pr_auc",
).reset_index()

if len(wide) != 240:
    abort(f"seed-level protocol pivot rows = {len(wide)}, expected 240")
if wide[list(PROTOCOLS)].isna().any().any():
    abort("missing VV/VO/OO value in seed-level protocol pivot")

wide["VO_minus_VV"] = wide["VO"] - wide["VV"]
wide["OO_minus_VO"] = wide["OO"] - wide["VO"]

analysis_rows = []
for (setup, method, q), group in wide.groupby(["setup", "method", "q"], sort=False):
    group = group.sort_values("seed").copy()
    if tuple(group["seed"].tolist()) != SEEDS:
        abort(f"bad seed rows for {setup}/{method}/q={q}")

    analysis_rows.append(
        {
            "setup": setup,
            "method": method,
            "method_display": DISPLAY[method],
            "q": float(q),
            "VV_mean": float(group["VV"].mean()),
            "VV_sample_sd": sample_sd(group["VV"]),
            "VO_mean": float(group["VO"].mean()),
            "VO_sample_sd": sample_sd(group["VO"]),
            "OO_mean": float(group["OO"].mean()),
            "OO_sample_sd": sample_sd(group["OO"]),
            "VO_minus_VV_mean": float(group["VO_minus_VV"].mean()),
            "VO_minus_VV_sample_sd": sample_sd(group["VO_minus_VV"]),
            "OO_minus_VO_mean": float(group["OO_minus_VO"].mean()),
            "OO_minus_VO_sample_sd": sample_sd(group["OO_minus_VO"]),
            "VO_minus_VV_seed_values_json": seed_json_from_group(group, "VO_minus_VV"),
            "OO_minus_VO_seed_values_json": seed_json_from_group(group, "OO_minus_VO"),
        }
    )

analysis = pd.DataFrame(analysis_rows)
if len(analysis) != 80:
    abort(f"analysis rows = {len(analysis)}, expected 80")

for protocol in PROTOCOLS:
    analysis[f"{protocol}_rank_all"] = analysis.groupby(["setup", "q"])[
        f"{protocol}_mean"
    ].rank(method="average", ascending=False)

    analysis[f"{protocol}_rank_federated"] = np.nan
    mask = analysis["method"].isin(FEDERATED)
    analysis.loc[mask, f"{protocol}_rank_federated"] = (
        analysis.loc[mask]
        .groupby(["setup", "q"])[f"{protocol}_mean"]
        .rank(method="average", ascending=False)
    )

analysis["VV_to_VO_rank_change_federated"] = (
    analysis["VV_rank_federated"] - analysis["VO_rank_federated"]
)
analysis["VO_to_OO_rank_change_federated"] = (
    analysis["VO_rank_federated"] - analysis["OO_rank_federated"]
)

reference_mask = analysis["method"].isin(("fully_local", "centralized"))
fed_rank_cols = ["VV_rank_federated", "VO_rank_federated", "OO_rank_federated"]
if analysis.loc[reference_mask, fed_rank_cols].notna().any().any():
    abort("non-federated reference received a federated rank")

for protocol in PROTOCOLS:
    frozen = summary[summary["protocol"] == protocol][
        ["setup", "method", "q", "macro_pr_auc_mean", "macro_pr_auc_sample_sd"]
    ]
    merged = analysis.merge(
        frozen,
        on=["setup", "method", "q"],
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        merged[f"{protocol}_mean"],
        merged["macro_pr_auc_mean"],
        rtol=0.0,
        atol=ATOL,
    ):
        abort(f"derived {protocol} means disagree with Step 3")
    if not np.allclose(
        merged[f"{protocol}_sample_sd"],
        merged["macro_pr_auc_sample_sd"],
        rtol=0.0,
        atol=ATOL,
    ):
        abort(f"derived {protocol} sample SDs disagree with Step 3")

method_order = {method: i for i, method in enumerate(METHODS)}
analysis["_method_order"] = analysis["method"].map(method_order)
analysis = (
    analysis.sort_values(["setup", "q", "_method_order"])
    .drop(columns="_method_order")
    .reset_index(drop=True)
)

OUT.parent.mkdir(parents=True, exist_ok=True)
analysis.to_csv(OUT, index=False, na_rep="")

print("Seed-level protocol rows  : 240")
print("Derived analysis rows     : 80 / 80")
print("rank_all                  : all 10 entries")
print("rank_federated            : 8 federated methods")
print("Fully Local fed rank      : NaN")
print("Global-Centralized fed rank: NaN")
print("Rank 1                    : highest mean PR-AUC")
print("Positive rank change      : movement toward rank 1 under later protocol")
print("Created:")
print(" ", OUT)

print()
print("STEP C - HUMAN-READABLE DIAGNOSTICS")
print("=" * 92)
print("Scores are Macro true PR-AUC (%).")
print("Ranks are descriptive mean-based orderings, not significance tests.")

for setup in SETUPS:
    for q in Q_VALUES:
        subset = analysis[(analysis["setup"] == setup) & (analysis["q"] == q)].copy()

        print()
        print("=" * 92)
        print(f"{setup} | q={q:.2f}")
        print("=" * 92)

        for protocol in PROTOCOLS:
            fed_subset = subset[subset["method"].isin(FEDERATED)].sort_values(
                f"{protocol}_mean", ascending=False
            )

            print(f"{protocol} top 3 federated methods:")
            for row in fed_subset.head(3).itertuples(index=False):
                mean = getattr(row, f"{protocol}_mean")
                sd = getattr(row, f"{protocol}_sample_sd")
                rank = getattr(row, f"{protocol}_rank_federated")
                print(
                    f"  #{rank:>3.0f} {row.method_display:<30} "
                    f"{fmt_mean_sd(mean, sd)}"
                )

            literature = (
                subset[subset["method"].isin(LITERATURE_FIVE)]
                .sort_values(f"{protocol}_mean", ascending=False)
                .iloc[0]
            )
            literature_text = fmt_mean_sd(
                literature[f"{protocol}_mean"],
                literature[f"{protocol}_sample_sd"],
            )
            print(
                "  best of FedAvg/FedProx/GCFL+/FedALA/APPLE: "
                f"{literature['method_display']} {literature_text}"
            )

        print()
        print("Key methods:")
        for method in (
            "fedavg",
            "cwfedavg_task",
            "support_simplex",
            "support_simplex_postala",
        ):
            row = subset[subset["method"] == method].iloc[0]
            print(
                f"  {row['method_display']:<30} "
                f"VV {fmt_mean_sd(row['VV_mean'], row['VV_sample_sd'])} | "
                f"VO {fmt_mean_sd(row['VO_mean'], row['VO_sample_sd'])} | "
                f"OO {fmt_mean_sd(row['OO_mean'], row['OO_sample_sd'])}"
            )
            print(
                f"    VO-VV {pct(row['VO_minus_VV_mean']):+7.2f} "
                f"+/- {pct(row['VO_minus_VV_sample_sd']):5.2f} pp | "
                f"OO-VO {pct(row['OO_minus_VO_mean']):+7.2f} "
                f"+/- {pct(row['OO_minus_VO_sample_sd']):5.2f} pp"
            )

        print()
        print("Largest absolute VO-VV changes among federated methods:")
        largest_delta = subset[subset["method"].isin(FEDERATED)].copy()
        largest_delta["_abs_delta"] = largest_delta["VO_minus_VV_mean"].abs()
        largest_delta = largest_delta.sort_values("_abs_delta", ascending=False)
        for row in largest_delta.head(3).itertuples(index=False):
            print(
                f"  {row.method_display:<30} "
                f"{pct(row.VO_minus_VV_mean):+7.2f} "
                f"+/- {pct(row.VO_minus_VV_sample_sd):5.2f} pp"
            )

        print()
        print("Largest VV->VO federated rank changes:")
        rank_changes = subset[subset["method"].isin(FEDERATED)].copy()
        rank_changes["_abs_rank_change"] = rank_changes[
            "VV_to_VO_rank_change_federated"
        ].abs()
        rank_changes = rank_changes.sort_values(
            ["_abs_rank_change", "VO_minus_VV_mean"],
            ascending=[False, False],
        )

        for row in rank_changes.head(3).itertuples(index=False):
            seed_deltas = json.loads(row.VO_minus_VV_seed_values_json)
            seed_text = ", ".join(
                f"s{seed}={pct(value):+.2f}"
                for seed, value in sorted(seed_deltas.items())
            )
            print(
                f"  {row.method_display:<30} "
                f"rank {row.VV_rank_federated:.0f} -> "
                f"{row.VO_rank_federated:.0f} "
                f"(change {row.VV_to_VO_rank_change_federated:+.0f})"
            )
            print(
                f"    VV {fmt_mean_sd(row.VV_mean, row.VV_sample_sd)} | "
                f"VO {fmt_mean_sd(row.VO_mean, row.VO_sample_sd)}"
            )
            print(f"    paired VO-VV seeds: {seed_text}")
            print(
                "    paired mean +/- SD: "
                f"{pct(row.VO_minus_VV_mean):+.2f} "
                f"+/- {pct(row.VO_minus_VV_sample_sd):.2f} pp"
            )

print()
print("=" * 92)
print("PASS - FIGURE 4 PROTOCOL AUDIT + ANALYSIS TABLE CREATED")
print("=" * 92)
print()
print("No source result file was modified.")
print("No Figure 4 plot was created.")
print("Next: inspect all q diagnostics, then choose q=.95-only versus q=.80+.95.")
