from fedptha.paths import generated_root

#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

INPUT = Path(str(generated_root() / "tables" / "main_protocol_results.csv"))

OUT_DIR = Path(str(generated_root() / "tables"))

FULL_OUT = OUT_DIR / "protocol_summary.csv"
VO_MACRO_OUT = OUT_DIR / "main_ac_summary.csv"
VO_TASK_OUT = OUT_DIR / "task_ac_summary.csv"
AUDIT_OUT = OUT_DIR / "summary_audit.txt"
NOTES_OUT = OUT_DIR / "summary_notes.txt"

SETUPS = (
    "five_client",
    "twenty_client",
)

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

Q_VALUES = (
    "0.20",
    "0.50",
    "0.80",
    "0.95",
)

SEEDS = (
    0,
    1,
    2,
)

PROTOCOLS = (
    "VV",
    "VO",
    "OO",
)

METRICS = (
    "macro_pr_auc",
    "cycle2_pr_auc",
    "cycle3_pr_auc",
    "cycle4_pr_auc",
    "cycle5_pr_auc",
    "cycle6_pr_auc",
)

TASKS = (
    ("cycle2", "cycle2_pr_auc"),
    ("cycle3", "cycle3_pr_auc"),
    ("cycle4", "cycle4_pr_auc"),
    ("cycle5", "cycle5_pr_auc"),
    ("cycle6", "cycle6_pr_auc"),
)

EXPECTED_DEFINITION = "trapezoidal_precision_recall_curve_auc"

errors = []


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    s = str(value).strip()

    if not s:
        return None

    if s.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }:
        return None

    try:
        x = float(s)
    except Exception:
        return None

    if not math.isfinite(x):
        return None

    return x


def mean_sample_sd(values):
    vals = list(values)

    if len(vals) != 3:
        raise ValueError(f"expected 3 seed values, found {len(vals)}")

    if any(v is None for v in vals):
        raise ValueError("undefined value in seed aggregation")

    return (
        statistics.mean(vals),
        statistics.stdev(vals),
    )


def f12(value):
    return f"{value:.12f}"


def pct(value):
    return f"{100.0 * value:.3f}"


print("=" * 76)
print("Summarize experiment metrics")
print("=" * 76)

if not INPUT.is_file():
    raise SystemExit(f"ABORT: missing extracted result table: {INPUT}")


rows = read_csv(INPUT)

if len(rows) != 720:
    raise SystemExit(f"ABORT: input rows={len(rows)}, expected 720.")

print("Canonical input rows : 720 / 720")


# ============================================================
# 1. Validate authoritative canonical input
# ============================================================

data = {}

for line_no, row in enumerate(rows, start=2):
    try:
        setup = str(row["setup"]).strip()
        method = str(row["method"]).strip()
        q = f"{float(row['q']):.2f}"
        seed = int(row["seed"])
        protocol = str(row["protocol"]).strip()
    except Exception as exc:
        errors.append(f"row {line_no}: malformed logical key: {exc}")
        continue

    key = (
        setup,
        method,
        q,
        seed,
        protocol,
    )

    if key in data:
        errors.append(f"row {line_no}: duplicate logical key {key}")
        continue

    if setup not in SETUPS:
        errors.append(f"row {line_no}: unexpected setup {setup}")

    if method not in METHODS:
        errors.append(f"row {line_no}: unexpected method {method}")

    if method == "task_weighted_fedavg":
        errors.append(
            f"row {line_no}: Task-Weighted FedAvg "
            "must not appear in final canonical data"
        )

    if q not in Q_VALUES:
        errors.append(f"row {line_no}: unexpected q={q}")

    if seed not in SEEDS:
        errors.append(f"row {line_no}: unexpected seed={seed}")

    if protocol not in PROTOCOLS:
        errors.append(f"row {line_no}: unexpected protocol={protocol}")

    definition = str(row.get("pr_auc_definition", "")).strip()

    if definition != EXPECTED_DEFINITION:
        errors.append(f"row {line_no}: invalid PR-AUC definition " f"{definition!r}")

    selector = str(row.get("selector_metric", "")).strip()

    if selector != "macro_pr_auc":
        errors.append(f"row {line_no}: invalid selector {selector!r}")

    metric_values = {}

    for metric in METRICS:
        value = finite_float(row.get(metric, ""))

        if value is None:
            errors.append(f"row {line_no}: undefined {metric} for {key}")

        metric_values[metric] = value

    data[key] = metric_values


expected_keys = {
    (
        setup,
        method,
        q,
        seed,
        protocol,
    )
    for setup in SETUPS
    for method in METHODS
    for q in Q_VALUES
    for seed in SEEDS
    for protocol in PROTOCOLS
}

actual_keys = set(data)

if len(actual_keys) != 720:
    errors.append(f"unique canonical keys={len(actual_keys)}, expected 720")

if actual_keys != expected_keys:
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    if missing:
        errors.append(f"missing canonical keys: {sorted(missing)[:20]}")

    if extra:
        errors.append(f"extra canonical keys: {sorted(extra)[:20]}")


# ============================================================
# 2. Verify every raw summary group has exactly seeds 0,1,2
# ============================================================

seed_groups = defaultdict(set)

for (
    setup,
    method,
    q,
    seed,
    protocol,
) in data:
    seed_groups[
        (
            setup,
            method,
            q,
            protocol,
        )
    ].add(seed)

if len(seed_groups) != 240:
    errors.append(f"raw seed groups={len(seed_groups)}, expected 240")

bad_seed_groups = []

for key, seeds in seed_groups.items():
    if seeds != set(SEEDS):
        bad_seed_groups.append(
            (
                key,
                sorted(seeds),
            )
        )

if bad_seed_groups:
    errors.append(
        f"bad seed groups={len(bad_seed_groups)}; " f"examples={bad_seed_groups[:10]}"
    )


# ============================================================
# 3. Aggregate
#
# Non-Centralized:
#   each q independently, sample SD across 3 model seeds.
#
# Centralized:
#   average q within each seed;
#   then sample SD across the 3 pooled seed values;
#   repeat same result at all q.
# ============================================================

summary_rows = []

for setup in SETUPS:
    for method in METHODS:
        for protocol in PROTOCOLS:

            central_pooled = {}

            if method == "centralized":
                for metric in METRICS:
                    central_pooled[metric] = {}

                    for seed in SEEDS:
                        q_vals = [
                            data[
                                (
                                    setup,
                                    method,
                                    q,
                                    seed,
                                    protocol,
                                )
                            ][metric]
                            for q in Q_VALUES
                        ]

                        if any(value is None for value in q_vals):
                            errors.append(
                                f"{setup}/centralized/"
                                f"{protocol}/{metric}/"
                                f"seed={seed}: undefined q value"
                            )
                            continue

                        central_pooled[metric][seed] = statistics.mean(q_vals)

            for q in Q_VALUES:
                out = {
                    "setup": setup,
                    "method": method,
                    "method_display": DISPLAY[method],
                    "q": q,
                    "protocol": protocol,
                    "n_model_seeds": 3,
                    "centralized_q_pooled": 1 if method == "centralized" else 0,
                    "aggregation_rule": (
                        "pool_q_within_seed_then_" "mean_sample_sd_across_seeds"
                        if method == "centralized"
                        else "mean_sample_sd_across_seeds"
                    ),
                }

                for metric in METRICS:
                    if method == "centralized":
                        seed_values = [central_pooled[metric][seed] for seed in SEEDS]
                    else:
                        seed_values = [
                            data[
                                (
                                    setup,
                                    method,
                                    q,
                                    seed,
                                    protocol,
                                )
                            ][metric]
                            for seed in SEEDS
                        ]

                    try:
                        mean, sd = mean_sample_sd(seed_values)
                    except Exception as exc:
                        errors.append(
                            f"{setup}/{method}/q={q}/" f"{protocol}/{metric}: {exc}"
                        )
                        continue

                    out[f"{metric}_mean"] = f12(mean)

                    out[f"{metric}_sample_sd"] = f12(sd)

                    if metric == "macro_pr_auc":
                        out["macro_seed_values_json"] = json.dumps(
                            {
                                str(seed): seed_values[idx]
                                for idx, seed in enumerate(SEEDS)
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )

                summary_rows.append(out)


# ============================================================
# 4. Final numerical audit before writing
# ============================================================

if len(summary_rows) != 240:
    errors.append(f"summary rows={len(summary_rows)}, expected 240")

summary_keys = {
    (
        row["setup"],
        row["method"],
        row["q"],
        row["protocol"],
    )
    for row in summary_rows
}

expected_summary_keys = {
    (
        setup,
        method,
        q,
        protocol,
    )
    for setup in SETUPS
    for method in METHODS
    for q in Q_VALUES
    for protocol in PROTOCOLS
}

if len(summary_keys) != 240:
    errors.append(f"unique summary keys={len(summary_keys)}, expected 240")

if summary_keys != expected_summary_keys:
    errors.append(
        "summary logical grid does not match expected "
        "10-method x setup x q x protocol grid"
    )


# Centralized must be identical across displayed q after pooling.
for setup in SETUPS:
    for protocol in PROTOCOLS:
        central_rows = [
            row
            for row in summary_rows
            if (
                row["setup"] == setup
                and row["method"] == "centralized"
                and row["protocol"] == protocol
            )
        ]

        for metric in METRICS:
            pairs = {
                (
                    row[f"{metric}_mean"],
                    row[f"{metric}_sample_sd"],
                )
                for row in central_rows
            }

            if len(pairs) != 1:
                errors.append(
                    f"{setup}/centralized/{protocol}/"
                    f"{metric}: pooled result varies across q"
                )


if errors:
    print()
    print("=" * 76)
    print(f"Result summary failed " f"({len(errors)} issue(s))")
    print("=" * 76)

    for error in errors:
        print(" -", error)

    print()
    print("No summary outputs were written.")
    raise SystemExit(1)


# ============================================================
# 5. Sort and write full 240-row summary
# ============================================================

setup_rank = {setup: i for i, setup in enumerate(SETUPS)}

method_rank = {method: i for i, method in enumerate(METHODS)}

protocol_rank = {protocol: i for i, protocol in enumerate(PROTOCOLS)}

summary_rows.sort(
    key=lambda row: (
        setup_rank[row["setup"]],
        method_rank[row["method"]],
        float(row["q"]),
        protocol_rank[row["protocol"]],
    )
)

full_fields = [
    "setup",
    "method",
    "method_display",
    "q",
    "protocol",
    "n_model_seeds",
    "macro_pr_auc_mean",
    "macro_pr_auc_sample_sd",
    "cycle2_pr_auc_mean",
    "cycle2_pr_auc_sample_sd",
    "cycle3_pr_auc_mean",
    "cycle3_pr_auc_sample_sd",
    "cycle4_pr_auc_mean",
    "cycle4_pr_auc_sample_sd",
    "cycle5_pr_auc_mean",
    "cycle5_pr_auc_sample_sd",
    "cycle6_pr_auc_mean",
    "cycle6_pr_auc_sample_sd",
    "macro_seed_values_json",
    "aggregation_rule",
    "centralized_q_pooled",
]

with FULL_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=full_fields,
    )
    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# 6. VO Macro table: 80 rows
# ============================================================

vo_rows = [row for row in summary_rows if row["protocol"] == "VO"]

if len(vo_rows) != 80:
    raise SystemExit(f"INTERNAL ERROR: VO rows={len(vo_rows)}, expected 80")

vo_macro_rows = []

for row in vo_rows:
    mean = float(row["macro_pr_auc_mean"])

    sd = float(row["macro_pr_auc_sample_sd"])

    vo_macro_rows.append(
        {
            "setup": row["setup"],
            "method": row["method"],
            "method_display": row["method_display"],
            "q": row["q"],
            "n_model_seeds": row["n_model_seeds"],
            "macro_pr_auc_mean": row["macro_pr_auc_mean"],
            "macro_pr_auc_sample_sd": row["macro_pr_auc_sample_sd"],
            "macro_pr_auc_mean_percent": f"{100 * mean:.6f}",
            "macro_pr_auc_sample_sd_percent": f"{100 * sd:.6f}",
            "formatted_percent": f"{pct(mean)} +/- {pct(sd)}",
            "macro_seed_values_json": row["macro_seed_values_json"],
            "aggregation_rule": row["aggregation_rule"],
        }
    )

vo_macro_fields = [
    "setup",
    "method",
    "method_display",
    "q",
    "n_model_seeds",
    "macro_pr_auc_mean",
    "macro_pr_auc_sample_sd",
    "macro_pr_auc_mean_percent",
    "macro_pr_auc_sample_sd_percent",
    "formatted_percent",
    "macro_seed_values_json",
    "aggregation_rule",
]

with VO_MACRO_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=vo_macro_fields,
    )
    writer.writeheader()
    writer.writerows(vo_macro_rows)


# ============================================================
# 7. VO task summary: 80 x 5 = 400 rows
# ============================================================

vo_task_rows = []

for row in vo_rows:
    for task_name, metric in TASKS:
        mean = float(row[f"{metric}_mean"])

        sd = float(row[f"{metric}_sample_sd"])

        vo_task_rows.append(
            {
                "setup": row["setup"],
                "method": row["method"],
                "method_display": row["method_display"],
                "q": row["q"],
                "task": task_name,
                "n_model_seeds": row["n_model_seeds"],
                "pr_auc_mean": f12(mean),
                "pr_auc_sample_sd": f12(sd),
                "pr_auc_mean_percent": f"{100 * mean:.6f}",
                "pr_auc_sample_sd_percent": f"{100 * sd:.6f}",
                "formatted_percent": f"{pct(mean)} +/- {pct(sd)}",
                "aggregation_rule": row["aggregation_rule"],
            }
        )

if len(vo_task_rows) != 400:
    raise SystemExit(
        f"INTERNAL ERROR: VO task rows=" f"{len(vo_task_rows)}, expected 400"
    )

vo_task_fields = [
    "setup",
    "method",
    "method_display",
    "q",
    "task",
    "n_model_seeds",
    "pr_auc_mean",
    "pr_auc_sample_sd",
    "pr_auc_mean_percent",
    "pr_auc_sample_sd_percent",
    "formatted_percent",
    "aggregation_rule",
]

with VO_TASK_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=vo_task_fields,
    )
    writer.writeheader()
    writer.writerows(vo_task_rows)


# ============================================================
# 8. Create copy-paste-friendly Markdown VO tables
# ============================================================


def markdown_table(setup):
    title = "FIVE-CLIENT" if setup == "five_client" else "TWENTY-CLIENT"

    lines = []

    lines.append(f"### {title}")
    lines.append("")
    lines.append("| Method | q=0.20 | q=0.50 | q=0.80 | q=0.95 |")
    lines.append("|---|---:|---:|---:|---:|")

    for method in METHODS:
        values = {}

        for row in vo_macro_rows:
            if row["setup"] == setup and row["method"] == method:
                values[row["q"]] = row["formatted_percent"]

        cells = [values[q] for q in Q_VALUES]

        lines.append("| " + DISPLAY[method] + " | " + " | ".join(cells) + " |")

    return "\n".join(lines)


five_table = markdown_table("five_client")

twenty_table = markdown_table("twenty_client")


input_sha = sha256(INPUT)
full_sha = sha256(FULL_OUT)
vo_macro_sha = sha256(VO_MACRO_OUT)
vo_task_sha = sha256(VO_TASK_OUT)

audit_text = f"""Experiment summary

Input: {INPUT}
Input SHA-256: {input_sha}
Protocol summary rows: {len(summary_rows)}
AC macro rows: {len(vo_macro_rows)}
Metric: trapezoidal precision-recall AUC.
Aggregation: mean and sample standard deviation across seeds 0, 1, 2.
Centralized: average q within each seed before summarizing across seeds.

Protocol summary: {FULL_OUT}
SHA-256: {full_sha}
AC macro summary: {VO_MACRO_OUT}
SHA-256: {vo_macro_sha}
AC task summary: {VO_TASK_OUT}
SHA-256: {vo_task_sha}
"""
AUDIT_OUT.write_text(audit_text, encoding="utf-8")
notes = f"""# AC Macro PR-AUC (%)

Mean ± sample standard deviation over seeds 0, 1, 2.
Global-Centralized pools q within each seed before summarizing across seeds.

{five_table}

{twenty_table}
"""
NOTES_OUT.write_text(notes, encoding="utf-8")
print(f"Wrote protocol, AC macro, and task summaries to {OUT_DIR}.")
