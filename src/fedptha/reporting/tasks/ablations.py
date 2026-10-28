#!/usr/bin/env python3

from __future__ import annotations

from fedptha.paths import frozen_root, generated_root

from collections import OrderedDict, defaultdict
from pathlib import Path
import csv
import json
import math
import re
import statistics

import numpy as np
import pandas as pd

ROOT = Path(str(generated_root()))
SUITE = Path(str(frozen_root()))
MAIN_SEED_SOURCE = ROOT / "tables" / "main_protocol_results.csv"
MAIN_FROZEN_SUMMARY = ROOT / "tables" / "protocol_summary.csv"
EXTRA_FROZEN_SUMMARY = ROOT / "atlases" / "true_prauc_atlas_extra_summary.csv"

SEED_OUT = ROOT / "tables" / "ablation_seed_results.csv"
VO_SUMMARY_OUT = ROOT / "tables" / "ablation_ac_summary.csv"
CONTRAST_OUT = ROOT / "tables" / "ablation_paired_contrasts.csv"
Q80_TASK_OUT = ROOT / "tables" / "ablation_task_diagnostic.csv"

SETUPS = ("five_client", "twenty_client")
Q_VALUES = (0.20, 0.50, 0.80, 0.95)
SEEDS = (0, 1, 2)
PROTOCOLS = ("VV", "VO", "OO")

MAIN_METHODS = ("support_simplex", "support_simplex_postala")
EXTRA_METHODS = ("learned_simplex_postala", "support_only_postala")
FIGURE5_METHODS = (
    "support_simplex",
    "support_simplex_postala",
    "learned_simplex_postala",
    "support_only_postala",
)

DISPLAY = {
    "support_simplex": "Support-Simplex",
    "support_simplex_postala": "Support-Simplex + PostALA",
    "learned_simplex_postala": "Learned-Simplex + PostALA",
    "support_only_postala": "Support-Only + PostALA",
}
SOURCE_CLASS = {
    "support_simplex": "main",
    "support_simplex_postala": "main",
    "learned_simplex_postala": "ablation",
    "support_only_postala": "ablation",
}

EXPECTED_DEFINITION = "trapezoidal_precision_recall_curve_auc"

Q_LABELS = {"0.20", "0.50", "0.80", "0.95"}

PHASE_PREFIX = {
    "VV": "best_realistic_visible",
    "VO": "best_realistic_selection_oracle",
    "OO": "best_oracle_full",
}

EXTRA_DIRS = {
    ("five_client", "learned_simplex_postala"): SUITE
    / "main/five_client/learned_simplex_postala",
    ("five_client", "support_only_postala"): SUITE
    / "additions/five_client/support_only_postala",
    ("twenty_client", "learned_simplex_postala"): SUITE
    / "main/twenty_client/learned_simplex_postala",
    ("twenty_client", "support_only_postala"): SUITE
    / "additions/twenty_client/support_only_postala",
}
METRICS = OrderedDict(
    [
        ("macro_pr_auc", "macro"),
        ("cycle2_pr_auc", 2),
        ("cycle3_pr_auc", 3),
        ("cycle4_pr_auc", 4),
        ("cycle5_pr_auc", 5),
        ("cycle6_pr_auc", 6),
    ]
)

CONTRASTS = (
    (
        "support_simplex_postala_minus_support_simplex",
        "support_simplex_postala",
        "support_simplex",
    ),
    (
        "support_simplex_postala_minus_learned_simplex_postala",
        "support_simplex_postala",
        "learned_simplex_postala",
    ),
    (
        "support_simplex_postala_minus_support_only_postala",
        "support_simplex_postala",
        "support_only_postala",
    ),
    (
        "support_only_postala_minus_learned_simplex_postala",
        "support_only_postala",
        "learned_simplex_postala",
    ),
)


def abort(message: str) -> None:
    raise SystemExit(f"\nABORT: {message}\n")


def require_file(path: Path) -> None:
    if not path.is_file():
        abort(f"missing required file: {path}")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    try:
        value = float(text)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def parse_task(value):
    text = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    if not text or text == "all":
        return None
    match = re.search(r"(?:cycle)?([2-6])$", text)
    return int(match.group(1)) if match else None


def find_result_csvs(directory):
    if not directory.is_dir():
        raise RuntimeError(f"Missing expected directory: {directory}")
    result = []
    for path in sorted(directory.glob("*.csv")):
        try:
            with path.open(newline="") as f:
                header = next(csv.reader(f))
        except Exception:
            continue
        columns = {str(x).strip() for x in header}
        if "phase" in columns and "pr_auc_definition" in columns:
            result.append(path)
    return result


def detect_seed(path, rows):
    for col in ("seed", "model_seed", "random_seed"):
        values = {
            str(row.get(col, "")).strip()
            for row in rows
            if str(row.get(col, "")).strip()
        }
        if len(values) == 1:
            try:
                return int(float(next(iter(values))))
            except Exception:
                pass

    match = re.search(r"seed([0-9]+)", path.name)
    if match:
        return int(match.group(1))

    match = re.search(r"(?:^|[_-])s([0-9]+)(?:[_\.-]|$)", path.name)
    if match:
        return int(match.group(1))

    raise RuntimeError(f"Could not determine seed from {path}")


def infer_q_from_client_ids(text):
    ids = {int(x) for x in re.findall(r"(?<!\d)([34]\d{6})(?!\d)", text)}
    ids = {x for x in ids if 3000000 <= x <= 3000019 or 4000000 <= x <= 4000079}
    if not ids:
        return None

    value = min(ids)

    if 3000000 <= value <= 3000004:
        return "0.20"
    if 3000005 <= value <= 3000009:
        return "0.50"
    if 3000010 <= value <= 3000014:
        return "0.80"
    if 3000015 <= value <= 3000019:
        return "0.95"
    if 4000000 <= value <= 4000019:
        return "0.20"
    if 4000020 <= value <= 4000039:
        return "0.50"
    if 4000040 <= value <= 4000059:
        return "0.80"
    if 4000060 <= value <= 4000079:
        return "0.95"

    return None


def detect_q(path, rows):
    for col in ("q", "q_value", "heterogeneity_q"):
        values = set()
        for row in rows:
            value = finite_float(row.get(col, ""))
            if value is not None:
                values.add(f"{value:.2f}")
        values &= Q_LABELS
        if len(values) == 1:
            return next(iter(values))

    tags = {
        "q20": "0.20",
        "q50": "0.50",
        "q80": "0.80",
        "q95": "0.95",
    }

    lower_name = path.name.lower()
    for tag, q in tags.items():
        if tag in lower_name:
            return q

    fields = (
        "subset_clients",
        "client_id",
        "client",
        "receiver_client_id",
        "receiver_id",
        "graph_id",
        "dataset_id",
        "subset_id",
    )

    parts = [path.name]
    for row in rows:
        for field in fields:
            parts.append(str(row.get(field, "")))

    q = infer_q_from_client_ids(" ".join(parts))
    if q is None:
        raise RuntimeError(f"Could not determine q from {path}")
    return q


def detect_identity_column(task_rows):
    candidates = (
        "client_id",
        "receiver_client_id",
        "receiver_id",
        "client_idx",
        "client_index",
        "client",
        "receiver",
        "graph_id",
    )

    best_col = None
    best_distinct = 0

    for col in candidates:
        values = [
            str(row.get(col, "")).strip()
            for row in task_rows
            if str(row.get(col, "")).strip()
        ]
        distinct = len(set(values))
        if distinct > best_distinct:
            best_col = col
            best_distinct = distinct

    return best_col


def protocol_task_rows(rows, protocol):
    prefix = PHASE_PREFIX[protocol]
    output = []

    for row in rows:
        phase = str(row.get("phase", "")).strip()
        task = parse_task(row.get("task", ""))

        if not phase.startswith(prefix):
            continue
        if "test" not in phase:
            continue
        if task is None:
            continue

        output.append((row, task))

    return output


def protocol_scalar_rows(rows, protocol):
    prefix = PHASE_PREFIX[protocol]
    output = []

    for row in rows:
        phase = str(row.get("phase", "")).strip()
        task_raw = str(row.get("task", "")).strip().lower()

        if not phase.startswith(prefix):
            continue
        if "test" not in phase:
            continue
        if "task" in phase.split("_")[-1:]:
            continue
        if phase.endswith("_task"):
            continue
        if phase.endswith("_mean"):
            continue
        if task_raw not in ("", "all"):
            continue

        value = finite_float(row.get("macro_pr_auc", ""))
        if value is not None:
            output.append(value)

    return output


def validate_metadata(path, rows):
    definitions = {
        str(row.get("pr_auc_definition", "")).strip()
        for row in rows
        if str(row.get("pr_auc_definition", "")).strip()
    }

    if definitions != {EXPECTED_DEFINITION}:
        raise RuntimeError(f"{path}: PR-AUC definition(s) {definitions}")

    selectors = set()

    for row in rows:
        for col in ("selector_metric", "selection_metric"):
            value = str(row.get(col, "")).strip()
            if value:
                selectors.add(value)
                break

    if selectors and selectors != {"macro_pr_auc"}:
        raise RuntimeError(f"{path}: selector(s) {selectors}")


def extract_seed_protocol_metrics(path, rows, protocol):
    validate_metadata(path, rows)

    pairs = protocol_task_rows(rows, protocol)
    if not pairs:
        raise RuntimeError(f"{path}: no {protocol} task rows")

    identity_col = detect_identity_column([row for row, _ in pairs])
    by_identity = defaultdict(dict)

    for row, task in pairs:
        if identity_col is None:
            identity = "__system__"
        else:
            identity = str(row.get(identity_col, "")).strip() or "__system__"

        value = finite_float(row.get("pr_auc", ""))
        if value is None:
            raise RuntimeError(f"{path}: undefined {protocol} Cycle-{task} PR-AUC")

        if task in by_identity[identity]:
            raise RuntimeError(
                f"{path}: duplicate {protocol} Cycle-{task} " f"for identity {identity}"
            )

        by_identity[identity][task] = value

    expected_tasks = {2, 3, 4, 5, 6}

    for identity, values in by_identity.items():
        if set(values) != expected_tasks:
            raise RuntimeError(
                f"{path}: {protocol} identity {identity} has tasks "
                f"{sorted(values)}, expected 2-6"
            )

    task_system = {
        task: statistics.mean(per_task[task] for per_task in by_identity.values())
        for task in (2, 3, 4, 5, 6)
    }

    macro = statistics.mean(
        statistics.mean(per_task[task] for task in (2, 3, 4, 5, 6))
        for per_task in by_identity.values()
    )

    scalar_values = protocol_scalar_rows(rows, protocol)

    if scalar_values:
        stored = statistics.mean(scalar_values)
        diff = abs(stored - macro)
        if diff > 5e-6:
            raise RuntimeError(
                f"{path}: {protocol} stored-vs-recomputed "
                f"Macro difference {diff:.12e}"
            )

    return {
        "macro": macro,
        2: task_system[2],
        3: task_system[3],
        4: task_system[4],
        5: task_system[5],
        6: task_system[6],
    }


def metric_value(metrics: dict, key) -> float:
    if key in metrics:
        return float(metrics[key])
    skey = str(key)
    if skey in metrics:
        return float(metrics[skey])
    abort(
        f"extract_seed_protocol_metrics returned no key {key!r}; "
        f"available keys={list(metrics.keys())}"
    )


def seed_json(group: pd.DataFrame, value_col: str) -> str:
    group = group.sort_values("seed")
    values = {
        str(int(row.seed)): round(float(getattr(row, value_col)), 12)
        for row in group.itertuples(index=False)
    }
    return json.dumps(values, sort_keys=True)


def mean_sd(values):
    values = [float(x) for x in values]
    if len(values) != 3:
        abort(f"expected 3 seed values, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


def pp(value: float) -> float:
    return 100.0 * float(value)


print("=" * 104)
print("FIGURE 5 - ABLATION AUDIT + VO ANALYSIS")
print("=" * 104)

for path in (
    MAIN_SEED_SOURCE,
    MAIN_FROZEN_SUMMARY,
    EXTRA_FROZEN_SUMMARY,
):
    require_file(path)

print("\nInputs:")
print(f"  main seed source     : {MAIN_SEED_SOURCE}")
print(f"  main frozen summary : {MAIN_FROZEN_SUMMARY}")
print("  extra raw extractor : audit_figure5_ablations.py")
print(f"  extra frozen summary: {EXTRA_FROZEN_SUMMARY}")

# -----------------------------------------------------------------------------
# A. Main variants: load seed-level frozen canonical rows.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP A - MAIN-VARIANT SEED ROWS")
print("=" * 104)

main = pd.read_csv(MAIN_SEED_SOURCE)
main["q"] = pd.to_numeric(main["q"], errors="raise").round(2)
required_main = {
    "setup",
    "method",
    "q",
    "seed",
    "protocol",
    *METRICS.keys(),
    "pr_auc_definition",
    "selector_metric",
    "source_result_paths",
}
missing = sorted(required_main - set(main.columns))
if missing:
    abort("main canonical source missing columns: " + ", ".join(missing))

main = main[main["method"].isin(MAIN_METHODS)].copy()
expected_main_rows = 2 * 2 * 4 * 3 * 3
print(f"Rows found: {len(main)} / {expected_main_rows}")
if len(main) != expected_main_rows:
    abort("unexpected number of main-variant rows")
if main.duplicated(["setup", "method", "q", "seed", "protocol"]).any():
    abort("duplicate logical keys in main canonical source")
if set(main["setup"]) != set(SETUPS):
    abort("main setup registry mismatch")
if set(main["method"]) != set(MAIN_METHODS):
    abort("main method registry mismatch")
if set(main["q"]) != set(Q_VALUES):
    abort("main q registry mismatch")
if set(main["seed"]) != set(SEEDS):
    abort("main seed registry mismatch")
if set(main["protocol"]) != set(PROTOCOLS):
    abort("main protocol registry mismatch")
if set(main["pr_auc_definition"].dropna().astype(str)) != {EXPECTED_DEFINITION}:
    abort("unexpected main PR-AUC definition")
if set(main["selector_metric"].dropna().astype(str)) != {"macro_pr_auc"}:
    abort("unexpected main selector metric")

for metric in METRICS:
    values = pd.to_numeric(main[metric], errors="coerce")
    if values.isna().any() or ((values < 0.0) | (values > 1.0)).any():
        abort(f"invalid main metric values in {metric}")

main_seed_rows = []
metadata_cols = (
    "selection_scope",
    "selected_round_values",
    "selected_epoch_values",
    "selection_provenance_json",
)
for row in main.itertuples(index=False):
    out = {
        "setup": row.setup,
        "method": row.method,
        "method_display": DISPLAY[row.method],
        "source_class": SOURCE_CLASS[row.method],
        "q": float(row.q),
        "seed": int(row.seed),
        "protocol": row.protocol,
        "pr_auc_definition": row.pr_auc_definition,
        "selector_metric": row.selector_metric,
        "source_result_paths": row.source_result_paths,
        "extraction_source": "main_protocol_results.csv",
    }
    for metric in METRICS:
        out[metric] = float(getattr(row, metric))
    for col in metadata_cols:
        out[col] = getattr(row, col) if col in main.columns else np.nan
    main_seed_rows.append(out)

print("Logical coverage   : PASS")
print("PR-AUC definition : true trapezoidal PR-AUC")

# -----------------------------------------------------------------------------
# B. Reproduce main frozen summary.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP B - REPRODUCE MAIN FROZEN SUMMARY")
print("=" * 104)

main_seed_df = pd.DataFrame(main_seed_rows)
main_summary = pd.read_csv(MAIN_FROZEN_SUMMARY)
main_summary["q"] = pd.to_numeric(main_summary["q"], errors="raise").round(2)
main_summary = main_summary[main_summary["method"].isin(MAIN_METHODS)].copy()
if len(main_summary) != 48:
    abort(f"expected 48 main frozen summary rows, found {len(main_summary)}")

rep_main_rows = []
for keys, group in main_seed_df.groupby(
    ["setup", "method", "q", "protocol"], sort=True
):
    if len(group) != 3:
        abort(f"main group {keys}: expected 3 seeds, found {len(group)}")
    out = dict(zip(["setup", "method", "q", "protocol"], keys))
    for metric in METRICS:
        mean, sd = mean_sd(group[metric].tolist())
        out[f"{metric}_mean"] = mean
        out[f"{metric}_sample_sd"] = sd
    rep_main_rows.append(out)
rep_main = pd.DataFrame(rep_main_rows)
merged_main = main_summary.merge(
    rep_main,
    on=["setup", "method", "q", "protocol"],
    suffixes=("_frozen", "_reproduced"),
    validate="one_to_one",
)
if len(merged_main) != 48:
    abort("main frozen/reproduced logical keys differ")
max_main_error = 0.0
for metric in METRICS:
    for stat in ("mean", "sample_sd"):
        a = pd.to_numeric(merged_main[f"{metric}_{stat}_frozen"], errors="raise")
        b = pd.to_numeric(merged_main[f"{metric}_{stat}_reproduced"], errors="raise")
        max_main_error = max(max_main_error, float((a - b).abs().max()))
print(f"Frozen rows         : {len(main_summary)} / 48")
print(f"Reproduced rows     : {len(rep_main)} / 48")
print(f"Max absolute error  : {max_main_error:.3e}")
if max_main_error > 5e-10:
    abort("main variants do not reproduce frozen Step-3 summary")
print("Main reproduction   : PASS")

# -----------------------------------------------------------------------------
# C. Extra ablations: use verified raw extractor functions.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP C - EXTRA-ABLATION RAW SEED EXTRACTION")
print("=" * 104)

targets = [
    (setup, method, Path(directory))
    for (setup, method), directory in EXTRA_DIRS.items()
]
if len(targets) != 4:
    abort(f"expected 4 extra-ablation directories, resolved {len(targets)}")

extra_seed_rows = []
for setup, method, directory in targets:
    if not directory.is_dir():
        abort(f"missing raw directory: {directory}")
    files = find_result_csvs(directory)
    if len(files) != 12:
        abort(f"{setup}/{method}: expected 12 raw result CSVs, found {len(files)}")

    logical_keys = set()
    for path in files:
        rows = read_csv(path)
        if not rows:
            abort(f"empty raw result file: {path}")
        q = round(float(detect_q(path, rows)), 2)
        seed = int(detect_seed(path, rows))
        if (q, seed) in logical_keys:
            abort(f"{setup}/{method}: duplicate q={q}, seed={seed}")
        logical_keys.add((q, seed))

        for protocol in PROTOCOLS:
            metrics = extract_seed_protocol_metrics(path, rows, protocol)
            out = {
                "setup": setup,
                "method": method,
                "method_display": DISPLAY[method],
                "source_class": SOURCE_CLASS[method],
                "q": q,
                "seed": seed,
                "protocol": protocol,
                "pr_auc_definition": EXPECTED_DEFINITION,
                "selector_metric": "macro_pr_auc",
                "source_result_paths": str(path),
                "extraction_source": (
                    "audit_figure5_ablations.py:" "extract_seed_protocol_metrics"
                ),
                "selection_scope": np.nan,
                "selected_round_values": np.nan,
                "selected_epoch_values": np.nan,
                "selection_provenance_json": np.nan,
            }
            for metric, key in METRICS.items():
                out[metric] = metric_value(metrics, key)
            extra_seed_rows.append(out)

    expected_keys = {(q, seed) for q in Q_VALUES for seed in SEEDS}
    if logical_keys != expected_keys:
        abort(f"{setup}/{method}: raw q x seed coverage mismatch")
    print(f"{setup:<14} {method:<28} 12 files | 12/12 q x seed keys")

extra_seed_df = pd.DataFrame(extra_seed_rows)
if len(extra_seed_df) != 144:
    abort(f"expected 144 extra seed/protocol rows, found {len(extra_seed_df)}")
if extra_seed_df.duplicated(["setup", "method", "q", "seed", "protocol"]).any():
    abort("duplicate extra-ablation logical keys")
print(f"Extra seed/protocol rows: {len(extra_seed_df)} / 144")

# -----------------------------------------------------------------------------
# D. Reproduce extra frozen atlas summary.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP D - REPRODUCE EXTRA FROZEN SUMMARY")
print("=" * 104)

extra_summary = pd.read_csv(EXTRA_FROZEN_SUMMARY)
extra_summary["q"] = pd.to_numeric(extra_summary["q"], errors="raise").round(2)
extra_summary = extra_summary[extra_summary["method"].isin(EXTRA_METHODS)].copy()
if len(extra_summary) != 48:
    abort(f"expected 48 extra frozen summary rows, found {len(extra_summary)}")

rep_extra_rows = []
for keys, group in extra_seed_df.groupby(
    ["setup", "method", "q", "protocol"], sort=True
):
    if len(group) != 3:
        abort(f"extra group {keys}: expected 3 seeds, found {len(group)}")
    out = dict(zip(["setup", "method", "q", "protocol"], keys))
    for metric in METRICS:
        mean, sd = mean_sd(group[metric].tolist())
        out[f"{metric}_mean"] = mean
        out[f"{metric}_sample_sd"] = sd
    rep_extra_rows.append(out)
rep_extra = pd.DataFrame(rep_extra_rows)
merged_extra = extra_summary.merge(
    rep_extra,
    on=["setup", "method", "q", "protocol"],
    suffixes=("_frozen", "_reproduced"),
    validate="one_to_one",
)
if len(merged_extra) != 48:
    abort("extra frozen/reproduced logical keys differ")
max_extra_error = 0.0
for metric in METRICS:
    for stat in ("mean", "sample_sd"):
        a = pd.to_numeric(merged_extra[f"{metric}_{stat}_frozen"], errors="raise")
        b = pd.to_numeric(merged_extra[f"{metric}_{stat}_reproduced"], errors="raise")
        max_extra_error = max(max_extra_error, float((a - b).abs().max()))
print(f"Frozen rows         : {len(extra_summary)} / 48")
print(f"Reproduced rows     : {len(rep_extra)} / 48")
print(f"Max absolute error  : {max_extra_error:.3e}")
if max_extra_error > 5e-10:
    abort("extra variants do not reproduce frozen atlas summary")
print("Extra reproduction  : PASS")

# -----------------------------------------------------------------------------
# E. Combine all four methods into one 288-row seed/protocol table.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP E - COMPLETE FOUR-METHOD SEED TABLE")
print("=" * 104)

seed_df = pd.concat([main_seed_df, extra_seed_df], ignore_index=True, sort=False)
if len(seed_df) != 288:
    abort(f"expected 288 combined seed/protocol rows, found {len(seed_df)}")
if seed_df.duplicated(["setup", "method", "q", "seed", "protocol"]).any():
    abort("duplicate logical keys in combined Figure-5 seed table")
if set(seed_df["setup"]) != set(SETUPS):
    abort("combined setup registry mismatch")
if set(seed_df["method"]) != set(FIGURE5_METHODS):
    abort("combined method registry mismatch")
if set(seed_df["q"]) != set(Q_VALUES):
    abort("combined q registry mismatch")
if set(seed_df["seed"]) != set(SEEDS):
    abort("combined seed registry mismatch")
if set(seed_df["protocol"]) != set(PROTOCOLS):
    abort("combined protocol registry mismatch")
seed_df = seed_df.sort_values(["setup", "q", "method", "seed", "protocol"]).reset_index(
    drop=True
)
SEED_OUT.parent.mkdir(parents=True, exist_ok=True)
seed_df.to_csv(SEED_OUT, index=False)
print(f"Rows    : {len(seed_df)} / 288")
print(f"Created : {SEED_OUT}")

# -----------------------------------------------------------------------------
# F. VO-only 32-row summary.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP F - VO SUMMARY")
print("=" * 104)

vo = seed_df[seed_df["protocol"] == "VO"].copy()
vo_summary_rows = []
for keys, group in vo.groupby(["setup", "method", "q"], sort=True):
    setup, method, q = keys
    if len(group) != 3:
        abort(f"VO group {keys}: expected 3 seeds, found {len(group)}")
    mean, sd = mean_sd(group["macro_pr_auc"].tolist())
    vo_summary_rows.append(
        {
            "setup": setup,
            "method": method,
            "method_display": DISPLAY[method],
            "source_class": SOURCE_CLASS[method],
            "q": q,
            "protocol": "VO",
            "n_model_seeds": 3,
            "macro_pr_auc_mean": mean,
            "macro_pr_auc_sample_sd": sd,
            "macro_seed_values_json": seed_json(group, "macro_pr_auc"),
        }
    )
vo_summary = pd.DataFrame(vo_summary_rows)
if len(vo_summary) != 32:
    abort(f"expected 32 VO summary rows, found {len(vo_summary)}")
vo_summary = vo_summary.sort_values(["setup", "q", "method"]).reset_index(drop=True)
vo_summary.to_csv(VO_SUMMARY_OUT, index=False)
print(f"Rows    : {len(vo_summary)} / 32")
print(f"Created : {VO_SUMMARY_OUT}")

# -----------------------------------------------------------------------------
# G. Matched-seed VO contrasts.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP G - MATCHED-SEED VO CONTRASTS")
print("=" * 104)

contrast_rows = []
for setup in SETUPS:
    for q in Q_VALUES:
        cell = vo[(vo["setup"] == setup) & (vo["q"] == q)].copy()
        if len(cell) != 12:
            abort(f"{setup}, q={q:.2f}: expected 12 VO rows, found {len(cell)}")

        for contrast_name, lhs_method, rhs_method in CONTRASTS:
            lhs = cell[cell["method"] == lhs_method][["seed", "macro_pr_auc"]].rename(
                columns={"macro_pr_auc": "lhs"}
            )
            rhs = cell[cell["method"] == rhs_method][["seed", "macro_pr_auc"]].rename(
                columns={"macro_pr_auc": "rhs"}
            )
            paired = lhs.merge(rhs, on="seed", validate="one_to_one").sort_values(
                "seed"
            )
            if len(paired) != 3:
                abort(f"{setup}, q={q:.2f}, {contrast_name}: expected 3 matched seeds")
            paired["difference"] = paired["lhs"] - paired["rhs"]
            mean_diff, sd_diff = mean_sd(paired["difference"].tolist())
            seed_diffs = {
                str(int(row.seed)): round(float(row.difference), 12)
                for row in paired.itertuples(index=False)
            }
            contrast_rows.append(
                {
                    "setup": setup,
                    "q": q,
                    "protocol": "VO",
                    "contrast": contrast_name,
                    "lhs_method": lhs_method,
                    "lhs_display": DISPLAY[lhs_method],
                    "rhs_method": rhs_method,
                    "rhs_display": DISPLAY[rhs_method],
                    "n_matched_seeds": 3,
                    "difference_mean": mean_diff,
                    "difference_sample_sd": sd_diff,
                    "difference_mean_pp": pp(mean_diff),
                    "difference_sample_sd_pp": pp(sd_diff),
                    "seed_differences_json": json.dumps(seed_diffs, sort_keys=True),
                    "interpretation_rule": (
                        "descriptive matched-training-seed difference; positive means lhs "
                        "has higher AC Macro PR-AUC; not same-state and not significance"
                    ),
                }
            )
contrasts = pd.DataFrame(contrast_rows)
if len(contrasts) != 32:
    abort(f"expected 32 contrast rows, found {len(contrasts)}")
contrasts.to_csv(CONTRAST_OUT, index=False)
print(f"Rows    : {len(contrasts)} / 32")
print(f"Created : {CONTRAST_OUT}")

# -----------------------------------------------------------------------------
# H. 5C q=.80 task-level diagnostic.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP H - 5C q=.80 TASK DIAGNOSTIC")
print("=" * 104)

q80 = vo[(vo["setup"] == "five_client") & (vo["q"] == 0.80)].copy()
q80_rows = []
for method in FIGURE5_METHODS:
    method_rows = q80[q80["method"] == method].sort_values("seed")
    if len(method_rows) != 3:
        abort(f"5C q=.80 {method}: expected 3 seeds, found {len(method_rows)}")
    for metric in METRICS:
        mean, sd = mean_sd(method_rows[metric].tolist())
        q80_rows.append(
            {
                "setup": "five_client",
                "q": 0.80,
                "protocol": "VO",
                "method": method,
                "method_display": DISPLAY[method],
                "metric": metric,
                "mean": mean,
                "sample_sd": sd,
                "mean_percent": pp(mean),
                "sample_sd_percent": pp(sd),
                "seed_values_json": seed_json(method_rows, metric),
            }
        )
q80_task = pd.DataFrame(q80_rows)
if len(q80_task) != 24:
    abort(f"expected 24 5C q=.80 diagnostic rows, found {len(q80_task)}")
q80_task.to_csv(Q80_TASK_OUT, index=False)
print(f"Rows    : {len(q80_task)} / 24")
print(f"Created : {Q80_TASK_OUT}")

# -----------------------------------------------------------------------------
# I. Human-readable all-q diagnostics.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP I - ALL-q VO DIAGNOSTICS")
print("=" * 104)

for setup in SETUPS:
    print("\n" + "#" * 104)
    print(setup.upper())
    print("#" * 104)
    for q in Q_VALUES:
        print(f"\nq={q:.2f}\n" + "-" * 104)
        cell = vo_summary[
            (vo_summary["setup"] == setup) & (vo_summary["q"] == q)
        ].sort_values("macro_pr_auc_mean", ascending=False)
        for rank, row in enumerate(cell.itertuples(index=False), start=1):
            seed_map = json.loads(row.macro_seed_values_json)
            seed_text = ", ".join(
                f"{100.0 * float(seed_map[str(seed)]):.2f}" for seed in SEEDS
            )
            print(
                f"  {rank}. {row.method_display:<29} "
                f"{pp(row.macro_pr_auc_mean):6.2f} +/- "
                f"{pp(row.macro_pr_auc_sample_sd):5.2f}   seeds=[{seed_text}]"
            )

        print("\n  Matched-seed contrasts (lhs - rhs, percentage points):")
        ccell = contrasts[(contrasts["setup"] == setup) & (contrasts["q"] == q)]
        for row in ccell.itertuples(index=False):
            seed_map = json.loads(row.seed_differences_json)
            seed_text = ", ".join(
                f"{100.0 * float(seed_map[str(seed)]):+.2f}" for seed in SEEDS
            )
            print(
                f"    {row.lhs_display} - {row.rhs_display}: "
                f"{row.difference_mean_pp:+.2f} +/- {row.difference_sample_sd_pp:.2f} pp   "
                f"seeds=[{seed_text}]"
            )

# -----------------------------------------------------------------------------
# J. Focused 5C q=.80 diagnostic.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("STEP J - FOCUSED CHECK: 5C q=.80")
print("=" * 104)

for method in ("support_simplex_postala", "support_only_postala"):
    row = vo_summary[
        (vo_summary["setup"] == "five_client")
        & (vo_summary["q"] == 0.80)
        & (vo_summary["method"] == method)
    ].iloc[0]
    print(
        f"{DISPLAY[method]:<30}: "
        f"{pp(row['macro_pr_auc_mean']):.2f} +/- "
        f"{pp(row['macro_pr_auc_sample_sd']):.2f}"
    )

focus = contrasts[
    (contrasts["setup"] == "five_client")
    & (contrasts["q"] == 0.80)
    & (contrasts["contrast"] == "support_simplex_postala_minus_support_only_postala")
].iloc[0]
focus_seed_diffs = json.loads(focus["seed_differences_json"])
print("\nSupport-Simplex + PostALA minus Support-Only + PostALA:")
print(
    f"  mean +/- SD : {focus['difference_mean_pp']:+.2f} +/- "
    f"{focus['difference_sample_sd_pp']:.2f} pp"
)
print(
    "  seed deltas : "
    + ", ".join(
        f"seed {seed}={100.0 * float(focus_seed_diffs[str(seed)]):+.2f}"
        for seed in SEEDS
    )
)

print("\nTask-level means at 5C q=.80:")
for metric in METRICS:
    print(f"\n  {metric}")
    rows = q80_task[q80_task["metric"] == metric].sort_values("mean", ascending=False)
    for row in rows.itertuples(index=False):
        print(
            f"    {row.method_display:<29} "
            f"{row.mean_percent:6.2f} +/- {row.sample_sd_percent:5.2f}"
        )

# -----------------------------------------------------------------------------
# Final status.
# -----------------------------------------------------------------------------
print("\n" + "=" * 104)
print("PASS - FIGURE 5 NUMERICAL ABLATION AUDIT")
print("=" * 104)
print(f"\nMain seed/protocol rows       : {len(main_seed_df)} / 144")
print(f"Extra seed/protocol rows      : {len(extra_seed_df)} / 144")
print(f"Combined Figure-5 seed rows   : {len(seed_df)} / 288")
print(f"VO summary rows               : {len(vo_summary)} / 32")
print(f"Matched-seed contrast rows    : {len(contrasts)} / 32")
print(f"5C q=.80 task diagnostic rows : {len(q80_task)} / 24")
print(f"Main frozen-summary max error : {max_main_error:.3e}")
print(f"Extra frozen-summary max error: {max_extra_error:.3e}")
print("True trapezoidal PR-AUC       : CONFIRMED")
print("\nCreated:")
print(f"  {SEED_OUT}")
print(f"  {VO_SUMMARY_OUT}")
print(f"  {CONTRAST_OUT}")
print(f"  {Q80_TASK_OUT}")
print("\nIMPORTANT:")
print(
    "  This audit establishes numerical provenance and matched-seed descriptive contrasts."
)
print(
    "  It does NOT by itself prove that the method variants isolate exactly one implementation component."
)
print(
    "  Causal component language remains blocked until method/configuration differences are inspected."
)
print("\nNo raw result, frozen result, atlas summary, or model output was modified.")
print("=" * 104)
