from fedptha.paths import frozen_root, generated_root

#!/usr/bin/env python3

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Paths
# ============================================================

ROOT = Path(".")
SUITE = ROOT / str(frozen_root())

MAIN_SUMMARY = ROOT / str(generated_root() / "tables") / "protocol_summary.csv"

OUT_DIR = ROOT / str(generated_root() / "atlases")

EXTRA_SUMMARY_OUT = OUT_DIR / "true_prauc_atlas_extra_summary.csv"

COMBINED_SUMMARY_OUT = OUT_DIR / "true_prauc_atlas_combined_summary.csv"

FIVE_PDF = OUT_DIR / "five_client_true_prauc_atlas.pdf"

TWENTY_PDF = OUT_DIR / "twenty_client_true_prauc_atlas.pdf"


# ============================================================
# Scientific contract
# ============================================================

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

PHASE_PREFIX = {
    "VV": "best_realistic_visible",
    "VO": "best_realistic_selection_oracle",
    "OO": "best_oracle_full",
}

EXPECTED_DEFINITION = "trapezoidal_precision_recall_curve_auc"

MAIN_ORDER = (
    "fully_local",
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
    "learned_simplex_postala": "Learned-Simplex + PostALA",
    "support_only_postala": "Support-Only + PostALA",
    "oracle_physical_community": "Oracle Physical-Community Isolation",
    "specialist_rep_transfer": "Specialist Representative Transfer",
    "oracle_hierarchical_taskfedavg_ala": "Oracle Hierarchical TaskFedAvg + ALA",
}

CLASS_FOR = {
    **{m: "Main" for m in MAIN_ORDER},
    "centralized": "Ref.",
    "learned_simplex_postala": "Abl.",
    "support_only_postala": "Abl.",
    "oracle_physical_community": "Collab.",
    "specialist_rep_transfer": "Collab.",
    "oracle_hierarchical_taskfedavg_ala": "Collab.",
}


# ============================================================
# Extra result locations
#
# These are NOT added to the final 10-method main registry.
# They are included only in this inspection atlas.
# ============================================================

EXTRAS = {
    "five_client": (
        (
            "learned_simplex_postala",
            SUITE / "main/five_client" / "learned_simplex_postala",
        ),
        (
            "support_only_postala",
            SUITE / "additions/five_client" / "support_only_postala",
        ),
    ),
    "twenty_client": (
        (
            "learned_simplex_postala",
            SUITE / "main/twenty_client" / "learned_simplex_postala",
        ),
        (
            "support_only_postala",
            SUITE / "additions/twenty_client" / "support_only_postala",
        ),
        (
            "oracle_physical_community",
            SUITE / "additions/twenty_client" / "oracle_physical_community",
        ),
        (
            "specialist_rep_transfer",
            SUITE / "additions/twenty_client" / "specialist_rep_transfer",
        ),
        (
            "oracle_hierarchical_taskfedavg_ala",
            SUITE / "additions/twenty_client" / "oracle_hierarchical_taskfedavg_ala",
        ),
    ),
}


# ============================================================
# Helpers
# ============================================================


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


def parse_task(value):
    s = str(value).strip().lower()

    if not s or s == "all":
        return None

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        s,
    )

    match = re.search(
        r"(?:cycle)?([2-6])$",
        compact,
    )

    if not match:
        return None

    return int(match.group(1))


def find_result_csvs(directory):
    if not directory.is_dir():
        raise RuntimeError(f"Missing expected directory: {directory}")

    result = []

    for path in sorted(directory.glob("*.csv")):
        try:
            with path.open(newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
        except Exception:
            continue

        header_set = {str(x).strip() for x in header}

        # Sidecars such as routing weights, cluster logs,
        # hierarchy weights, etc. should not satisfy this.
        if "phase" in header_set and "pr_auc_definition" in header_set:
            result.append(path)

    return result


def detect_seed(path, rows):
    candidates = (
        "seed",
        "model_seed",
        "random_seed",
    )

    for col in candidates:
        vals = {
            str(row.get(col, "")).strip()
            for row in rows
            if str(row.get(col, "")).strip()
        }

        if len(vals) == 1:
            try:
                return int(float(next(iter(vals))))
            except Exception:
                pass

    match = re.search(
        r"seed([0-9]+)",
        path.name,
    )

    if match:
        return int(match.group(1))

    match = re.search(
        r"(?:^|[_-])s([0-9]+)(?:[_\.-]|$)",
        path.name,
    )

    if match:
        return int(match.group(1))

    raise RuntimeError(f"Could not determine seed from {path}")


def infer_q_from_client_ids(text):
    ids = {
        int(x)
        for x in re.findall(
            r"(?<!\d)([34]\d{6})(?!\d)",
            text,
        )
    }

    # Only benchmark client ranges count.
    ids = {x for x in ids if (3000000 <= x <= 3000019 or 4000000 <= x <= 4000079)}

    if not ids:
        return None

    minimum = min(ids)

    if 3000000 <= minimum <= 3000004:
        return "0.20"

    if 3000005 <= minimum <= 3000009:
        return "0.50"

    if 3000010 <= minimum <= 3000014:
        return "0.80"

    if 3000015 <= minimum <= 3000019:
        return "0.95"

    if 4000000 <= minimum <= 4000019:
        return "0.20"

    if 4000020 <= minimum <= 4000039:
        return "0.50"

    if 4000040 <= minimum <= 4000059:
        return "0.80"

    if 4000060 <= minimum <= 4000079:
        return "0.95"

    return None


def detect_q(path, rows):
    candidates = (
        "q",
        "q_value",
        "heterogeneity_q",
    )

    for col in candidates:
        vals = set()

        for row in rows:
            value = finite_float(row.get(col, ""))

            if value is not None:
                vals.add(f"{value:.2f}")

        vals &= set(Q_VALUES)

        if len(vals) == 1:
            return next(iter(vals))

    # Explicit q20/q50/q80/q95 filename tags.
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

    provenance_fields = (
        "subset_clients",
        "client_id",
        "client",
        "receiver_client_id",
        "receiver_id",
        "graph_id",
        "dataset_id",
        "subset_id",
    )

    parts = [
        path.name,
    ]

    for row in rows:
        for field in provenance_fields:
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
        vals = [
            str(row.get(col, "")).strip()
            for row in task_rows
            if str(row.get(col, "")).strip()
        ]

        distinct = len(set(vals))

        if distinct > best_distinct:
            best_col = col
            best_distinct = distinct

    return best_col


def protocol_task_rows(rows, protocol):
    prefix = PHASE_PREFIX[protocol]
    out = []

    for row in rows:
        phase = str(row.get("phase", "")).strip()

        task = parse_task(row.get("task", ""))

        if not phase.startswith(prefix):
            continue

        if "test" not in phase:
            continue

        if task is None:
            continue

        out.append(
            (
                row,
                task,
            )
        )

    return out


def protocol_scalar_rows(rows, protocol):
    prefix = PHASE_PREFIX[protocol]
    out = []

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
            out.append(value)

    return out


def validate_metadata(path, rows):
    definitions = {
        str(
            row.get(
                "pr_auc_definition",
                "",
            )
        ).strip()
        for row in rows
        if str(
            row.get(
                "pr_auc_definition",
                "",
            )
        ).strip()
    }

    if definitions != {EXPECTED_DEFINITION}:
        raise RuntimeError(f"{path}: PR-AUC definition(s) " f"{definitions}")

    selector_values = set()

    for row in rows:
        for col in (
            "selector_metric",
            "selection_metric",
        ):
            value = str(row.get(col, "")).strip()

            if value:
                selector_values.add(value)
                break

    if selector_values and selector_values != {"macro_pr_auc"}:
        raise RuntimeError(f"{path}: selector(s) " f"{selector_values}")


def extract_seed_protocol_metrics(
    path,
    rows,
    protocol,
):
    validate_metadata(
        path,
        rows,
    )

    pairs = protocol_task_rows(
        rows,
        protocol,
    )

    if not pairs:
        raise RuntimeError(f"{path}: no {protocol} task rows")

    raw_rows = [row for row, _ in pairs]

    identity_col = detect_identity_column(raw_rows)

    # If no client identity exists, treat it as a
    # single system-level evaluation identity.
    by_identity = defaultdict(dict)

    for row, task in pairs:
        if identity_col is None:
            identity = "__system__"
        else:
            identity = str(
                row.get(
                    identity_col,
                    "",
                )
            ).strip()

            if not identity:
                identity = "__system__"

        value = finite_float(row.get("pr_auc", ""))

        if value is None:
            raise RuntimeError(f"{path}: undefined {protocol} " f"Cycle-{task} PR-AUC")

        if task in by_identity[identity]:
            raise RuntimeError(
                f"{path}: duplicate {protocol} "
                f"Cycle-{task} for identity "
                f"{identity}"
            )

        by_identity[identity][task] = value

    if not by_identity:
        raise RuntimeError(f"{path}: no usable {protocol} " "identity/task data")

    # Every evaluation identity should provide all 5 tasks.
    for identity, values in by_identity.items():
        if set(values) != {
            2,
            3,
            4,
            5,
            6,
        }:
            raise RuntimeError(
                f"{path}: {protocol} identity "
                f"{identity} has tasks "
                f"{sorted(values)}, expected 2-6"
            )

    task_system = {}

    for task in (
        2,
        3,
        4,
        5,
        6,
    ):
        values = [per_task[task] for per_task in by_identity.values()]

        task_system[task] = statistics.mean(values)

    client_macros = [
        statistics.mean(
            [
                per_task[task]
                for task in (
                    2,
                    3,
                    4,
                    5,
                    6,
                )
            ]
        )
        for per_task in by_identity.values()
    ]

    macro = statistics.mean(client_macros)

    # Cross-check against stored scalar Macro PR-AUC
    # whenever the file provides scalar rows.
    scalar_values = protocol_scalar_rows(
        rows,
        protocol,
    )

    if scalar_values:
        stored = statistics.mean(scalar_values)

        diff = abs(stored - macro)

        if diff > 5e-6:
            raise RuntimeError(
                f"{path}: {protocol} stored-vs-"
                f"recomputed Macro difference "
                f"{diff:.12e}"
            )

    return {
        "macro": macro,
        2: task_system[2],
        3: task_system[3],
        4: task_system[4],
        5: task_system[5],
        6: task_system[6],
    }


def aggregate_extra_method(
    setup,
    method,
    directory,
):
    files = find_result_csvs(directory)

    # Most extra methods store one result CSV per q x seed.
    #
    # Oracle Physical-Community Isolation is different:
    # each logical 20-client q x seed run is stored as four
    # separate 5-client community result CSVs.
    files_per_logical = 4 if method == "oracle_physical_community" else 1

    expected_total_files = len(Q_VALUES) * len(SEEDS) * files_per_logical

    if len(files) != expected_total_files:
        raise RuntimeError(
            f"{setup}/{method}: found "
            f"{len(files)} result CSVs, "
            f"expected {expected_total_files} "
            f"({files_per_logical} per q x seed)"
        )

    grouped_files = defaultdict(list)

    for path in files:
        rows = read_csv(path)

        q = detect_q(
            path,
            rows,
        )

        seed = detect_seed(
            path,
            rows,
        )

        if q not in Q_VALUES:
            raise RuntimeError(f"{path}: unexpected q={q}")

        if seed not in SEEDS:
            raise RuntimeError(f"{path}: unexpected seed={seed}")

        grouped_files[
            (
                q,
                seed,
            )
        ].append(
            (
                path,
                rows,
            )
        )

    expected_logical = {
        (
            q,
            seed,
        )
        for q in Q_VALUES
        for seed in SEEDS
    }

    if set(grouped_files) != expected_logical:
        missing = expected_logical - set(grouped_files)

        extra = set(grouped_files) - expected_logical

        raise RuntimeError(
            f"{setup}/{method}: logical "
            f"coverage mismatch; "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    for logical, physical in grouped_files.items():
        if len(physical) != files_per_logical:
            raise RuntimeError(
                f"{setup}/{method}/"
                f"q={logical[0]}/seed={logical[1]}: "
                f"{len(physical)} physical files, "
                f"expected {files_per_logical}"
            )

    # --------------------------------------------------------
    # Build one system-level result per q x seed x protocol.
    #
    # For Oracle Physical-Community Isolation:
    #   4 equal-sized 5-client communities are first averaged
    #   into one 20-client seed-level result.
    #
    # They are NOT treated as independent replicates.
    # --------------------------------------------------------

    seed_data = {}

    for q in Q_VALUES:
        for seed in SEEDS:
            physical = grouped_files[
                (
                    q,
                    seed,
                )
            ]

            seed_data[
                (
                    q,
                    seed,
                )
            ] = {}

            for protocol in PROTOCOLS:
                physical_metrics = []

                for path, rows in physical:
                    metrics = extract_seed_protocol_metrics(
                        path,
                        rows,
                        protocol,
                    )

                    physical_metrics.append(metrics)

                merged = {}

                for key in (
                    "macro",
                    2,
                    3,
                    4,
                    5,
                    6,
                ):
                    values = [metrics[key] for metrics in physical_metrics]

                    # For oracle physical-community isolation,
                    # all four physical communities contain
                    # exactly five clients, so their simple mean
                    # equals the 20-client mean.
                    merged[key] = statistics.mean(values)

                seed_data[
                    (
                        q,
                        seed,
                    )
                ][protocol] = merged

    # --------------------------------------------------------
    # Mean +/- sample SD across the three genuine model seeds.
    # --------------------------------------------------------

    output = []

    for q in Q_VALUES:
        for protocol in PROTOCOLS:
            seed_metrics = [
                seed_data[
                    (
                        q,
                        seed,
                    )
                ][protocol]
                for seed in SEEDS
            ]

            out = {
                "setup": setup,
                "method": method,
                "class": CLASS_FOR[method],
                "method_display": DISPLAY[method],
                "q": q,
                "protocol": protocol,
            }

            for key, stem in (
                (
                    "macro",
                    "macro_pr_auc",
                ),
                (
                    2,
                    "cycle2_pr_auc",
                ),
                (
                    3,
                    "cycle3_pr_auc",
                ),
                (
                    4,
                    "cycle4_pr_auc",
                ),
                (
                    5,
                    "cycle5_pr_auc",
                ),
                (
                    6,
                    "cycle6_pr_auc",
                ),
            ):
                values = [metrics[key] for metrics in seed_metrics]

                mean = statistics.mean(values)

                sd = statistics.stdev(values)

                out[f"{stem}_mean"] = mean

                out[f"{stem}_sample_sd"] = sd

            output.append(out)

    return output


def load_main_summary():
    rows = read_csv(MAIN_SUMMARY)

    if len(rows) != 240:
        raise RuntimeError(f"Main summary has {len(rows)} " "rows, expected 240")

    output = []

    methods_found = set()

    for row in rows:
        method = str(row["method"]).strip()

        if method == ("task_weighted_fedavg"):
            raise RuntimeError(
                "Task-Weighted FedAvg found " "in final 10-method summary"
            )

        methods_found.add(method)

        if method not in (set(MAIN_ORDER) | {"centralized"}):
            raise RuntimeError("Unexpected method in final " f"main summary: {method}")

        out = {
            "setup": row["setup"],
            "method": method,
            "class": CLASS_FOR[method],
            "method_display": DISPLAY[method],
            "q": f"{float(row['q']):.2f}",
            "protocol": row["protocol"],
        }

        for stem in (
            "macro_pr_auc",
            "cycle2_pr_auc",
            "cycle3_pr_auc",
            "cycle4_pr_auc",
            "cycle5_pr_auc",
            "cycle6_pr_auc",
        ):
            out[f"{stem}_mean"] = float(row[f"{stem}_mean"])

            out[f"{stem}_sample_sd"] = float(row[f"{stem}_sample_sd"])

        output.append(out)

    expected_methods = set(MAIN_ORDER) | {"centralized"}

    if methods_found != expected_methods:
        raise RuntimeError(
            "Main method set mismatch. " f"Found={sorted(methods_found)}"
        )

    return output


# ============================================================
# Build combined atlas summary
# ============================================================

print("=" * 78)
print("TRUE PR-AUC INSPECTION ATLAS BUILD")
print("=" * 78)

if not MAIN_SUMMARY.is_file():
    raise SystemExit(f"ABORT: missing {MAIN_SUMMARY}")

main_rows = load_main_summary()

extra_rows = []

for setup, methods in EXTRAS.items():
    for method, directory in methods:
        print(f"Extracting {setup}: " f"{method}")

        extracted = aggregate_extra_method(
            setup,
            method,
            directory,
        )

        extra_rows.extend(extracted)

print()
print(
    "Extra summary rows:",
    len(extra_rows),
)

# 5C: 2 extras * 4q * 3 protocols = 24
# 20C: 5 extras * 4q * 3 protocols = 60
# total = 84
if len(extra_rows) != 84:
    raise SystemExit("ABORT: extra summary rows=" f"{len(extra_rows)}, expected 84")

combined = main_rows + extra_rows

# Main = 240
# Extras = 84
# Total = 324
if len(combined) != 324:
    raise SystemExit("ABORT: combined rows=" f"{len(combined)}, expected 324")

if any(row["method"] == "task_weighted_fedavg" for row in combined):
    raise SystemExit("ABORT: Task-Weighted FedAvg " "appears in atlas data.")


# ============================================================
# Save audit-friendly CSVs
# ============================================================

FIELDS = [
    "setup",
    "class",
    "method",
    "method_display",
    "q",
    "protocol",
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
]


def write_rows(path, rows):
    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in FIELDS
                }
            )


write_rows(
    EXTRA_SUMMARY_OUT,
    extra_rows,
)

write_rows(
    COMBINED_SUMMARY_OUT,
    combined,
)


# ============================================================
# Atlas layout
# ============================================================

Q_LABEL = {
    "0.20": "balanced allocation / low specialization",
    "0.50": "moderate specialization",
    "0.80": "high specialization",
    "0.95": "extreme specialization",
}

PROTOCOL_LABEL = {
    "VV": "visible validation -> visible test",
    "VO": "visible validation -> complete test",
    "OO": "complete validation -> complete test",
}

SETUP_LABEL = {
    "five_client": "5-client",
    "twenty_client": "20-client",
}

CLASS_ORDER = {
    "Main": 0,
    "Abl.": 1,
    "Collab.": 2,
    "Ref.": 3,
}

METHOD_RANK = {method: rank for rank, method in enumerate(MAIN_ORDER)}

METHOD_RANK.update(
    {
        "learned_simplex_postala": 100,
        "support_only_postala": 101,
        "oracle_physical_community": 200,
        "specialist_rep_transfer": 201,
        "oracle_hierarchical_taskfedavg_ala": 202,
        "centralized": 300,
    }
)

METRIC_COLUMNS = [
    (
        "Macro PR-AUC",
        "macro_pr_auc",
    ),
    (
        "Cycle-2",
        "cycle2_pr_auc",
    ),
    (
        "Cycle-3",
        "cycle3_pr_auc",
    ),
    (
        "Cycle-4",
        "cycle4_pr_auc",
    ),
    (
        "Cycle-5",
        "cycle5_pr_auc",
    ),
    (
        "Cycle-6",
        "cycle6_pr_auc",
    ),
]


def cell_text(row, stem):
    mean = row[f"{stem}_mean"]

    sd = row[f"{stem}_sample_sd"]

    if mean is None or sd is None:
        return "--"

    return f"{100.0 * float(mean):.2f}" f" +/- " f"{100.0 * float(sd):.2f}"


def page_rows(
    setup,
    q,
    protocol,
):
    rows = [
        row
        for row in combined
        if (row["setup"] == setup and row["q"] == q and row["protocol"] == protocol)
    ]

    rows.sort(
        key=lambda row: (
            CLASS_ORDER[row["class"]],
            METHOD_RANK[row["method"]],
        )
    )

    expected = 12 if setup == "five_client" else 15

    if len(rows) != expected:
        raise RuntimeError(
            f"{setup}/q={q}/{protocol}: "
            f"{len(rows)} atlas rows, "
            f"expected {expected}"
        )

    return rows


def best_eligible_by_metric(rows):
    eligible = [
        row
        for row in rows
        if row["class"]
        in {
            "Main",
            "Abl.",
        }
    ]

    best = {}

    for _, stem in METRIC_COLUMNS:
        best_value = max(float(row[f"{stem}_mean"]) for row in eligible)

        best[stem] = best_value

    return best


def make_page(
    pdf,
    setup,
    q,
    protocol,
    page_no,
    total_pages,
):
    rows = page_rows(
        setup,
        q,
        protocol,
    )

    best = best_eligible_by_metric(rows)

    fig = plt.figure(
        figsize=(
            16,
            9,
        )
    )

    ax = fig.add_axes(
        [
            0.025,
            0.08,
            0.95,
            0.80,
        ]
    )

    ax.axis("off")

    title = (
        f"{SETUP_LABEL[setup]} true PR-AUC atlas"
        f" | q={q}"
        f" | {protocol}"
        f" | {Q_LABEL[q]}"
        f" | page {page_no}/{total_pages}"
    )

    fig.text(
        0.025,
        0.955,
        title,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )

    subtitle = (
        f"Protocol: {PROTOCOL_LABEL[protocol]}. "
        "Values are mean +/- sample SD over model-training seeds "
        "0, 1, 2. All values are true trapezoidal PR-AUC (%) "
        "from the Macro PR-AUC-selected model state."
    )

    fig.text(
        0.025,
        0.915,
        subtitle,
        fontsize=9.5,
        ha="left",
        va="top",
    )

    headers = [
        "Class",
        "Method",
    ] + [label for label, _ in METRIC_COLUMNS]

    table_data = []

    for row in rows:
        table_data.append(
            [
                row["class"],
                row["method_display"],
            ]
            + [
                cell_text(
                    row,
                    stem,
                )
                for _, stem in METRIC_COLUMNS
            ]
        )

    col_widths = [
        0.07,
        0.25,
        0.113,
        0.113,
        0.113,
        0.113,
        0.113,
        0.113,
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        loc="upper center",
    )

    table.auto_set_font_size(False)

    table.set_fontsize(8.4)

    table.scale(
        1.0,
        1.55,
    )

    # Header styling.
    for col in range(len(headers)):
        cell = table[
            0,
            col,
        ]

        cell.set_facecolor("#24384c")

        cell.set_text_props(
            color="white",
            weight="bold",
        )

        cell.set_edgecolor("#d0d7de")

        cell.set_linewidth(0.6)

    class_fill = {
        "Main": "#ffffff",
        "Abl.": "#fff6d9",
        "Collab.": "#eaf3f8",
        "Ref.": "#eeeeee",
    }

    # Body formatting.
    for row_idx, row in enumerate(
        rows,
        start=1,
    ):
        base_fill = class_fill[row["class"]]

        for col_idx in range(len(headers)):
            cell = table[
                row_idx,
                col_idx,
            ]

            cell.set_facecolor(base_fill)

            cell.set_edgecolor("#c9d1d9")

            cell.set_linewidth(0.5)

            if col_idx == 1:
                cell._loc = "left"
                cell.set_text_props(ha="left")

        # Highlight best Main/Abl metric cells.
        if row["class"] in {
            "Main",
            "Abl.",
        }:
            for metric_idx, (
                _,
                stem,
            ) in enumerate(
                METRIC_COLUMNS,
                start=2,
            ):
                value = float(row[f"{stem}_mean"])

                if math.isclose(
                    value,
                    best[stem],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    table[
                        row_idx,
                        metric_idx,
                    ].set_facecolor("#e2f2df")

                    table[
                        row_idx,
                        metric_idx,
                    ].set_text_props(weight="bold")

    # Stronger separator between classes.
    previous_class = None

    for row_idx, row in enumerate(
        rows,
        start=1,
    ):
        current_class = row["class"]

        if previous_class is not None and current_class != previous_class:
            for col_idx in range(len(headers)):
                table[
                    row_idx,
                    col_idx,
                ].set_linewidth(1.2)

        previous_class = current_class

    footer = (
        "Light green = highest mean among Main and Abl. rows "
        "for that metric. Collaboration diagnostics and the "
        "Global-Centralized reference are excluded from winner "
        "highlighting. Abl. rows are component ablations; Collab. "
        "rows are oracle/collaboration diagnostics and are not "
        "members of the final 10-method main comparison."
    )

    fig.text(
        0.025,
        0.035,
        footer,
        fontsize=8.2,
        ha="left",
        va="bottom",
    )

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(fig)


def build_atlas(
    setup,
    output,
):
    page_no = 0
    total_pages = 12

    with PdfPages(output) as pdf:
        # Match the previous atlas:
        # q outer loop, VV -> VO -> OO.
        for q in Q_VALUES:
            for protocol in PROTOCOLS:
                page_no += 1

                make_page(
                    pdf,
                    setup,
                    q,
                    protocol,
                    page_no,
                    total_pages,
                )

    return page_no


# ============================================================
# Generate PDFs
# ============================================================

five_pages = build_atlas(
    "five_client",
    FIVE_PDF,
)

twenty_pages = build_atlas(
    "twenty_client",
    TWENTY_PDF,
)

if five_pages != 12:
    raise SystemExit("ABORT: five-client PDF " f"pages={five_pages}, expected 12")

if twenty_pages != 12:
    raise SystemExit("ABORT: twenty-client PDF " f"pages={twenty_pages}, expected 12")


# ============================================================
# Final report
# ============================================================

print()
print("=" * 78)
print("PASS - TRUE PR-AUC INSPECTION ATLASES CREATED")
print("=" * 78)
print()
print("Main registry rows:")
print("  final 10-method analysis only")
print()
print("5C ablations:")
print("  Learned-Simplex + PostALA")
print("  Support-Only + PostALA")
print()
print("5C collaboration diagnostics:")
print("  none")
print()
print("20C ablations:")
print("  Learned-Simplex + PostALA")
print("  Support-Only + PostALA")
print()
print("20C collaboration diagnostics:")
print("  Oracle Physical-Community Isolation")
print("  Specialist Representative Transfer")
print("  Oracle Hierarchical TaskFedAvg + ALA")
print()
print("GCFL+ cluster logs:")
print("  not added as a separate method row")
print()
print("Task-Weighted FedAvg:")
print("  ABSENT")
print()
print("Pages:")
print("  five_client   : 12")
print("  twenty_client : 12")
print()
print("Created:")
print(" ", FIVE_PDF)
print(" ", TWENTY_PDF)
print(" ", EXTRA_SUMMARY_OUT)
print(" ", COMBINED_SUMMARY_OUT)
print()
print("Raw frozen result suite was not modified.")
print("=" * 78)
