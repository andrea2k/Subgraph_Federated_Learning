from fedptha.paths import frozen_root, generated_root

#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

SUITE = Path(str(frozen_root()))
FREEZE = SUITE / "manifests" / "frozen_main_comparison_10.csv"
FREEZE_MD = SUITE / "manifests" / "FROZEN_MAIN_COMPARISON_10.md"
CANON_SHA = SUITE / "manifests" / "canonical_artifacts_sha256.csv"

OUT_DIR = Path(str(generated_root() / "tables"))
MASTER_OUT = OUT_DIR / "main_protocol_results.csv"
CLIENT_OUT = OUT_DIR / "client_metrics.csv"
AUDIT_OUT = OUT_DIR / "extraction_audit.txt"

Q_VALUES = ("0.20", "0.50", "0.80", "0.95")
SEEDS = (0, 1, 2)
PROTOCOLS = ("VV", "VO", "OO")

METHOD_ORDER = (
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

PROTOCOL_PREFIX = {
    "VV": "best_realistic_visible",
    "VO": "best_realistic_selection_oracle",
    "OO": "best_oracle_full",
}

TASKS = ("cycle2", "cycle3", "cycle4", "cycle5", "cycle6")

IDENTITY_CANDIDATES = (
    "client_id",
    "client_idx",
    "client_index",
    "client",
    "client_name",
    "receiver_client_id",
    "receiver_id",
    "receiver",
    "graph_id",
    "dataset_id",
)

MACRO_TOL = 5e-7

errors = []
master_rows = []
client_rows = []


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    s = str(value).strip()

    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None

    try:
        x = float(s)
    except Exception:
        return None

    return x if math.isfinite(x) else None


def mean_defined(values):
    vals = [x for x in values if x is not None]
    return statistics.mean(vals) if vals else None


def fmt_float(value):
    return "NaN" if value is None else f"{value:.12f}"


def normalize_task(value):
    s = str(value).strip().lower()

    if not s:
        return None

    compact = s.replace("-", "").replace("_", "").replace(" ", "")

    aliases = {
        "cycle2": "cycle2",
        "cycle3": "cycle3",
        "cycle4": "cycle4",
        "cycle5": "cycle5",
        "cycle6": "cycle6",
        "2": "cycle2",
        "3": "cycle3",
        "4": "cycle4",
        "5": "cycle5",
        "6": "cycle6",
        "task0": "cycle2",
        "task1": "cycle3",
        "task2": "cycle4",
        "task3": "cycle5",
        "task4": "cycle6",
        "0": "cycle2",
        "1": "cycle3",
    }

    return aliases.get(compact)


def is_protocol_test_row(row, protocol):
    phase = str(row.get("phase", "")).strip()
    return (
        phase.startswith(PROTOCOL_PREFIX[protocol])
        and "test" in phase
        and not phase.endswith("_mean")
    )


def scalar_rows(rows, protocol):
    out = []

    for row in rows:
        if not is_protocol_test_row(row, protocol):
            continue

        task = str(row.get("task", "")).strip().lower()

        if task in ("", "all"):
            out.append(row)

    return out


def task_rows(rows, protocol):
    out = []

    for row in rows:
        if not is_protocol_test_row(row, protocol):
            continue

        raw_task = str(row.get("task", "")).strip()

        if not raw_task or raw_task.lower() == "all":
            continue

        out.append(row)

    return out


def choose_identity_column(rows):
    if len(rows) == 1:
        for col in IDENTITY_CANDIDATES:
            value = str(rows[0].get(col, "")).strip()
            if value:
                return col
        return "__single__"

    for col in IDENTITY_CANDIDATES:
        vals = [str(row.get(col, "")).strip() for row in rows]

        if all(vals) and len(set(vals)) == len(vals):
            return col

    return None


def local_identity(row, identity_col):
    if identity_col == "__single__":
        return "single"

    return str(row.get(identity_col, "")).strip()


def selected_round(row):
    for col in ("selected_round", "round"):
        value = str(row.get(col, "")).strip()
        if value:
            return value
    return ""


def selected_epoch(row):
    for col in (
        "selected_epoch",
        "epoch",
        "local_epoch",
    ):
        value = str(row.get(col, "")).strip()
        if value:
            return value
    return ""


def expected_clients(setup, method):
    if method == "centralized":
        return 1

    if setup == "five_client":
        return 5

    if setup == "twenty_client":
        return 20

    raise ValueError(setup)


def compare_macro(stored, recomputed):
    if stored is None and recomputed is None:
        return 0.0

    if stored is None or recomputed is None:
        return math.inf

    return abs(stored - recomputed)


# ------------------------------------------------------------------
# Safety / authoritative inputs
# ------------------------------------------------------------------

print("=" * 72)
print("STEP 2 - CANONICAL EXTRACTED RESULTS TABLE")
print("=" * 72)

for required in (FREEZE, FREEZE_MD, CANON_SHA):
    if not required.is_file():
        raise SystemExit(f"ABORT: missing authoritative input: {required}")

for output in (MASTER_OUT, CLIENT_OUT, AUDIT_OUT):
    if output.exists():
        raise SystemExit(
            f"ABORT: output already exists and will not be overwritten: " f"{output}"
        )

freeze_rows = read_csv(FREEZE)

if len(freeze_rows) != 240:
    raise SystemExit(
        f"ABORT: frozen manifest contains {len(freeze_rows)} rows, " "expected 240"
    )

freeze_keys = {
    (
        row["setup"],
        row["method"],
        row["q"],
        int(row["seed"]),
    )
    for row in freeze_rows
}

if len(freeze_keys) != 240:
    raise SystemExit("ABORT: frozen manifest does not contain 240 unique logical keys")

print("Frozen manifest rows : 240 / 240")
print("Primary protocol     : VO (already locked)")
print("Extraction target    : 240 x 3 = 720 protocol rows")
print()


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------

for frozen in freeze_rows:
    setup = frozen["setup"]
    method = frozen["method"]
    q = f"{float(frozen['q']):.2f}"
    seed = int(frozen["seed"])

    if method not in METHOD_ORDER:
        errors.append(
            f"{setup}/{method}/q={q}/seed={seed}: "
            "method outside frozen 11-method registry"
        )
        continue

    try:
        raw_paths = json.loads(frozen["result_paths_json"])
    except Exception as exc:
        errors.append(
            f"{setup}/{method}/q={q}/seed={seed}: "
            f"cannot decode result_paths_json: {exc}"
        )
        continue

    if (
        not isinstance(raw_paths, list)
        or not raw_paths
        or not all(isinstance(p, str) and p.strip() for p in raw_paths)
    ):
        errors.append(
            f"{setup}/{method}/q={q}/seed={seed}: " "invalid result_paths_json"
        )
        continue

    expected_physical = int(frozen["physical_result_files"])

    if len(raw_paths) != expected_physical:
        errors.append(
            f"{setup}/{method}/q={q}/seed={seed}: "
            f"decoded {len(raw_paths)} physical paths, "
            f"expected {expected_physical}"
        )
        continue

    paths = [SUITE / p for p in raw_paths]

    for path in paths:
        if not path.is_file():
            errors.append(
                f"{setup}/{method}/q={q}/seed={seed}: " f"missing result file {path}"
            )

    if any(not p.is_file() for p in paths):
        continue

    physical = [(path, read_csv(path)) for path in paths]

    # Revalidate the metric contract.
    definitions = set()
    selectors = set()

    for path, rows in physical:
        for row in rows:
            d = str(row.get("pr_auc_definition", "")).strip()
            if d:
                definitions.add(d)

            selector = (
                str(row.get("selector_metric", "")).strip()
                or str(row.get("selection_metric", "")).strip()
            )
            if selector:
                selectors.add(selector)

    if definitions != {"trapezoidal_precision_recall_curve_auc"}:
        errors.append(
            f"{setup}/{method}/q={q}/seed={seed}: "
            f"invalid PR-AUC definition(s): {definitions}"
        )

    if selectors != {"macro_pr_auc"}:
        errors.append(
            f"{setup}/{method}/q={q}/seed={seed}: " f"invalid selector(s): {selectors}"
        )

    for protocol in PROTOCOLS:
        clients = {}
        scalar_phase_names = set()
        task_phase_names = set()

        for path, rows in physical:
            scalars = scalar_rows(rows, protocol)
            tasks = task_rows(rows, protocol)

            if not scalars:
                errors.append(f"{path}: no scalar {protocol} final-test rows")
                continue

            identity_col = choose_identity_column(scalars)

            if identity_col is None:
                errors.append(
                    f"{path}: cannot identify clients uniquely for "
                    f"{protocol}; scalar rows={len(scalars)}"
                )
                continue

            scalar_map = {}

            for row in scalars:
                local_id = local_identity(row, identity_col)

                if not local_id:
                    errors.append(f"{path}: empty client identity for {protocol}")
                    continue

                client_key = f"{path.name}::{local_id}"

                if client_key in scalar_map:
                    errors.append(
                        f"{path}: duplicate scalar client "
                        f"{client_key} for {protocol}"
                    )
                    continue

                scalar_map[client_key] = row

                scalar_phase_names.add(str(row.get("phase", "")).strip())

            task_map = {client_key: {} for client_key in scalar_map}

            for row in tasks:
                local_id = local_identity(row, identity_col)

                if not local_id:
                    errors.append(
                        f"{path}: task row has empty client identity " f"for {protocol}"
                    )
                    continue

                client_key = f"{path.name}::{local_id}"

                if client_key not in task_map:
                    errors.append(
                        f"{path}: task row has no matching scalar "
                        f"client for {protocol}: {client_key}"
                    )
                    continue

                task = normalize_task(row.get("task", ""))

                if task is None:
                    errors.append(
                        f"{path}: unknown task label "
                        f"{row.get('task', '')!r} for {protocol}"
                    )
                    continue

                if task in task_map[client_key]:
                    errors.append(
                        f"{path}: duplicate {protocol} task "
                        f"{task} for client {client_key}"
                    )
                    continue

                task_map[client_key][task] = row

                task_phase_names.add(str(row.get("phase", "")).strip())

            for client_key, scalar in scalar_map.items():
                task_dict = task_map.get(client_key, {})

                missing_tasks = [task for task in TASKS if task not in task_dict]

                if missing_tasks:
                    errors.append(
                        f"{path}: {protocol} client {client_key} "
                        f"missing task rows {missing_tasks}"
                    )
                    continue

                task_values = {}

                for task in TASKS:
                    task_values[task] = finite_float(task_dict[task].get("pr_auc", ""))

                recomputed_macro = mean_defined(task_values.values())

                stored_macro = finite_float(scalar.get("macro_pr_auc", ""))

                abs_diff = compare_macro(stored_macro, recomputed_macro)

                if abs_diff > MACRO_TOL:
                    errors.append(
                        f"{path}: {protocol} client {client_key}: "
                        f"stored macro={stored_macro}, "
                        f"recomputed={recomputed_macro}, "
                        f"abs_diff={abs_diff}"
                    )

                if client_key in clients:
                    errors.append(
                        f"{setup}/{method}/q={q}/seed={seed}/"
                        f"{protocol}: duplicate client key "
                        f"{client_key}"
                    )
                    continue

                clients[client_key] = {
                    "source_path": str(path.relative_to(SUITE)),
                    "stored_macro": stored_macro,
                    "recomputed_macro": recomputed_macro,
                    "tasks": task_values,
                    "selected_round": selected_round(scalar),
                    "selected_epoch": selected_epoch(scalar),
                    "scalar_phase": str(scalar.get("phase", "")).strip(),
                    "identity_column": identity_col,
                }

        want_clients = expected_clients(setup, method)

        if len(clients) != want_clients:
            errors.append(
                f"{setup}/{method}/q={q}/seed={seed}/"
                f"{protocol}: extracted {len(clients)} clients, "
                f"expected {want_clients}"
            )
            continue

        client_macros = [info["recomputed_macro"] for info in clients.values()]

        stored_client_macros = [info["stored_macro"] for info in clients.values()]

        system_macro = mean_defined(client_macros)

        stored_system_macro = mean_defined(stored_client_macros)

        system_macro_diff = compare_macro(stored_system_macro, system_macro)

        if system_macro_diff > MACRO_TOL:
            errors.append(
                f"{setup}/{method}/q={q}/seed={seed}/"
                f"{protocol}: stored-vs-recomputed system "
                f"macro mismatch {system_macro_diff}"
            )

        system_tasks = {}

        for task in TASKS:
            system_tasks[task] = mean_defined(
                info["tasks"][task] for info in clients.values()
            )

        selected_rounds = sorted(
            {
                info["selected_round"]
                for info in clients.values()
                if info["selected_round"]
            }
        )

        selected_epochs = sorted(
            {
                info["selected_epoch"]
                for info in clients.values()
                if info["selected_epoch"]
            }
        )

        selection_scope = "per_client" if method == "fully_local" else "runner_recorded"

        selection_provenance = []

        for client_key in sorted(clients):
            info = clients[client_key]

            selection_provenance.append(
                {
                    "client": client_key,
                    "selected_round": info["selected_round"],
                    "selected_epoch": info["selected_epoch"],
                }
            )

            client_rows.append(
                {
                    "setup": setup,
                    "method": method,
                    "q": q,
                    "seed": seed,
                    "protocol": protocol,
                    "client_identity": client_key,
                    "source_result_path": info["source_path"],
                    "identity_column": info["identity_column"],
                    "scalar_phase": info["scalar_phase"],
                    "stored_macro_pr_auc": fmt_float(info["stored_macro"]),
                    "recomputed_macro_pr_auc": fmt_float(info["recomputed_macro"]),
                    "macro_abs_diff": (
                        "NaN"
                        if (
                            info["stored_macro"] is None
                            or info["recomputed_macro"] is None
                        )
                        else "{:.12e}".format(
                            abs(info["stored_macro"] - info["recomputed_macro"])
                        )
                    ),
                    "defined_task_count": sum(
                        value is not None for value in info["tasks"].values()
                    ),
                    "cycle2_pr_auc": fmt_float(info["tasks"]["cycle2"]),
                    "cycle3_pr_auc": fmt_float(info["tasks"]["cycle3"]),
                    "cycle4_pr_auc": fmt_float(info["tasks"]["cycle4"]),
                    "cycle5_pr_auc": fmt_float(info["tasks"]["cycle5"]),
                    "cycle6_pr_auc": fmt_float(info["tasks"]["cycle6"]),
                    "selected_round": info["selected_round"],
                    "selected_epoch": info["selected_epoch"],
                }
            )

        master_rows.append(
            {
                "setup": setup,
                "method": method,
                "q": q,
                "seed": seed,
                "protocol": protocol,
                "macro_pr_auc": fmt_float(system_macro),
                "cycle2_pr_auc": fmt_float(system_tasks["cycle2"]),
                "cycle3_pr_auc": fmt_float(system_tasks["cycle3"]),
                "cycle4_pr_auc": fmt_float(system_tasks["cycle4"]),
                "cycle5_pr_auc": fmt_float(system_tasks["cycle5"]),
                "cycle6_pr_auc": fmt_float(system_tasks["cycle6"]),
                "n_clients": len(clients),
                "n_clients_macro_defined": sum(x is not None for x in client_macros),
                "cycle2_n_defined": sum(
                    info["tasks"]["cycle2"] is not None for info in clients.values()
                ),
                "cycle3_n_defined": sum(
                    info["tasks"]["cycle3"] is not None for info in clients.values()
                ),
                "cycle4_n_defined": sum(
                    info["tasks"]["cycle4"] is not None for info in clients.values()
                ),
                "cycle5_n_defined": sum(
                    info["tasks"]["cycle5"] is not None for info in clients.values()
                ),
                "cycle6_n_defined": sum(
                    info["tasks"]["cycle6"] is not None for info in clients.values()
                ),
                "selection_scope": selection_scope,
                "selected_round_values": "|".join(selected_rounds),
                "selected_epoch_values": "|".join(selected_epochs),
                "selection_provenance_json": json.dumps(
                    selection_provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "scalar_phase_names": "|".join(sorted(scalar_phase_names)),
                "task_phase_names": "|".join(sorted(task_phase_names)),
                "source_result_paths": "|".join(raw_paths),
                "pr_auc_definition": "trapezoidal_precision_recall_curve_auc",
                "selector_metric": "macro_pr_auc",
                "paper_q_treatment": frozen["paper_q_treatment"],
                "centralized_q_pooling_applied": 0,
            }
        )


# ------------------------------------------------------------------
# Final acceptance audit
# ------------------------------------------------------------------

expected_master_keys = {
    (
        setup,
        method,
        q,
        seed,
        protocol,
    )
    for setup in ("five_client", "twenty_client")
    for method in METHOD_ORDER
    for q in Q_VALUES
    for seed in SEEDS
    for protocol in PROTOCOLS
}

actual_master_keys = {
    (
        row["setup"],
        row["method"],
        row["q"],
        int(row["seed"]),
        row["protocol"],
    )
    for row in master_rows
}

if len(master_rows) != 720:
    errors.append(f"master row count={len(master_rows)}, " "expected 720")

if len(actual_master_keys) != len(master_rows):
    errors.append("duplicate master logical keys detected")

if actual_master_keys != expected_master_keys:
    missing = expected_master_keys - actual_master_keys
    extra = actual_master_keys - expected_master_keys

    if missing:
        errors.append(f"missing master keys: " f"{sorted(missing)[:20]}")

    if extra:
        errors.append(f"extra master keys: " f"{sorted(extra)[:20]}")

expected_client_rows = (
    # 5C: 9 non-central methods * 5 clients + 1 central
    (9 * 5 + 1) * 4 * 3 * 3
    +
    # 20C: 9 non-central methods * 20 clients + 1 central
    (9 * 20 + 1) * 4 * 3 * 3
)

if len(client_rows) != expected_client_rows:
    errors.append(
        f"client audit rows={len(client_rows)}, " f"expected {expected_client_rows}"
    )

client_keys = {
    (
        row["setup"],
        row["method"],
        row["q"],
        int(row["seed"]),
        row["protocol"],
        row["client_identity"],
    )
    for row in client_rows
}

if len(client_keys) != len(client_rows):
    errors.append("duplicate client-level audit keys detected")

# Search for any accidental non-true metric definition.
bad_defs = {
    row["pr_auc_definition"]
    for row in master_rows
    if row["pr_auc_definition"] != "trapezoidal_precision_recall_curve_auc"
}

if bad_defs:
    errors.append(f"unexpected PR-AUC definitions: {bad_defs}")

# Client-level max numerical discrepancy.
macro_diffs = []

for row in client_rows:
    value = str(row["macro_abs_diff"]).strip()

    if value and value != "NaN":
        macro_diffs.append(float(value))

max_macro_diff = max(macro_diffs) if macro_diffs else 0.0

if max_macro_diff > MACRO_TOL:
    errors.append(
        f"max client macro discrepancy "
        f"{max_macro_diff} exceeds tolerance "
        f"{MACRO_TOL}"
    )

# Count genuinely undefined task values. These remain NaN.
undefined_task_cells = 0

for row in master_rows:
    for task in TASKS:
        if row[f"{task}_pr_auc"] == "NaN":
            undefined_task_cells += 1


# ------------------------------------------------------------------
# Fail before writing any canonical extraction outputs
# ------------------------------------------------------------------

if errors:
    print("=" * 72)
    print(f"FAIL - STEP 2 EXTRACTION NOT ACCEPTED " f"({len(errors)} issue(s))")
    print("=" * 72)

    for error in errors:
        print(" -", error)

    print()
    print("No canonical extracted-result CSV was written.")
    print("Raw frozen suite was not modified.")
    raise SystemExit(1)


# ------------------------------------------------------------------
# Sort and write accepted outputs
# ------------------------------------------------------------------

method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}

protocol_rank = {protocol: i for i, protocol in enumerate(PROTOCOLS)}

master_rows.sort(
    key=lambda r: (
        r["setup"],
        method_rank[r["method"]],
        float(r["q"]),
        int(r["seed"]),
        protocol_rank[r["protocol"]],
    )
)

client_rows.sort(
    key=lambda r: (
        r["setup"],
        method_rank[r["method"]],
        float(r["q"]),
        int(r["seed"]),
        protocol_rank[r["protocol"]],
        r["client_identity"],
    )
)

master_fields = [
    "setup",
    "method",
    "q",
    "seed",
    "protocol",
    "macro_pr_auc",
    "cycle2_pr_auc",
    "cycle3_pr_auc",
    "cycle4_pr_auc",
    "cycle5_pr_auc",
    "cycle6_pr_auc",
    "n_clients",
    "n_clients_macro_defined",
    "cycle2_n_defined",
    "cycle3_n_defined",
    "cycle4_n_defined",
    "cycle5_n_defined",
    "cycle6_n_defined",
    "selection_scope",
    "selected_round_values",
    "selected_epoch_values",
    "selection_provenance_json",
    "scalar_phase_names",
    "task_phase_names",
    "source_result_paths",
    "pr_auc_definition",
    "selector_metric",
    "paper_q_treatment",
    "centralized_q_pooling_applied",
]

client_fields = [
    "setup",
    "method",
    "q",
    "seed",
    "protocol",
    "client_identity",
    "source_result_path",
    "identity_column",
    "scalar_phase",
    "stored_macro_pr_auc",
    "recomputed_macro_pr_auc",
    "macro_abs_diff",
    "defined_task_count",
    "cycle2_pr_auc",
    "cycle3_pr_auc",
    "cycle4_pr_auc",
    "cycle5_pr_auc",
    "cycle6_pr_auc",
    "selected_round",
    "selected_epoch",
]

with MASTER_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=master_fields)
    writer.writeheader()
    writer.writerows(master_rows)

with CLIENT_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=client_fields)
    writer.writeheader()
    writer.writerows(client_rows)

master_sha = sha256(MASTER_OUT)
client_sha = sha256(CLIENT_OUT)
freeze_sha = sha256(FREEZE)
canonical_sha = sha256(CANON_SHA)

audit_text = f"""STEP 2 - CANONICAL EXTRACTED RESULTS AUDIT
============================================================

STATUS: PASS

Authoritative input:
  {FREEZE}
  SHA-256: {freeze_sha}

Canonical artifact checksum manifest:
  {CANON_SHA}
  SHA-256: {canonical_sha}

Main registry:
  10 methods/references

Frozen logical q-indexed entries:
  240

Protocols:
  VV, VO, OO

Canonical extracted rows:
  720 / 720

Unique canonical logical keys:
  720 / 720

Client-level audit rows:
  {len(client_rows)} / {expected_client_rows}

Metric:
  macro_pr_auc

Definition:
  trapezoidal_precision_recall_curve_auc

Client macro rule:
  arithmetic mean of defined task-level PR-AUC values

Undefined task PR-AUC:
  preserved as NaN
  never replaced by zero

Maximum stored-vs-recomputed client macro difference:
  {max_macro_diff:.12e}

Tolerance:
  {MACRO_TOL:.12e}

Fully Local:
  per-client selected states preserved
  no artificial single selected round created

Centralized:
  raw q-tagged entries preserved in this canonical table
  q pooling NOT applied in Step 2
  pooling is deferred to Step 3 paper-summary analysis

Primary Figure 3 protocol:
  VO

Output:
  {MASTER_OUT}
  SHA-256: {master_sha}

Client macro audit:
  {CLIENT_OUT}
  SHA-256: {client_sha}

Undefined task cells in 720-row system table:
  {undefined_task_cells}

NEXT STEP:
  Step 3 - audit the canonical extracted table and produce
  mean +/- SAMPLE SD summaries over seeds.

  Centralized must use the frozen special rule:
    pool q within each seed,
    then mean/sample-SD across seeds,
    then repeat the same summary at every q.

  Do not plot Figure 3 until the Step-3 VO numerical
  summary has been inspected.
"""

AUDIT_OUT.write_text(audit_text, encoding="utf-8")

print("=" * 72)
print("PASS - STEP 2 CANONICAL EXTRACTION COMPLETE")
print("=" * 72)
print()
print("Canonical protocol rows : 720 / 720")
print("Unique logical keys     : " f"{len(actual_master_keys)} / 720")
print("Client audit rows       : " f"{len(client_rows)} / {expected_client_rows}")
print()
print("Max client macro diff   : " f"{max_macro_diff:.12e}")
print("Undefined task cells    : " f"{undefined_task_cells}")
print("NaN zero-filling        : NEVER")
print()
print("Fully Local             : per-client selection preserved")
print("Centralized             : raw q-tagged entries preserved")
print("Centralized q pooling   : NOT applied yet")
print()
print("Created:")
print(" ", MASTER_OUT)
print(" ", CLIENT_OUT)
print(" ", AUDIT_OUT)
print()
print("Raw frozen suite was NOT modified.")
print()
print("=" * 72)
print("NEXT STEP")
print("=" * 72)
print("Step 3: produce audited mean +/- sample SD summaries")
print("from main_protocol_results.csv.")
print("Inspect VO numbers BEFORE making Figure 3.")
print("=" * 72)
