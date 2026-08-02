from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

# ============================================================
# Merge experiment logs for plotting.
#
# Includes:
#   - simple baselines
#   - FL baselines
#   - FedALA variants
#   - APPLE variants
#   - APPLE-ALA / APPLE-PostALA
#   - TaskExpert-APPLE / TaskExpert-APPLE-ExpertProx
# ============================================================
SOURCE_DIRS = [
    Path(
        "./andrea/fully_local_multihead_clustering_experiment_multi_select_multiselect"
    ),
    Path(
        "./andrea/local_centralized_multihead_clustering_experiment_multi_select_multiselect_next"
    ),
    Path("./andrea/fedavg_multihead_clustering_experiment_multi_select_multiselect"),
    Path(
        "./andrea/fedprox_multihead_clustering_experiment_multi_select_multiselect_next"
    ),
    Path(
        "./andrea/gcflplus_multihead_clustering_experiment_multi_select_multiselect_gcfl"
    ),
    Path(
        "./andrea/fedala_fedavg_multihead_clustering_experiment_multi_select_multiselect_fedala"
    ),
    Path(
        "./andrea/apple_backbone_taskhead_clustering_experiment_multi_select_multiselect_apple"
    ),
    Path(
        "./andrea/apple_ala_fedavg_backbone_taskhead_clustering_experiment_multi_select_multiselect_newmethods_0710"
    ),
    Path(
        "./andrea/apple_post_ala_fedavg_backbone_taskhead_clustering_experiment_multi_select_postala_0720"
    ),
]

MERGED_LOG_CSV = Path("./andrea/q_multihead_final_manifest.csv")

EXCLUDED_RUN_TYPES = set()
EXCLUDED_OUT_CSV_PREFIXES = tuple()


def find_experiment_logs(source_dirs: list[Path]) -> list[Path]:
    """
    Prefer seed-specific logs:
        experiment_log_0.csv
        experiment_log_1.csv
        experiment_log_2.csv

    Only fall back to experiment_log.csv if no seed-specific logs exist.

    This avoids accidentally using old/empty merged logs.
    """
    logs: list[Path] = []

    for root in source_dirs:
        if not root.exists():
            print(f"[skip] missing directory: {root}")
            continue

        seed_logs = sorted(root.glob("experiment_log_[0-9]*.csv"))

        if seed_logs:
            logs.extend(seed_logs)
            continue

        merged_log = root / "experiment_log.csv"
        if merged_log.exists():
            logs.append(merged_log)

    logs = sorted(set(p.resolve() for p in logs))
    return [Path(p) for p in logs]


def read_log_safely(path: Path) -> pd.DataFrame | None:
    """
    Read one experiment log.

    Returns None for empty or unreadable logs, instead of crashing the merge.
    """
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        print(f"[skip] empty / no columns: {path}")
        return None
    except Exception as e:
        print(f"[skip] read error: {path} | {type(e).__name__}: {e}")
        return None

    if df.empty:
        print(f"[skip] empty dataframe: {path}")
        return None

    return df


def normalize_known_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize method names that may still be ambiguous in older logs.

    This mainly separates APPLE variants and FedALA variants using output filename
    and/or mode columns.
    """
    df = df.copy()

    if "out_csv" not in df.columns:
        return df

    out_csv = df["out_csv"].astype(str)

    # ------------------------------------------------------------
    # APPLE variants
    # ------------------------------------------------------------
    is_apple_fedavg_backbone_qinit_taskhead = out_csv.str.contains(
        "apple_fedavg_backbone_qinit_taskhead", na=False
    )
    is_apple_fedavg_backbone_oracleq_taskhead = out_csv.str.contains(
        "apple_fedavg_backbone_oracleq_taskhead", na=False
    )
    is_apple_fedavg_backbone_taskhead = out_csv.str.contains(
        "apple_fedavg_backbone_taskhead", na=False
    )
    is_apple_backbone_taskhead = out_csv.str.contains(
        "apple_backbone_taskhead", na=False
    )
    is_apple_taskhead = out_csv.str.contains("apple_taskhead", na=False)

    if "apple_mixing_mode" in df.columns:
        mix = df["apple_mixing_mode"].astype(str)

        is_apple_fedavg_backbone_qinit_taskhead = (
            is_apple_fedavg_backbone_qinit_taskhead
            | mix.eq("fedavg_backbone_qinit_task_head")
        )

        is_apple_fedavg_backbone_oracleq_taskhead = (
            is_apple_fedavg_backbone_oracleq_taskhead
            | mix.eq("fedavg_backbone_oracleq_task_head")
        )

        is_apple_fedavg_backbone_taskhead = is_apple_fedavg_backbone_taskhead | mix.eq(
            "fedavg_backbone_task_head"
        )

        is_apple_backbone_taskhead = is_apple_backbone_taskhead | mix.eq(
            "backbone_task_head"
        )

        is_apple_taskhead = is_apple_taskhead | mix.eq("task_head")

    # Avoid overlap. Most specific methods win first.
    is_apple_fedavg_backbone_taskhead = (
        is_apple_fedavg_backbone_taskhead
        & ~is_apple_fedavg_backbone_oracleq_taskhead
        & ~is_apple_fedavg_backbone_qinit_taskhead
    )

    is_apple_backbone_taskhead = (
        is_apple_backbone_taskhead
        & ~is_apple_fedavg_backbone_taskhead
        & ~is_apple_fedavg_backbone_oracleq_taskhead
        & ~is_apple_fedavg_backbone_qinit_taskhead
    )

    is_apple_taskhead = (
        is_apple_taskhead
        & ~is_apple_backbone_taskhead
        & ~is_apple_fedavg_backbone_taskhead
        & ~is_apple_fedavg_backbone_oracleq_taskhead
        & ~is_apple_fedavg_backbone_qinit_taskhead
    )

    df.loc[is_apple_taskhead, "run_type"] = "apple_taskhead"
    df.loc[is_apple_taskhead, "algorithm"] = "apple_taskhead"
    df.loc[is_apple_taskhead, "apple_variant"] = "task_head"

    df.loc[is_apple_backbone_taskhead, "run_type"] = "apple_backbone_taskhead"
    df.loc[is_apple_backbone_taskhead, "algorithm"] = "apple_backbone_taskhead"
    df.loc[is_apple_backbone_taskhead, "apple_variant"] = "backbone_task_head"

    df.loc[
        is_apple_fedavg_backbone_taskhead,
        "run_type",
    ] = "apple_fedavg_backbone_taskhead"
    df.loc[
        is_apple_fedavg_backbone_taskhead,
        "algorithm",
    ] = "apple_fedavg_backbone_taskhead"
    df.loc[
        is_apple_fedavg_backbone_taskhead,
        "apple_variant",
    ] = "fedavg_backbone_task_head"

    df.loc[
        is_apple_fedavg_backbone_oracleq_taskhead,
        "run_type",
    ] = "apple_fedavg_backbone_oracleq_taskhead"
    df.loc[
        is_apple_fedavg_backbone_oracleq_taskhead,
        "algorithm",
    ] = "apple_fedavg_backbone_oracleq_taskhead"
    df.loc[
        is_apple_fedavg_backbone_oracleq_taskhead,
        "apple_variant",
    ] = "fedavg_backbone_oracleq_task_head"

    df.loc[
        is_apple_fedavg_backbone_qinit_taskhead,
        "run_type",
    ] = "apple_fedavg_backbone_qinit_taskhead"
    df.loc[
        is_apple_fedavg_backbone_qinit_taskhead,
        "algorithm",
    ] = "apple_fedavg_backbone_qinit_taskhead"
    df.loc[
        is_apple_fedavg_backbone_qinit_taskhead,
        "apple_variant",
    ] = "fedavg_backbone_qinit_task_head"

    # ------------------------------------------------------------
    # FedALA variants
    # ------------------------------------------------------------
    is_fedala_fedavg = out_csv.str.contains("fedala_all", na=False)
    is_fedala_head_only = out_csv.str.contains("fedala_head_only", na=False)

    if "fedala_mode" in df.columns:
        fedala_mode = df["fedala_mode"].astype(str)
        is_fedala_fedavg = is_fedala_fedavg | fedala_mode.eq("all")
        is_fedala_head_only = is_fedala_head_only | fedala_mode.eq("head_only")

    # Avoid overlap.
    is_fedala_fedavg = is_fedala_fedavg & ~is_fedala_head_only

    df.loc[is_fedala_fedavg, "run_type"] = "fedala_fedavg"
    df.loc[is_fedala_fedavg, "algorithm"] = "fedala_fedavg"
    df.loc[is_fedala_fedavg, "fedala_variant"] = "all_model"

    df.loc[is_fedala_head_only, "run_type"] = "fedala_head_only"
    df.loc[is_fedala_head_only, "algorithm"] = "fedala_head_only"
    df.loc[is_fedala_head_only, "fedala_variant"] = "head_only"

    # ------------------------------------------------------------
    # New experimental methods
    # ------------------------------------------------------------
    # These methods also use apple_mixing_mode="fedavg_backbone_task_head",
    # so they must be restored after the generic APPLE normalization.

    # Original new methods from 0710
    is_taskexpert_apple = out_csv.str.contains(
        "taskexpert_apple_multi_select", na=False
    )

    is_apple_ala = (
        out_csv.str.contains("apple_ala_multi_select", na=False) & ~is_taskexpert_apple
    )

    df.loc[is_apple_ala, "run_type"] = "apple_ala_multi_select"
    df.loc[is_apple_ala, "algorithm"] = "apple_ala_multi_select"
    df.loc[is_apple_ala, "apple_variant"] = "apple_ala"

    df.loc[is_taskexpert_apple, "run_type"] = "taskexpert_apple_multi_select"
    df.loc[is_taskexpert_apple, "algorithm"] = "taskexpert_apple_multi_select"
    df.loc[is_taskexpert_apple, "apple_variant"] = "taskexpert_apple"

    # ------------------------------------------------------------
    # TaskExpert-APPLE-ExpertProx method from 0715
    # ------------------------------------------------------------
    is_taskexpert_apple_expertprox = out_csv.str.contains(
        "taskexpert_apple_expertprox_multi_select", na=False
    ) | out_csv.str.contains("te_apple_xprox_", na=False)

    df.loc[
        is_taskexpert_apple_expertprox,
        "run_type",
    ] = "taskexpert_apple_expertprox_multi_select"
    df.loc[
        is_taskexpert_apple_expertprox,
        "algorithm",
    ] = "taskexpert_apple_expertprox_multi_select"
    df.loc[
        is_taskexpert_apple_expertprox,
        "apple_variant",
    ] = "taskexpert_apple_expertprox"

    # ------------------------------------------------------------
    # APPLE-PostALA method from 0720
    # ------------------------------------------------------------
    # This method still reports apple_mixing_mode="fedavg_backbone_task_head",
    # so the generic APPLE normalization above would otherwise relabel it as
    # apple_fedavg_backbone_taskhead. Restore its true method identity here.
    # Detect both the explicit long stem and a possible shortened postala stem.
    is_apple_post_ala = out_csv.str.contains(
        "apple_post_ala_multi_select", na=False
    ) | out_csv.str.contains("post_ala", na=False)

    df.loc[is_apple_post_ala, "run_type"] = "apple_post_ala_multi_select"
    df.loc[is_apple_post_ala, "algorithm"] = "apple_post_ala_multi_select"
    df.loc[is_apple_post_ala, "apple_variant"] = "apple_post_ala"
    df.loc[is_apple_post_ala, "fedala_variant"] = "post_ala_filter"

    return df


def remove_temporarily_excluded_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety filter.

    Even if a fedala_fedavg log accidentally appears, remove it for now.
    """
    df = df.copy()

    if "run_type" in df.columns:
        df = df[~df["run_type"].astype(str).isin(EXCLUDED_RUN_TYPES)]

    if "algorithm" in df.columns:
        df = df[~df["algorithm"].astype(str).isin(EXCLUDED_RUN_TYPES)]

    if "out_csv" in df.columns:
        out_name = df["out_csv"].astype(str).map(lambda s: Path(s).name)
        for prefix in EXCLUDED_OUT_CSV_PREFIXES:
            df = df[~out_name.str.startswith(prefix)]

    return df


def merge_logs(log_paths: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    total_input_rows = 0
    total_kept_rows = 0

    print("Found experiment logs:")
    for path in log_paths:
        print(f"  {path}")

    print()

    for log_path in log_paths:
        df = read_log_safely(log_path)
        if df is None:
            continue

        before = len(df)

        df = normalize_known_variants(df)
        df = remove_temporarily_excluded_rows(df)

        after = len(df)

        total_input_rows += before
        total_kept_rows += after

        if after == 0:
            print(f"[skip after filter] {log_path} | input_rows={before}")
            continue

        print(f"[merge] {log_path} | input_rows={before} kept_rows={after}")
        pieces.append(df)

    if not pieces:
        raise RuntimeError("No usable experiment-log rows remained after filtering.")

    merged = pd.concat(pieces, axis=0, ignore_index=True, sort=False)

    # out_csv uniquely identifies one experiment run.
    # If the same run appears in multiple logs, keep the latest occurrence.
    if "out_csv" in merged.columns:
        before_dedup = len(merged)
        merged = merged.drop_duplicates(subset=["out_csv"], keep="last")
        print()
        print(f"deduplicate by out_csv: {before_dedup} -> {len(merged)}")
    else:
        before_dedup = len(merged)
        merged = merged.drop_duplicates(keep="last")
        print()
        print(f"deduplicate full rows: {before_dedup} -> {len(merged)}")

    print()
    print(f"total input rows before filter: {total_input_rows}")
    print(f"total rows after exclude filter: {total_kept_rows}")

    return merged


PROJECT_ROOT = Path(".").resolve()
REMOTE_PROJECT_ROOTS = [
    "/home/nfs/ali7/Subgraph_Federated_Learning",
]


def resolve_local_path(value) -> Path | None:
    """
    Convert an out_csv / dr_csv_path from an experiment log into a local path.

    Handles:
      - relative paths like andrea/runs_multiselect/xxx.csv
      - ./andrea/runs_multiselect/xxx.csv
      - remote absolute paths like /home/nfs/ali7/Subgraph_Federated_Learning/andrea/...
      - local absolute paths
    """
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    # Remove leading ./ for consistency.
    if s.startswith("./"):
        s = s[2:]

    # Map remote absolute project paths to local project paths.
    for remote_root in REMOTE_PROJECT_ROOTS:
        prefix = remote_root.rstrip("/") + "/"
        if s.startswith(prefix):
            rel = s[len(prefix) :]
            return PROJECT_ROOT / rel

    p = Path(s)

    # Local absolute path.
    if p.is_absolute():
        return p

    # Relative path from project root.
    return PROJECT_ROOT / p


def add_local_file_existence_flags(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add local path/existence columns for plotting/debugging.
    The actual run directory is inferred from out_csv, not from run script defaults.
    """
    merged_df = merged_df.copy()

    if "out_csv" in merged_df.columns:
        local_out_paths = []
        local_out_dirs = []
        local_out_exists = []

        for value in merged_df["out_csv"]:
            p = resolve_local_path(value)
            local_out_paths.append(str(p) if p is not None else "")
            local_out_dirs.append(str(p.parent) if p is not None else "")
            local_out_exists.append(bool(p is not None and p.exists()))

        merged_df["local_out_csv_path"] = local_out_paths
        merged_df["actual_run_dir"] = local_out_dirs
        merged_df["local_out_csv_exists"] = local_out_exists
    else:
        merged_df["local_out_csv_path"] = ""
        merged_df["actual_run_dir"] = ""
        merged_df["local_out_csv_exists"] = False

    if "dr_csv_path" in merged_df.columns:
        local_dr_paths = []
        local_dr_exists = []

        for value in merged_df["dr_csv_path"]:
            p = resolve_local_path(value)
            local_dr_paths.append(str(p) if p is not None else "")
            local_dr_exists.append(bool(p is not None and p.exists()))

        merged_df["local_dr_csv_path"] = local_dr_paths
        merged_df["local_dr_csv_exists"] = local_dr_exists

    if "ala_csv_path" in merged_df.columns:
        local_ala_paths = []
        local_ala_exists = []

        for value in merged_df["ala_csv_path"]:
            p = resolve_local_path(value)
            local_ala_paths.append(str(p) if p is not None else "")
            local_ala_exists.append(bool(p is not None and p.exists()))

        merged_df["local_ala_csv_path"] = local_ala_paths
        merged_df["local_ala_csv_exists"] = local_ala_exists

    return merged_df


def main() -> None:
    log_paths = find_experiment_logs(SOURCE_DIRS)

    if not log_paths:
        raise FileNotFoundError("No experiment_log*.csv files found.")

    if MERGED_LOG_CSV.exists():
        print(f"[remove old merged file] {MERGED_LOG_CSV}")
        MERGED_LOG_CSV.unlink()

    merged_df = merge_logs(log_paths)
    merged_df = add_local_file_existence_flags(merged_df)

    MERGED_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(MERGED_LOG_CSV, index=False)

    print()
    print("============================================================")
    print("MERGE DONE")
    print("============================================================")
    print(f"merged rows: {len(merged_df)}")
    print(f"saved to: {MERGED_LOG_CSV}")

    print()
    print("============================================================")
    print("FILE EXISTENCE CHECKS")
    print("============================================================")

    if "local_out_csv_exists" in merged_df.columns:
        print()
        print("local_out_csv_exists:")
        print(merged_df["local_out_csv_exists"].value_counts(dropna=False))

    if "local_dr_csv_exists" in merged_df.columns:
        print()
        print("local_dr_csv_exists:")
        print(merged_df["local_dr_csv_exists"].value_counts(dropna=False))

    if "local_ala_csv_exists" in merged_df.columns:
        nonempty_ala = merged_df["local_ala_csv_path"].astype(str).str.len() > 0
        if nonempty_ala.any():
            print()
            print("local_ala_csv_exists, only rows with ala_csv_path:")
            print(
                merged_df.loc[nonempty_ala, "local_ala_csv_exists"].value_counts(
                    dropna=False
                )
            )

    print()
    print("============================================================")
    print("ACTUAL RUN DIRECTORIES FROM out_csv")
    print("============================================================")

    if "actual_run_dir" in merged_df.columns:
        run_dir_counts = (
            merged_df["actual_run_dir"]
            .astype(str)
            .replace("", "<missing>")
            .value_counts()
            .sort_index()
        )
        print(run_dir_counts.to_string())

    print()
    print("============================================================")
    print("ROWS BY run_type / seed / q_value")
    print("============================================================")

    if "run_type" in merged_df.columns:
        print()
        print("Rows by run_type:")
        print(merged_df["run_type"].value_counts(dropna=False).sort_index())

    if "run_type" in merged_df.columns and "seed" in merged_df.columns:
        print()
        print("Rows by run_type and seed:")
        print(pd.crosstab(merged_df["run_type"], merged_df["seed"], dropna=False))

    if "run_type" in merged_df.columns and "q_value" in merged_df.columns:
        print()
        print("Rows by run_type and q_value:")
        print(pd.crosstab(merged_df["run_type"], merged_df["q_value"], dropna=False))

    missing_out = merged_df[~merged_df["local_out_csv_exists"]]
    if len(missing_out) > 0:
        print()
        print("WARNING: missing local out_csv files:")
        cols = [
            c
            for c in [
                "run_type",
                "algorithm",
                "seed",
                "q_value",
                "subset_clients",
                "out_csv",
                "local_out_csv_path",
            ]
            if c in missing_out.columns
        ]
        print(missing_out[cols].to_string(index=False))

    if "local_dr_csv_exists" in merged_df.columns:
        # Only check rows that actually have a dr_csv_path.
        has_dr = merged_df["local_dr_csv_path"].astype(str).str.len() > 0
        missing_dr = merged_df[has_dr & ~merged_df["local_dr_csv_exists"]]
        if len(missing_dr) > 0:
            print()
            print("WARNING: missing local DR csv files:")
            cols = [
                c
                for c in [
                    "run_type",
                    "algorithm",
                    "seed",
                    "q_value",
                    "subset_clients",
                    "dr_csv_path",
                    "local_dr_csv_path",
                ]
                if c in missing_dr.columns
            ]
            print(missing_dr[cols].to_string(index=False))

    if "local_out_csv_exists" in merged_df.columns:
        print()
        print("Local out_csv availability:")
        print(merged_df["local_out_csv_exists"].value_counts(dropna=False))

        missing = merged_df[~merged_df["local_out_csv_exists"]]
        if len(missing) > 0:
            print()
            print("WARNING: missing local out_csv files:")
            cols = [
                c
                for c in ["run_type", "seed", "q_value", "subset_clients", "out_csv"]
                if c in missing.columns
            ]
            print(missing[cols].to_string(index=False))

    if "run_type" in merged_df.columns:
        print()
        print("Rows by run_type:")
        print(merged_df["run_type"].value_counts(dropna=False).sort_index())

    if "q_value" in merged_df.columns:
        print()
        print("Rows by q_value:")
        print(merged_df["q_value"].value_counts(dropna=False).sort_index())

    if "seed" in merged_df.columns and "run_type" in merged_df.columns:
        print()
        print("Rows by run_type and seed:")
        print(
            pd.crosstab(
                merged_df["run_type"],
                merged_df["seed"],
                dropna=False,
            )
        )


if __name__ == "__main__":
    main()
