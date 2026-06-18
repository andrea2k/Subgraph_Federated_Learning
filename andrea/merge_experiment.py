from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# New multi-head experiment-log folders
# ============================================================

SOURCE_DIRS = [
    # non-communication baselines
    Path("./andrea/fully_local_multihead_clustering_experiment"),
    Path("./andrea/local_centralized_multihead_clustering_experiment"),
    # multi-head FL baselines
    Path("./andrea/fedavg_multihead_clustering_experiment"),
    Path("./andrea/fedprox_multihead_clustering_experiment"),
    Path("./andrea/gcflplus_multihead_clustering_experiment"),
    # APPLE multi-head variants
    Path("./andrea/apple_taskhead_clustering_experiment"),
    Path("./andrea/apple_backbone_taskhead_clustering_experiment"),
    Path("./andrea/apple_fedavg_backbone_taskhead_clustering_experiment"),
    # oracle-q ablation
    Path("./andrea/apple_fedavg_backbone_oracleq_taskhead_clustering_experiment"),
    # q-init learned DR ablation
    Path("./andrea/apple_fedavg_backbone_qinit_taskhead_clustering_experiment"),
    # FedALA variants
    Path("./andrea/fedala_fedavg_multihead_clustering_experiment"),
    Path("./andrea/fedala_head_only_multihead_clustering_experiment"),
]


# New merged output name.
# This should be the file that your plotting notebook reads next.
MERGED_LOG_CSV = Path("./andrea/q_multihead_80.csv")


def find_experiment_logs(source_dirs: list[Path]) -> list[Path]:
    logs: list[Path] = []

    for root in source_dirs:
        if not root.exists():
            print(f"[skip] missing directory: {root}")
            continue

        # Include:
        #   experiment_log.csv
        #   experiment_log_0.csv
        #   experiment_log_1.csv
        #   experiment_log_2.csv
        for path in root.rglob("experiment_log*.csv"):
            logs.append(path)

    logs = sorted(set(p.resolve() for p in logs))
    return [Path(p) for p in logs]


def upsert_experiment_rows_local(csv_path: Path, rows: list[dict]) -> None:
    """
    Local version of upsert_experiment_rows.

    We keep this merge script self-contained so it does not matter whether
    the helper is from helper_funcs or helper_funcs_multihead.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        merged = pd.concat([old_df, new_df], axis=0, ignore_index=True)
    else:
        merged = new_df.copy()

    # out_csv uniquely identifies one experiment run.
    # If a row appears again, keep the latest one.
    if "out_csv" in merged.columns:
        merged = merged.drop_duplicates(subset=["out_csv"], keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")

    merged.to_csv(csv_path, index=False)


def normalize_known_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize method names that may still be ambiguous in older logs.

    This mainly separates APPLE variants using output filename and/or
    apple_mixing_mode.
    """
    df = df.copy()

    if "out_csv" not in df.columns:
        return df

    out_csv = df["out_csv"].astype(str)

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

    # Avoid overlap. The most specific methods win first.
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

    return df


def main() -> None:
    log_paths = find_experiment_logs(SOURCE_DIRS)

    if not log_paths:
        raise FileNotFoundError("No experiment_log*.csv files found.")

    print("Found experiment logs:")
    for path in log_paths:
        print(f"  {path}")

    # Start fresh every time, so old merged rows do not accidentally remain.
    if MERGED_LOG_CSV.exists():
        print()
        print(f"[remove old merged file] {MERGED_LOG_CSV}")
        MERGED_LOG_CSV.unlink()

    total_input_rows = 0

    for log_path in log_paths:
        df = pd.read_csv(log_path)

        if df.empty:
            print(f"[skip] empty log: {log_path}")
            continue

        df = normalize_known_variants(df)

        total_input_rows += len(df)
        rows = df.to_dict(orient="records")

        print(f"[merge] {log_path} -> rows={len(df)}")
        upsert_experiment_rows_local(MERGED_LOG_CSV, rows)

    merged_df = pd.read_csv(MERGED_LOG_CSV)

    print()
    print("============================================================")
    print("MERGE DONE")
    print("============================================================")
    print(f"input rows before upsert: {total_input_rows}")
    print(f"merged rows after upsert: {len(merged_df)}")
    print(f"saved to: {MERGED_LOG_CSV}")

    if "run_type" in merged_df.columns:
        print()
        print("Rows by run_type:")
        print(merged_df["run_type"].value_counts(dropna=False))

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
