from __future__ import annotations

import time
from pathlib import Path

import os
import pandas as pd

from andrea.helper_funcs.fl_run_helper import upsert_experiment_rows

SOURCE_DIRS = [
    Path("./andrea/local_clustering_experiment"),
    Path("./andrea/fedavg_clustering_experiment"),
    Path("./andrea/fedprox_clustering_experiment"),
]

MERGED_LOG_CSV = Path("./andrea/experiment_log.csv")


def find_experiment_logs(source_dirs: list[Path]) -> list[Path]:
    logs: list[Path] = []

    for root in source_dirs:
        if not root.exists():
            print(f"[skip] missing directory: {root}")
            continue

        # include experiment_log.csv and experiment_log_<seed>.csv
        for path in root.rglob("experiment_log*.csv"):
            logs.append(path)

    # stable order
    logs = sorted(set(p.resolve() for p in logs))
    return [Path(p) for p in logs]


def main():

    log_paths = find_experiment_logs(SOURCE_DIRS)

    if not log_paths:
        raise FileNotFoundError("No experiment_log*.csv files found.")

    total_rows = 0

    for log_path in log_paths:

        df = pd.read_csv(log_path)
        if df.empty:
            print(f"[skip] empty log: {log_path}")
            continue

        rows = df.to_dict(orient="records")
        total_rows += len(rows)

        upsert_experiment_rows(MERGED_LOG_CSV, rows)

    merged_df = pd.read_csv(MERGED_LOG_CSV)
    print()
    print(f"total experiments run: {len(merged_df)}")
    print(f"saved to: {MERGED_LOG_CSV}")

    return
    chosen_df = pd.read_csv(SELECTED_SUBSETS_CSV_PATH)

    id_to_client = load_clients(chosen_df, ALL_DATA_LOGS)
    base_cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)

    ONLY_SEED = os.environ.get("ONLY_SEED")
    if ONLY_SEED is not None:
        SEEDS = [int(ONLY_SEED)]
        print("working only with seed:", SEEDS)
        EXPERIMENT_LOG_CSV = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log_{ONLY_SEED}.csv"
        )
    else:
        SEEDS = [0, 1, 2]
        print("working with all seed:", SEEDS)
        EXPERIMENT_LOG_CSV = Path(
            f"./andrea/{EXPERIMENT_LOG_FOLDER}/experiment_log.csv"
        )

    sweep = list(
        iter_sweep_cfgs(
            base_cfg,
            seeds=SEEDS,
            mcw=MCW,
            num_layers=NUM_LAYERS,
            lrs=LRS,
            weight_decays=WEIGHT_DECAYS,
            dropouts=DROPOUTS,
            hidden_dims=HIDDEN_DIMS,
            use_ego_ids=USE_EGO_IDS,
            batch_sizes=BATCH_SIZE,
        )
    )

    total_runs = len(sweep) * len(chosen_df)
    print("Number of runs:", total_runs)
    progress = ProgressPrinter(total_runs)

    for cfg, meta in sweep:
        seed = int(meta["seed"])

        for _, row in chosen_df.iterrows():

            subset_id = parse_subset_clients(row["subset_clients"])

            run_start = time.perf_counter()

            subset_clients = [id_to_client[graph_id] for graph_id in subset_id]

            print()
            print(
                f"Starting with {len(id_to_client)}-client fedavg training",
                row["subset_clients"],
            )

            fed_paths = run_fedavg(
                subset_clients,
                cfg,
                seed,
                RUNS_ROOT,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                device=DEVICE,
                selection_metric=SELECTION_METRIC,
            )

            fed_row = build_fedavg_log_row(
                subset_clients=subset_clients,
                cfg=cfg,
                seed=seed,
                run_paths=fed_paths,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                client_fraction=CLIENT_FRACTION,
                selection_metric=SELECTION_METRIC,
            )
            fed_row.update(
                {
                    "subset_clients": str(row["subset_clients"]),
                }
            )

            upsert_experiment_rows(EXPERIMENT_LOG_CSV, [fed_row])

            progress.step(
                label=f"fedavg done -> {fed_paths.csv_path}",
                run_start_time=run_start,
            )


if __name__ == "__main__":
    main()
