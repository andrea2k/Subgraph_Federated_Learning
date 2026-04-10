from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.train_utils import load_datasets
from utils.hetero import make_bidirected_hetero


@dataclass
class ClientData:
    graph_id: str
    data_dir: str
    dataset_id: str
    train_g: object
    val_g: object
    test_g: object
    train_h: object
    val_h: object
    test_h: object


def load_client_from_dir(client_dir: str):
    train_g, val_g, test_g = load_datasets(
        log_dir=client_dir,
        train_data_file="train.pt",
        val_data_file="val.pt",
        test_data_file="test.pt",
    )
    return train_g, val_g, test_g


def parse_subset_clients(s: str) -> List[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip() != ""]


def collect_needed_client_ids(chosen_df: pd.DataFrame) -> List[int]:
    needed_client_ids = set()
    for s in chosen_df["subset_clients"]:
        needed_client_ids.update(parse_subset_clients(s))
    return sorted(needed_client_ids)


def load_clients(
    chosen_df: pd.DataFrame,
    csv_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[int, ClientData]:
    if csv_path is None:
        df_needed = chosen_df
    else:
        needed_graph_ids = collect_needed_client_ids(chosen_df)
        df_data = pd.read_csv(csv_path).copy()
        df_needed = df_data[df_data["graph_id"].isin(needed_graph_ids)].copy()

        if verbose:
            print(f"\nNeed to load {len(df_needed)} unique clients.")

    id_to_client: Dict[int, ClientData] = {}
    for _, row in df_needed.iterrows():
        graph_id = int(row["graph_id"])
        data_dir = row["data_dir"]
        dataset_id = row["dataset_id"]

        train_g, val_g, test_g = load_client_from_dir(data_dir)
        train_h = make_bidirected_hetero(train_g)
        val_h = make_bidirected_hetero(val_g)
        test_h = make_bidirected_hetero(test_g)

        id_to_client[graph_id] = ClientData(
            graph_id=str(graph_id),
            data_dir=data_dir,
            dataset_id=dataset_id,
            train_g=train_g,
            val_g=val_g,
            test_g=test_g,
            train_h=train_h,
            val_h=val_h,
            test_h=test_h,
        )

    return id_to_client


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressPrinter:
    def __init__(self, total_runs: int):
        self.total_runs = total_runs
        self.done = 0

    def step(self, label: str, run_start_time: float | None = None):
        self.done += 1
        print(label)
        if run_start_time is not None:
            run_elapsed = time.perf_counter() - run_start_time
            print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(
            f"==================== {self.done} / {self.total_runs} ===================="
        )
        print("=======================================================")
