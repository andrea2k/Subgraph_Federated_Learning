from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from andrea.helper_funcs_multihead.benchmark_config import resolve_benchmark_paths
from andrea.helper_funcs_multihead.load_client_helper import (
    TASKS,
    audit_global_centralized_client,
    audit_loaded_q_label_masks,
    load_client_from_dir,
    load_clients,
    load_global_centralized_client_from_subset_row,
    parse_subset_clients,
)


MASK_SPLITS = ("train", "val", "test")


def _filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    only_q = os.environ.get("ONLY_Q")
    if only_q is None:
        return df
    if "q_value" not in df.columns:
        raise ValueError("ONLY_Q is set but selected_subset.csv has no q_value column.")
    q = float(only_q)
    return df[
        pd.to_numeric(df["q_value"], errors="coerce").round(10).eq(round(q, 10))
    ].copy()


def _audit_apple_initialization(row: pd.Series, id_to_client) -> None:
    ids = parse_subset_clients(str(row["subset_clients"]))
    sizes = torch.tensor(
        [float(id_to_client[gid].train_g.num_nodes) for gid in ids],
        dtype=torch.float64,
    )
    p0 = sizes / sizes.sum().clamp_min(1.0)
    expected = torch.full_like(p0, 1.0 / len(ids))
    print("APPLE p0:", p0.tolist())
    if not torch.allclose(p0, expected, atol=1e-12, rtol=0.0):
        raise AssertionError(
            "APPLE sample-size initialization is not uniform because client sizes differ. "
            f"sizes={sizes.tolist()}, p0={p0.tolist()}"
        )
    print(f"PASS APPLE initialization: every donor starts at 1/{len(ids)}")


def _audit_label_invariance(row: pd.Series, registry: pd.DataFrame, global_client) -> None:
    global_g = global_client.train_g
    if not hasattr(global_g, "planted_community_id"):
        print("Label-invariance audit: not a planted-community graph; skipped.")
        return

    community_id = global_g.planted_community_id.detach().cpu().long()
    subset_ids = parse_subset_clients(str(row["subset_clients"]))
    client_rows = registry[
        pd.to_numeric(registry["graph_id"], errors="coerce").isin(subset_ids)
    ].copy()
    if client_rows.empty:
        raise AssertionError("No registry rows found for the selected clients.")

    if "structural_cluster_id" in client_rows.columns:
        cluster_col = "structural_cluster_id"
    elif "planted_community_id" in client_rows.columns:
        cluster_col = "planted_community_id"
    else:
        raise AssertionError("Planted registry lacks structural_cluster_id.")

    local_edge_sum = 0
    for cid, group in client_rows.groupby(cluster_col):
        source = group.iloc[0]
        local_train, _, _ = load_client_from_dir(str(source["data_dir"]))
        local_edge_sum += int(local_train.edge_index.size(1))
        nodes = torch.where(community_id == int(cid))[0]
        global_y = global_g.y.detach().cpu()[nodes]
        local_y = local_train.y.detach().cpu()
        if global_y.shape != local_y.shape:
            raise AssertionError(
                f"community {cid}: global/local label shape mismatch "
                f"{tuple(global_y.shape)} vs {tuple(local_y.shape)}"
            )
        mismatch = global_y.ne(local_y)
        mismatch_per_task = {
            task: int(mismatch[:, task_idx].sum().item())
            for task_idx, task in enumerate(TASKS)
        }
        print(f"community {cid} label mismatch:", mismatch_per_task)
        if int(mismatch.sum().item()) != 0:
            raise AssertionError(
                f"Option C violated in community {cid}: {mismatch_per_task}"
            )

    global_edges = int(global_g.edge_index.size(1))
    bridge_edges = global_edges - local_edge_sum
    if bridge_edges < 0:
        raise AssertionError(
            f"Global edge count {global_edges} is below local sum {local_edge_sum}."
        )
    print(
        "global/local edges:",
        {"global": global_edges, "local_sum": local_edge_sum, "bridges": bridge_edges},
    )
    print("PASS Option C: local and global labels are exactly identical.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup",
        choices=["five_client", "twenty_client", "5client", "20client"],
        default=None,
    )
    args = parser.parse_args()
    if args.setup is not None:
        os.environ["BENCHMARK_SETUP"] = args.setup

    benchmark = resolve_benchmark_paths()
    selected_path = Path(benchmark.selected_subsets_csv_path)
    registry_path = Path(benchmark.registry_csv_path)

    print("=" * 100)
    print("BENCHMARK PIPELINE AUDIT")
    print("=" * 100)
    print("setup:", benchmark.setup)
    print("selected subsets:", selected_path)
    print("client registry:", registry_path)

    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)

    chosen_df = _filter_rows(pd.read_csv(selected_path))
    registry = pd.read_csv(registry_path)
    if chosen_df.empty:
        raise ValueError("No selected subsets remain after filtering.")

    id_to_client = load_clients(
        chosen_df,
        csv_path=str(registry_path),
        verbose=True,
        mask_splits=MASK_SPLITS,
    )
    audit_loaded_q_label_masks(
        chosen_df,
        id_to_client,
        strict=True,
        mask_splits=MASK_SPLITS,
    )

    for _, row in chosen_df.iterrows():
        print("\n" + "#" * 100)
        print("subset_id:", row.get("subset_id"))
        print("q_value:", row.get("q_value"))
        print("#" * 100)

        _audit_apple_initialization(row, id_to_client)

        global_client = load_global_centralized_client_from_subset_row(
            row,
            str(registry_path),
            verbose=True,
        )
        audit_global_centralized_client(global_client, strict=True)
        _audit_label_invariance(row, registry, global_client)

    print("\n" + "=" * 100)
    print("ALL PIPELINE AUDITS PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()
