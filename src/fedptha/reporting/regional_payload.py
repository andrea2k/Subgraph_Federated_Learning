#!/usr/bin/env python3
"""Compute payloads for the physical-community communication diagnostic.

Uses the current twenty-client registry and the same model configuration as training.
"""

from __future__ import annotations

from fedptha.paths import absolute, generated_root

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch

from fedptha.config import read_config
from fedptha.data.clients import (
    load_clients,
    parse_subset_clients,
)
from fedptha.training.runtime import build_model_context
from fedptha.training.runtime import make_task_head_model

# Import the exact expansion routine used by the final oracle runner rather
# than reimplementing physical-community membership logic here.
from fedptha.training.oracle import (
    expand_oracle_hard_clusters,
)

HEAD_PREFIX = "output_head.heads."
PREVIOUS_PHYSICAL_PER_ROUND_BYTES = 216_053_840
PREVIOUS_PHYSICAL_TOTAL_GIB = 16.097265
PREVIOUS_HIER_INTER_BYTES = 1_690_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark-setup",
        choices=["twenty_client"],
        default="twenty_client",
    )
    p.add_argument("--selected-subsets-csv", default=None)
    p.add_argument("--registry-csv", default=None)
    p.add_argument("--config-path", default="configs/training.json")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument(
        "--output-dir",
        default=str(
            generated_root() / "communication_payload" / "twenty_client/oracle_regional"
        ),
    )
    p.add_argument(
        "--mask-splits",
        default="train,val,test",
        help="Same client-loading path as the final oracle runner.",
    )
    return p.parse_args()


def load_cfg(path: str) -> Dict:
    from fedptha.config import read_config

    return dict(read_config(path)["model"])


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def summarize(items: Iterable[Tuple[str, torch.Tensor]]) -> Tuple[int, int]:
    elements = 0
    nbytes = 0
    for _, tensor in items:
        elements += int(tensor.numel())
        nbytes += tensor_bytes(tensor)
    return elements, nbytes


def audit_payload(model: torch.nn.Module) -> Dict[str, int]:
    state = dict(model.state_dict())
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    state_head = [(n, t) for n, t in state.items() if n.startswith(HEAD_PREFIX)]
    state_nonhead = [(n, t) for n, t in state.items() if not n.startswith(HEAD_PREFIX)]
    param_head = [(n, t) for n, t in params.items() if n.startswith(HEAD_PREFIX)]
    param_nonhead = [(n, t) for n, t in params.items() if not n.startswith(HEAD_PREFIX)]

    state_full_e, state_full_b = summarize(state.items())
    state_head_e, state_head_b = summarize(state_head)
    state_nonhead_e, state_nonhead_b = summarize(state_nonhead)
    param_full_e, param_full_b = summarize(params.items())
    param_head_e, param_head_b = summarize(param_head)
    param_nonhead_e, param_nonhead_b = summarize(param_nonhead)
    buffer_e, buffer_b = summarize(buffers.items())

    if state_full_b != state_nonhead_b + state_head_b:
        raise RuntimeError("state_dict payload did not partition into head + non-head")
    if param_full_b != param_nonhead_b + param_head_b:
        raise RuntimeError("parameter payload did not partition into head + non-head")

    return {
        "P_state_elements": state_full_e,
        "P_state_bytes": state_full_b,
        "P_param_elements": param_full_e,
        "P_param_bytes": param_full_b,
        "B_state_elements": state_nonhead_e,
        "B_state_bytes": state_nonhead_b,
        "B_param_elements": param_nonhead_e,
        "B_param_bytes": param_nonhead_b,
        "H_state_elements": state_head_e,
        "H_state_bytes": state_head_b,
        "H_param_elements": param_head_e,
        "H_param_bytes": param_head_b,
        "buffer_elements": buffer_e,
        "buffer_bytes": buffer_b,
        "head_state_equals_head_params": int(state_head_b == param_head_b),
    }


def mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def main() -> None:
    args = parse_args()

    bench = read_config(f"configs/benchmarks/{args.benchmark_setup}.json")
    selected_path = Path(
        args.selected_subsets_csv or absolute(bench["selected_subsets"])
    )
    registry_path = Path(args.registry_csv or absolute(bench["registry"]))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("benchmark setup: twenty_client")
    print("selected subsets:", selected_path)
    print("registry:", registry_path)
    print("config:", args.config_path)

    parent_df = pd.read_csv(selected_path)
    if parent_df.empty:
        raise RuntimeError("Selected-subset table is empty.")

    # Exact physical-community expansion used by the final oracle runner.
    regional_df = expand_oracle_hard_clusters(parent_df)

    expected_rows = 4 * len(parent_df)
    if len(regional_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} regional rows from {len(parent_df)} parent rows, "
            f"got {len(regional_df)}."
        )

    mask_splits = tuple(s.strip() for s in args.mask_splits.split(",") if s.strip())
    id_to_client = load_clients(
        regional_df,
        csv_path=str(registry_path),
        verbose=True,
        mask_splits=mask_splits,
    )

    cfg = load_cfg(args.config_path)
    payload_rows: List[Dict] = []

    for row_idx, row in regional_df.iterrows():
        subset_ids = parse_subset_clients(str(row["subset_clients"]))
        clients = [id_to_client[int(gid)] for gid in subset_ids]
        n_r = len(clients)
        if n_r != 5:
            raise RuntimeError(
                f"Oracle region row {row_idx} has {n_r} clients; expected 5."
            )

        ctx = build_model_context(clients, cfg)
        model = make_task_head_model(
            cfg,
            ctx["x_dim"],
            ctx["out_dim"],
            ctx["deg_fwd_hist"],
            ctx["deg_rev_hist"],
            ctx["ego_dim"],
            ctx["in_vocab"],
            ctx["out_vocab"],
        ).cpu()

        payload = audit_payload(model)
        q_value = row.get("q_value", None)
        region_id = int(row["oracle_hard_cluster_id"])

        record = {
            "q_value": q_value,
            "parent_subset_id": str(row.get("oracle_parent_subset_id", "")),
            "parent_subset_clients": str(row.get("oracle_parent_subset_clients", "")),
            "region_id": region_id,
            "base_graph_id": int(row["oracle_hard_base_graph_id"]),
            "subset_clients": str(row["subset_clients"]),
            "n_r": n_r,
            "x_dim": int(ctx["x_dim"]),
            "out_dim": int(ctx["out_dim"]),
            "ego_dim": int(ctx["ego_dim"]),
            "in_vocab": int(ctx["in_vocab"]),
            "out_vocab": int(ctx["out_vocab"]),
            **payload,
        }
        payload_rows.append(record)

        print(
            f"q={q_value} region={region_id} n={n_r} "
            f"in_vocab={ctx['in_vocab']} out_vocab={ctx['out_vocab']} "
            f"B_state={payload['B_state_bytes']} B "
            f"H_state={payload['H_state_bytes']} B "
            f"H_param={payload['H_param_bytes']} B "
            f"P_state={payload['P_state_bytes']} B"
        )

    payload_df = pd.DataFrame(payload_rows)
    payload_df = payload_df.sort_values(
        ["q_value", "parent_subset_id", "region_id"],
        kind="stable",
    ).reset_index(drop=True)
    payload_path = out_dir / "regional_payloads.csv"
    payload_df.to_csv(payload_path, index=False)

    # Each parent 20-client selected subset must contain exactly four regions.
    group_cols = ["q_value", "parent_subset_id", "parent_subset_clients"]
    communication_rows: List[Dict] = []

    for keys, group in payload_df.groupby(group_cols, dropna=False, sort=True):
        q_value, parent_subset_id, parent_subset_clients = keys
        group = group.sort_values("region_id")

        if len(group) != 4 or group["region_id"].nunique() != 4:
            raise RuntimeError(
                f"q={q_value} parent={parent_subset_id}: expected exactly four regions."
            )
        if int(group["n_r"].sum()) != 20:
            raise RuntimeError(
                f"q={q_value} parent={parent_subset_id}: regional client counts do not sum to 20."
            )

        # Exact region-specific repeated communication:
        # sum_r [2*n_r*B_state_r + n_r^2*H_param_r].
        region_terms = 2 * group["n_r"].astype(int) * group["B_state_bytes"].astype(
            int
        ) + (group["n_r"].astype(int) ** 2) * group["H_param_bytes"].astype(int)
        physical_per_round = int(region_terms.sum())
        physical_total = int(args.rounds) * physical_per_round

        # Exact canonical hierarchy delivery: one complete external head-state
        # collection is delivered to each receiving client. Writing this as a
        # regional sum keeps the expression valid even if head payloads were to
        # differ across model contexts.
        hierarchy_inter = int(
            (group["n_r"].astype(int) * group["H_state_bytes"].astype(int)).sum()
        )
        hierarchy_total = physical_total + hierarchy_inter

        heads_equal = group["H_state_bytes"].nunique() == 1
        head_state_param_equal = bool(
            (group["head_state_equals_head_params"] == 1).all()
        )

        communication_rows.append(
            {
                "q_value": q_value,
                "parent_subset_id": parent_subset_id,
                "parent_subset_clients": parent_subset_clients,
                "rounds": int(args.rounds),
                "regions": 4,
                "total_clients": 20,
                "physical_per_round_bytes": physical_per_round,
                "physical_per_round_MiB": mib(physical_per_round),
                "physical_total_bytes": physical_total,
                "physical_total_MiB": mib(physical_total),
                "physical_total_GiB": gib(physical_total),
                "hierarchy_inter_bytes": hierarchy_inter,
                "hierarchy_inter_MiB": mib(hierarchy_inter),
                "hierarchy_total_bytes": hierarchy_total,
                "hierarchy_total_MiB": mib(hierarchy_total),
                "hierarchy_total_GiB": gib(hierarchy_total),
                "all_region_head_state_sizes_equal": heads_equal,
                "head_state_equals_head_params_all_regions": head_state_param_equal,
                "previous_physical_per_round_bytes": PREVIOUS_PHYSICAL_PER_ROUND_BYTES,
                "delta_vs_previous_physical_per_round_bytes": (
                    physical_per_round - PREVIOUS_PHYSICAL_PER_ROUND_BYTES
                ),
                "previous_hierarchy_inter_bytes": PREVIOUS_HIER_INTER_BYTES,
                "delta_vs_previous_hierarchy_inter_bytes": (
                    hierarchy_inter - PREVIOUS_HIER_INTER_BYTES
                ),
            }
        )

    comm_df = pd.DataFrame(communication_rows)
    comm_path = out_dir / "oracle_regional_communication.csv"
    comm_df.to_csv(comm_path, index=False)

    # Summarize whether exact regional values are invariant across q and whether
    # the previous whole-20C approximation remains exact.
    payload_signature_cols = [
        "region_id",
        "in_vocab",
        "out_vocab",
        "B_state_bytes",
        "H_state_bytes",
        "H_param_bytes",
        "P_state_bytes",
    ]

    # Compare each region's signature across q by dropping q / parent id.
    signatures_by_region = {
        str(region): int(
            len(
                payload_df.loc[
                    payload_df["region_id"] == region, payload_signature_cols
                ].drop_duplicates()
            )
        )
        for region in sorted(payload_df["region_id"].unique())
    }

    summary = {
        "setup": "twenty_client_oracle_regional",
        "num_parent_selected_subsets": int(len(parent_df)),
        "num_regional_rows": int(len(payload_df)),
        "regional_rows_per_parent": 4,
        "regional_client_count": 5,
        "region_signature_unique_rows": signatures_by_region,
        "physical_per_round_invariant_across_parents": bool(
            comm_df["physical_per_round_bytes"].nunique() == 1
        ),
        "hierarchy_inter_invariant_across_parents": bool(
            comm_df["hierarchy_inter_bytes"].nunique() == 1
        ),
        "previous_physical_per_round_remains_exact_all": bool(
            (
                comm_df["physical_per_round_bytes"] == PREVIOUS_PHYSICAL_PER_ROUND_BYTES
            ).all()
        ),
        "previous_hierarchy_inter_remains_exact_all": bool(
            (comm_df["hierarchy_inter_bytes"] == PREVIOUS_HIER_INTER_BYTES).all()
        ),
        "previous_physical_total_gib_reference": PREVIOUS_PHYSICAL_TOTAL_GIB,
    }

    if comm_df["physical_per_round_bytes"].nunique() == 1:
        value = int(comm_df.iloc[0]["physical_per_round_bytes"])
        summary["final_physical_per_round_bytes"] = value
        summary["final_physical_per_round_MiB"] = mib(value)
        summary["final_physical_total_bytes"] = int(args.rounds) * value
        summary["final_physical_total_GiB"] = gib(int(args.rounds) * value)

    if comm_df["hierarchy_inter_bytes"].nunique() == 1:
        value = int(comm_df.iloc[0]["hierarchy_inter_bytes"])
        summary["final_hierarchy_inter_bytes"] = value
        summary["final_hierarchy_inter_MiB"] = mib(value)

    if comm_df["hierarchy_total_bytes"].nunique() == 1:
        value = int(comm_df.iloc[0]["hierarchy_total_bytes"])
        summary["final_hierarchy_total_bytes"] = value
        summary["final_hierarchy_total_GiB"] = gib(value)

    summary_path = out_dir / "oracle_regional_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExact oracle regional communication:")
    display_cols = [
        "q_value",
        "physical_per_round_bytes",
        "physical_per_round_MiB",
        "physical_total_GiB",
        "hierarchy_inter_bytes",
        "hierarchy_total_GiB",
        "delta_vs_previous_physical_per_round_bytes",
    ]
    print(comm_df[display_cols].to_string(index=False))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nWrote:")
    print(" ", payload_path)
    print(" ", comm_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
