#!/usr/bin/env python3
"""Compute exact model payloads for the paper's communication accounting.

The reproduction pipeline calls this for both benchmarks using the shared JSON
architecture and current client registries.
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
from fedptha.data.clients import load_clients, parse_subset_clients
from fedptha.training.runtime import build_model_context
from fedptha.training.runtime import make_task_head_model

HEAD_PREFIX = "output_head.heads."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark-setup",
        choices=["five_client", "twenty_client"],
        default="five_client",
    )
    p.add_argument("--selected-subsets-csv", default=None)
    p.add_argument("--registry-csv", default=None)
    p.add_argument("--config-path", default="configs/training.json")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--regions", type=int, default=4)
    p.add_argument("--representatives", type=int, default=5)
    p.add_argument(
        "--output-dir",
        default=str(generated_root() / "communication_payload"),
    )
    p.add_argument(
        "--mask-splits",
        default="train,val,test",
        help="Use the same loading path as final experiments; masks do not affect model shape.",
    )
    return p.parse_args()


def load_cfg(path: str) -> Dict:
    from fedptha.config import read_config

    return dict(read_config(path)["model"])


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def summarize_items(items: Iterable[Tuple[str, torch.Tensor]]) -> Dict[str, int]:
    elems = 0
    nbytes = 0
    ntensors = 0
    for _, t in items:
        ntensors += 1
        elems += int(t.numel())
        nbytes += tensor_bytes(t)
    return {"tensors": ntensors, "elements": elems, "bytes": nbytes}


def is_head_name(name: str) -> bool:
    return str(name).startswith(HEAD_PREFIX)


def audit_model(model: torch.nn.Module) -> Tuple[Dict[str, int], pd.DataFrame]:
    state = dict(model.state_dict())
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    rows: List[Dict] = []
    for name, t in state.items():
        is_param = name in params
        is_buffer = name in buffers
        rows.append(
            {
                "name": name,
                "shape": "x".join(str(int(v)) for v in t.shape),
                "dtype": str(t.dtype),
                "element_size": int(t.element_size()),
                "elements": int(t.numel()),
                "bytes": tensor_bytes(t),
                "is_parameter": bool(is_param),
                "is_buffer": bool(is_buffer),
                "is_head": bool(is_head_name(name)),
                "section": "head" if is_head_name(name) else "non_head",
            }
        )

    tensor_df = pd.DataFrame(rows)

    def sm(names: List[str]) -> Dict[str, int]:
        return summarize_items((name, state[name]) for name in names)

    state_names = list(state)
    param_names = list(params)
    buffer_names = list(buffers)
    head_state_names = [n for n in state_names if is_head_name(n)]
    nonhead_state_names = [n for n in state_names if not is_head_name(n)]
    head_param_names = [n for n in param_names if is_head_name(n)]
    nonhead_param_names = [n for n in param_names if not is_head_name(n)]
    head_buffer_names = [n for n in buffer_names if is_head_name(n)]
    nonhead_buffer_names = [n for n in buffer_names if not is_head_name(n)]

    out: Dict[str, int] = {}
    groups = {
        "state_full": sm(state_names),
        "state_head": sm(head_state_names),
        "state_nonhead": sm(nonhead_state_names),
        "param_full": summarize_items(params.items()),
        "param_head": summarize_items((n, params[n]) for n in head_param_names),
        "param_nonhead": summarize_items((n, params[n]) for n in nonhead_param_names),
        "buffer_full": summarize_items(buffers.items()),
        "buffer_head": summarize_items((n, buffers[n]) for n in head_buffer_names),
        "buffer_nonhead": summarize_items(
            (n, buffers[n]) for n in nonhead_buffer_names
        ),
    }
    for group, vals in groups.items():
        for key, value in vals.items():
            out[f"{group}_{key}"] = int(value)

    # Exact partition identities.
    out["state_partition_bytes_ok"] = int(
        out["state_full_bytes"] == out["state_nonhead_bytes"] + out["state_head_bytes"]
    )
    out["param_partition_bytes_ok"] = int(
        out["param_full_bytes"] == out["param_nonhead_bytes"] + out["param_head_bytes"]
    )
    out["head_state_equals_head_params"] = int(
        out["state_head_bytes"] == out["param_head_bytes"]
    )
    out["full_state_equals_full_params"] = int(
        out["state_full_bytes"] == out["param_full_bytes"]
    )

    return out, tensor_df


def mib(n: int | float) -> float:
    return float(n) / (1024.0**2)


def gib(n: int | float) -> float:
    return float(n) / (1024.0**3)


def communication_rows(
    *,
    setup: str,
    K: int,
    Q: int,
    R: int,
    J: int,
    audit: Dict[str, int],
) -> List[Dict]:
    """Compute byte-level communication using the implementation audit.

    Exact byte conventions used here:
      - FedAvg/FedProx/FedALA/GCFL+/TaskWeighted/cwFedAvg use full state_dict.
      - APPLE donor mixing uses learnable parameters only; receiver-local buffers
        are not donor-mixed.
      - Support-Simplex repeated backbone exchange uses non-head state_dict,
        while receiver-specific donor-head routing uses head parameters.
      - Hierarchical one-shot head summaries are built from head state_dict
        entries in the final hierarchical runner; if heads have no buffers this
        equals the head-parameter payload.
    """
    P_state = int(audit["state_full_bytes"])
    P_param = int(audit["param_full_bytes"])
    B_state = int(audit["state_nonhead_bytes"])
    B_param = int(audit["param_nonhead_bytes"])
    H_state = int(audit["state_head_bytes"])
    H_param = int(audit["param_head_bytes"])

    rows: List[Dict] = []

    def add(method: str, per_round: int | None, total: int | None, note: str = ""):
        rows.append(
            {
                "setup": setup,
                "K": int(K),
                "Q": int(Q),
                "method": method,
                "per_round_bytes": None if per_round is None else int(per_round),
                "per_round_MiB": None if per_round is None else mib(per_round),
                "total_bytes": None if total is None else int(total),
                "total_MiB": None if total is None else mib(total),
                "total_GiB": None if total is None else gib(total),
                "note": note,
            }
        )

    add("Fully Local", 0, 0)

    fedavg_round = 2 * K * P_state
    for name in [
        "FedAvg",
        "FedProx",
        "FedALA",
        "GCFL+",
        "Task-Weighted FedAvg",
        "cwFedAvg-Task",
    ]:
        add(name, fedavg_round, Q * fedavg_round, "full state_dict payload")

    apple_round = K * K * P_param
    add(
        "APPLE",
        apple_round,
        Q * apple_round,
        "learnable donor parameters; receiver buffers remain local",
    )

    ss_round = 2 * K * B_state + K * K * H_param
    for name in [
        "Support-Simplex",
        "Support-Simplex + PostALA",
        "Learned-Simplex + PostALA",
        "Support-Only + PostALA",
    ]:
        add(
            name,
            ss_round,
            Q * ss_round,
            "non-head state_dict for backbone; head parameters for donor routing",
        )

    if setup == "twenty_client":
        # Physical and hierarchical costs use region-specific model contexts;
        # regional_payload.py computes their exact parameter counts.
        rep_round = 2 * J * B_state + J * J * H_param
        add(
            "Task-Specialist Representatives",
            rep_round,
            Q * rep_round,
            f"training only, J={J}; validation broadcasts excluded",
        )

    add(
        "Global-Centralized",
        None,
        None,
        "N/A: not a federated model-exchange competitor",
    )
    return rows


def main() -> None:
    args = parse_args()

    bench = read_config(f"configs/benchmarks/{args.benchmark_setup}.json")
    selected_path = Path(
        args.selected_subsets_csv or absolute(bench["selected_subsets"])
    )
    registry_path = Path(args.registry_csv or absolute(bench["registry"]))
    out_dir = Path(args.output_dir) / args.benchmark_setup
    out_dir.mkdir(parents=True, exist_ok=True)

    print("benchmark setup:", args.benchmark_setup)
    print("selected subsets:", selected_path)
    print("registry:", registry_path)
    print("config:", args.config_path)

    chosen_df = pd.read_csv(selected_path)
    mask_splits = tuple(s.strip() for s in args.mask_splits.split(",") if s.strip())
    id_to_client = load_clients(
        chosen_df,
        csv_path=str(registry_path),
        verbose=True,
        mask_splits=mask_splits,
    )

    cfg = load_cfg(args.config_path)
    subset_rows: List[Dict] = []
    tensor_frames: List[pd.DataFrame] = []
    audits: List[Dict] = []

    for row_idx, row in chosen_df.iterrows():
        subset_ids = parse_subset_clients(str(row["subset_clients"]))
        clients = [id_to_client[int(gid)] for gid in subset_ids]
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

        audit, tensor_df = audit_model(model)
        q_value = row.get("q_value", None)
        record = {
            "setup": args.benchmark_setup,
            "row_idx": int(row_idx),
            "q_value": q_value,
            "subset_clients": str(row["subset_clients"]),
            "K": len(clients),
            "x_dim": int(ctx["x_dim"]),
            "out_dim": int(ctx["out_dim"]),
            "ego_dim": int(ctx["ego_dim"]),
            "in_vocab": int(ctx["in_vocab"]),
            "out_vocab": int(ctx["out_vocab"]),
            **audit,
        }
        subset_rows.append(record)
        audits.append(record)

        tensor_df.insert(0, "setup", args.benchmark_setup)
        tensor_df.insert(1, "q_value", q_value)
        tensor_df.insert(2, "subset_clients", str(row["subset_clients"]))
        tensor_frames.append(tensor_df)

        print(
            f"q={q_value} K={len(clients)} "
            f"P_state={audit['state_full_bytes']} B "
            f"P_param={audit['param_full_bytes']} B "
            f"B_state={audit['state_nonhead_bytes']} B "
            f"H_state={audit['state_head_bytes']} B "
            f"H_param={audit['param_head_bytes']} B"
        )

    subset_df = pd.DataFrame(subset_rows)

    # Defensive consistency check: one benchmark setup must use one client count.
    if subset_df["K"].nunique() != 1:
        raise RuntimeError(
            "Client count differs across selected subsets; "
            "cannot use one communication table."
        )

    tensor_all = pd.concat(tensor_frames, ignore_index=True)
    subset_df.to_csv(out_dir / "payload_by_subset.csv", index=False)
    tensor_all.to_csv(out_dir / "tensor_inventory.csv", index=False)

    # Shape/payload invariance audit across q/subsets.
    payload_cols = [
        "state_full_bytes",
        "param_full_bytes",
        "state_head_bytes",
        "param_head_bytes",
        "state_nonhead_bytes",
        "param_nonhead_bytes",
        "buffer_full_bytes",
        "buffer_head_bytes",
        "buffer_nonhead_bytes",
    ]
    unique_payloads = subset_df[payload_cols].drop_duplicates()
    invariant = len(unique_payloads) == 1

    summary = {
        "setup": args.benchmark_setup,
        "num_selected_subsets": int(len(subset_df)),
        "payload_identical_across_selected_subsets": bool(invariant),
        "unique_payload_rows": int(len(unique_payloads)),
        "head_state_equals_head_params_all": bool(
            (subset_df["head_state_equals_head_params"] == 1).all()
        ),
        "full_state_equals_full_params_all": bool(
            (subset_df["full_state_equals_full_params"] == 1).all()
        ),
    }
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Communication values require one stable architecture payload per setup.
    # If not stable, report the issue rather than silently choosing one.
    if not invariant:
        print(
            "WARNING: payload differs across selected subsets; communication table not frozen."
        )
        print(unique_payloads.to_string(index=False))
    else:
        a = subset_df.iloc[0].to_dict()
        K = int(subset_df.iloc[0]["K"])

        # Equal-region formulas used for the 20-client diagnostics require
        # an integer number of clients per region.
        if args.benchmark_setup == "twenty_client" and K % int(args.regions) != 0:
            raise RuntimeError(
                f"K={K} is not divisible by R={args.regions}; "
                "cannot apply the equal-region communication formula."
            )

        comm = pd.DataFrame(
            communication_rows(
                setup=args.benchmark_setup,
                K=K,
                Q=int(args.rounds),
                R=int(args.regions),
                J=int(args.representatives),
                audit=a,
            )
        )
        comm.to_csv(out_dir / "communication_table.csv", index=False)
        print("\nCommunication table:")
        print(comm.to_string(index=False))

    # Separate compact buffer report, one copy per unique tensor name/dtype/shape.
    buffer_df = tensor_all[tensor_all["is_buffer"]].copy()
    buffer_df = buffer_df.drop_duplicates(subset=["name", "shape", "dtype", "bytes"])
    buffer_df.to_csv(out_dir / "buffer_inventory.csv", index=False)

    print("\nWrote:")
    for p in [
        out_dir / "payload_by_subset.csv",
        out_dir / "tensor_inventory.csv",
        out_dir / "buffer_inventory.csv",
        out_dir / "audit_summary.json",
        out_dir / "communication_table.csv",
    ]:
        if p.exists():
            print(" ", p)


if __name__ == "__main__":
    main()
