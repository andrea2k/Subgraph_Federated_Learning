#!/usr/bin/env python3
"""
Generate a planted-community chordal benchmark for the multi-head FGL pipeline.

This script does everything in one pass:

1. Generate train/val/test global big graphs.
   - Each global graph consists of several chordal communities.
   - Communities are connected by forward-only bridge edges on a community DAG.
   - Local labels are computed before bridging.
   - Global labels are recomputed after bridging and must be exactly identical.

2. Save community-induced client graphs.
   - Each planted community becomes one base client directory:
       train.pt / val.pt / test.pt
   - Labels are computed from each local community graph itself.
   - Ports are computed independently for local and global graph views.

3. Create q-masked virtual clients.
   - For each community base client and each assigned task, create one virtual row.
   - The virtual rows point to the community base client data_dir.
   - The actual label_mask is still applied later by load_client_helper.py.

Default first benchmark:
  4 communities, 2000 nodes each
  d = 3,3,3.5,3.5
  delta = 8,12,8,12
  bridge_edges_per_pair = 40
  q = 0.8

Run from project root:
  python -m andrea.generate_planted_community_graph_data --overwrite

Or test only high q:
  python -m andrea.generate_planted_community_graph_data --q-values 0.8 --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import subgraph

from scripts.data.simulator import GraphSimulator
from utils.gcn_utils import GraphData
from andrea.multigraph_generation import TASKS, TASK_FUNCS, set_y_and_get_motifs

SPLITS = ["train", "val", "test"]
MASK_MODE = "q_task_label_allocation"
MASK_APPLY_SPLIT = "train"
Q_ALLOCATION_MODE = "deterministic_disjoint_positive_label_chunks"
CONTROLLED_BENCHMARK = "planted_community_q_task_label_heterogeneity"
VIRTUAL_GRAPH_ID_START = 4_000_000
BASE_GRAPH_ID_START = 3_900_000
BRIDGE_MODE = "forward_dag"
LABEL_SEMANTICS = "label_invariant_local_equals_global"


# =============================================================================
# small utilities
# =============================================================================

def stable_seed(base_seed: int, *parts: object) -> int:
    """Deterministic 31-bit seed from base_seed and string parts."""
    msg = "|".join([str(base_seed)] + [str(p) for p in parts]).encode("utf-8")
    h = hashlib.sha256(msg).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def parse_float_list(s: str, name: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} cannot be empty.")
    return vals


def parse_q_values(s: str) -> List[float]:
    vals = parse_float_list(s, "q-values")
    for q in vals:
        validate_q(q)
    return vals


def q_iid_value() -> float:
    return 1.0 / float(len(TASKS))


def q_other_share(q: float) -> float:
    return (1.0 - float(q)) / float(len(TASKS) - 1)


def q_tag(q: float) -> str:
    return str(int(round(float(q) * 100)))


def validate_q(q: float) -> None:
    q = float(q)
    if q < 0.0 or q > 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    if q + 1e-12 < q_iid_value():
        raise ValueError(
            f"q={q} is below IID q=1/T={q_iid_value():.6f}. "
            "Use q >= 1/T for this benchmark."
        )


def allocation_counts(num_items: int, shares: Sequence[float]) -> List[int]:
    """
    Convert fractional q shares into exact integer counts that sum to num_items.

    This mirrors the logic used by load_client_helper.py for the real tensor
    label_mask allocation, so CSV metadata and loaded tensors agree exactly.
    """
    n = int(num_items)
    raw = [float(s) * n for s in shares]
    counts = [int(np.floor(x)) for x in raw]
    remainder = n - int(sum(counts))
    if remainder < 0:
        raise ValueError(f"allocation counts overfull: n={n}, counts={counts}")

    frac_order = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - np.floor(raw[i]), -i),
        reverse=True,
    )
    for i in frac_order[:remainder]:
        counts[i] += 1

    if int(sum(counts)) != n:
        raise AssertionError(f"counts do not sum to {n}: {counts}")
    return counts


def q_counts_for_label_task(num_positives: int, *, label_task: str, q_value: float) -> List[int]:
    """
    Counts are ordered by receiving client's assigned task, i.e. TASKS order.
    The client whose assigned_task == label_task receives q of that task's positives.
    Every other assigned-task client receives (1-q)/(T-1).
    """
    q = float(q_value)
    off = q_other_share(q)
    shares = [q if receiving_task == label_task else off for receiving_task in TASKS]
    return allocation_counts(int(num_positives), shares)


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {path}\n"
                "Use --overwrite if you intentionally want to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 110)
    print(title)
    print("-" * 110)


def safe_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# =============================================================================
# config
# =============================================================================

@dataclass
class GenerationConfig:
    dataset_id: str
    nodes_per_community: int
    community_ds: List[float]
    community_deltas: List[float]
    bridge_edges_per_pair: int
    seed: int
    generator: str
    q_values: List[float]
    global_out_dir: str
    community_client_root: str
    q_manifest_dir: str
    overwrite: bool
    max_time: int
    bridge_mode: str
    label_semantics: str


# =============================================================================
# graph construction
# =============================================================================

def make_simulator(
    *,
    n: int,
    d: float,
    delta: float,
    generator: str,
    seed: int,
    max_time: int,
) -> GraphSimulator:
    return GraphSimulator(
        num_nodes=int(n),
        avg_degree=float(d),
        num_edges=None,
        max_time=int(max_time),
        network_type="type1",
        readout="node",
        node_feats=False,
        bidirectional=False,
        delta=float(delta),
        num_graphs=1,
        generator=str(generator),
        seed=int(seed),
    )


def raw_timestamp_edge_attr(data) -> torch.Tensor:
    """Keep only original timestamp-like columns before add_ports appends port columns."""
    if data.edge_attr is None:
        return torch.zeros((data.edge_index.size(1), 2), dtype=torch.float32)
    if data.edge_attr.size(1) >= 2:
        return data.edge_attr[:, :2].detach().clone().float()
    if data.edge_attr.size(1) == 1:
        return torch.cat([data.edge_attr.float(), data.edge_attr.float()], dim=1)
    raise ValueError("Unexpected empty edge_attr.")


def make_bridge_edges(
    *,
    node_ranges: List[Tuple[int, int]],
    edges_per_pair: int,
    rng: np.random.Generator,
    torch_gen: torch.Generator,
    max_time: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Create forward-only bridges between adjacent communities.

    Community-level direction is a DAG:
        c0 -> c1 -> c2 -> ...

    Therefore a directed cycle cannot traverse a bridge and return to an
    earlier community. This gives Option C: cycle2-cycle6 labels are invariant
    between each local community graph and the connected global graph.
    """
    bridge_edges: List[torch.Tensor] = []
    bridge_attrs: List[torch.Tensor] = []
    bridge_records: List[Dict] = []

    if edges_per_pair <= 0 or len(node_ranges) <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.float32),
            bridge_records,
        )

    for cid in range(len(node_ranges) - 1):
        src_start, src_end = node_ranges[cid]
        dst_start, dst_end = node_ranges[cid + 1]
        count = int(edges_per_pair)

        # Draw unique cross-community pairs so the requested raw edge count is
        # also the unique bridge-edge count.
        pairs = set()
        while len(pairs) < count:
            src_node = int(rng.integers(src_start, src_end))
            dst_node = int(rng.integers(dst_start, dst_end))
            pairs.add((src_node, dst_node))

        ordered = sorted(pairs)
        src_nodes = torch.tensor([u for u, _ in ordered], dtype=torch.long)
        dst_nodes = torch.tensor([v for _, v in ordered], dtype=torch.long)
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        edge_attr = torch.randint(
            low=0,
            high=max(1, int(max_time)),
            size=(count, 2),
            generator=torch_gen,
            dtype=torch.long,
        ).float()

        bridge_edges.append(edge_index)
        bridge_attrs.append(edge_attr)
        bridge_records.append(
            {
                "src_community": int(cid),
                "dst_community": int(cid + 1),
                "num_edges": int(count),
                "direction": "forward_dag",
                "src_node_min": int(src_nodes.min().item()),
                "src_node_max": int(src_nodes.max().item()),
                "dst_node_min": int(dst_nodes.min().item()),
                "dst_node_max": int(dst_nodes.max().item()),
            }
        )

    return (
        torch.cat(bridge_edges, dim=1),
        torch.cat(bridge_attrs, dim=0),
        bridge_records,
    )

def build_planted_big_graph_for_split(
    *,
    split: str,
    cfg: GenerationConfig,
) -> Tuple[
    GraphData,
    Dict[str, List],
    List[Dict],
    Dict[str, object],
    Dict[int, GraphData],
    Dict[int, Dict[str, List]],
]:
    """Build one connected global graph and its label-consistent communities."""
    n = int(cfg.nodes_per_community)
    total_nodes = int(n * len(cfg.community_ds))
    node_ranges: List[Tuple[int, int]] = []
    community_ids = torch.empty(total_nodes, dtype=torch.long)

    global_edge_parts: List[torch.Tensor] = []
    global_attr_parts: List[torch.Tensor] = []
    component_records: List[Dict] = []
    community_graphs: Dict[int, GraphData] = {}
    community_motifs: Dict[int, Dict[str, List]] = {}

    print_subheader(f"GENERATING AND LABELING LOCAL COMMUNITIES FOR SPLIT={split}")

    for cid, (d, delta) in enumerate(zip(cfg.community_ds, cfg.community_deltas)):
        offset = cid * n
        node_ranges.append((offset, offset + n))
        community_ids[offset : offset + n] = int(cid)

        seed = stable_seed(cfg.seed, split, "community", cid, "d", d, "delta", delta)
        sim = make_simulator(
            n=n,
            d=float(d),
            delta=float(delta),
            generator=cfg.generator,
            seed=seed,
            max_time=cfg.max_time,
        )
        raw = sim.generate_pytorch_graph()
        local_edge_index = raw.edge_index.detach().clone().long()
        local_edge_attr = raw_timestamp_edge_attr(raw)

        local_g = GraphData(
            x=torch.ones((n, 1), dtype=torch.float32),
            y=torch.zeros((n, len(TASKS)), dtype=torch.float32),
            edge_index=local_edge_index,
            edge_attr=local_edge_attr,
            readout="node",
            num_nodes=n,
        )
        local_g = local_g.add_ports()
        local_g, local_motifs = set_y_and_get_motifs(local_g, TASK_FUNCS)
        local_g.original_node_start = int(offset)
        local_g.original_node_end_exclusive = int(offset + n)
        local_g.planted_community_id = torch.full((n,), int(cid), dtype=torch.long)

        community_graphs[int(cid)] = local_g
        community_motifs[int(cid)] = local_motifs

        global_edge_parts.append(local_edge_index + int(offset))
        global_attr_parts.append(local_edge_attr)

        rec = {
            "split": split,
            "community_id": int(cid),
            "seed": int(seed),
            "num_nodes": int(n),
            "d": float(d),
            "delta": float(delta),
            "raw_edges": int(local_edge_index.size(1)),
            "node_start": int(offset),
            "node_end_exclusive": int(offset + n),
        }
        for task_idx, task in enumerate(TASKS):
            rec[f"local_{task}_motif_count"] = int(len(local_motifs.get(task, [])))
            rec[f"local_{task}_pos_nodes"] = int(local_g.y[:, task_idx].sum().item())
        component_records.append(rec)

        print(
            f"community={cid} | seed={seed} | nodes={n} | d={d} | delta={delta} | "
            f"raw_edges={local_edge_index.size(1)} | node_range=[{offset},{offset+n})"
        )

    rng = np.random.default_rng(stable_seed(cfg.seed, split, "bridges"))
    torch_gen = torch.Generator().manual_seed(stable_seed(cfg.seed, split, "bridge_attr"))
    bridge_edge_index, bridge_edge_attr, bridge_records = make_bridge_edges(
        node_ranges=node_ranges,
        edges_per_pair=int(cfg.bridge_edges_per_pair),
        rng=rng,
        torch_gen=torch_gen,
        max_time=cfg.max_time,
    )

    print_subheader(f"ADDING LABEL-INVARIANT FORWARD BRIDGES FOR SPLIT={split}")
    for rec in bridge_records:
        print(rec)

    edge_index = torch.cat(global_edge_parts + [bridge_edge_index], dim=1)
    edge_attr = torch.cat(global_attr_parts + [bridge_edge_attr], dim=0)
    if edge_index.size(1) != edge_attr.size(0):
        raise AssertionError("edge_index / edge_attr mismatch")

    global_g = GraphData(
        x=torch.ones((total_nodes, 1), dtype=torch.float32),
        y=torch.zeros((total_nodes, len(TASKS)), dtype=torch.float32),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
        readout="node",
        num_nodes=total_nodes,
    )
    global_g.planted_community_id = community_ids
    global_g = global_g.add_ports()
    global_g, global_motifs = set_y_and_get_motifs(global_g, TASK_FUNCS)
    global_g.planted_community_id = community_ids

    expected_y = torch.cat(
        [community_graphs[cid].y.detach().cpu() for cid in sorted(community_graphs)],
        dim=0,
    )
    actual_y = global_g.y.detach().cpu()
    mismatch = actual_y.ne(expected_y)
    mismatch_per_task = {
        task: int(mismatch[:, task_idx].sum().item())
        for task_idx, task in enumerate(TASKS)
    }
    mismatch_total = int(mismatch.sum().item())

    print("label invariance mismatch per task:", mismatch_per_task)
    if mismatch_total != 0:
        bad_nodes = torch.where(mismatch.any(dim=1))[0][:20].tolist()
        raise AssertionError(
            "Option C failed: global labels differ from local labels. "
            f"mismatch_total={mismatch_total}, first_bad_nodes={bad_nodes}"
        )

    meta = {
        "split": split,
        "total_nodes": int(total_nodes),
        "intra_edges_raw": int(sum(e.size(1) for e in global_edge_parts)),
        "bridge_edges_raw": int(bridge_edge_index.size(1)),
        "bridge_edge_fraction_raw": float(
            bridge_edge_index.size(1) / max(int(edge_index.size(1)), 1)
        ),
        "node_ranges": [(int(a), int(b)) for a, b in node_ranges],
        "bridge_records": bridge_records,
        "bridge_mode": cfg.bridge_mode,
        "label_semantics": cfg.label_semantics,
        "label_invariance_mismatch_total": mismatch_total,
        "label_invariance_mismatch_per_task": mismatch_per_task,
    }

    return (
        global_g,
        global_motifs,
        component_records,
        meta,
        community_graphs,
        community_motifs,
    )


# =============================================================================
# statistics and audits
# =============================================================================

def graph_stats(g: GraphData) -> Dict[str, object]:
    edge_index = g.edge_index.detach().cpu()
    n = int(g.num_nodes)
    e_raw = int(edge_index.size(1))

    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    out_deg = np.bincount(src, minlength=n).astype(np.int64)
    in_deg = np.bincount(dst, minlength=n).astype(np.int64)
    unique_edges = {(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())}
    e_unique = len(unique_edges)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(unique_edges)
    weak_components = list(nx.weakly_connected_components(graph))
    weak_sizes = [len(c) for c in weak_components]

    return {
        "num_nodes": int(n),
        "num_edges_raw": int(e_raw),
        "num_edges_unique": int(e_unique),
        "duplicate_edge_fraction": float(1.0 - e_unique / max(e_raw, 1)),
        "mean_in_degree_raw": float(in_deg.mean()) if len(in_deg) else 0.0,
        "mean_out_degree_raw": float(out_deg.mean()) if len(out_deg) else 0.0,
        "max_in_degree_raw": int(in_deg.max()) if len(in_deg) else 0,
        "max_out_degree_raw": int(out_deg.max()) if len(out_deg) else 0,
        "num_weak_cc": int(len(weak_sizes)),
        "largest_weak_cc": int(max(weak_sizes) if weak_sizes else 0),
        "largest_weak_cc_fraction": float(max(weak_sizes) / max(n, 1)) if weak_sizes else 0.0,
    }


def task_support_stats(g: GraphData, motifs: Dict[str, List]) -> Dict[str, object]:
    out = {}
    y = g.y.detach().cpu()
    n = int(g.num_nodes)

    for col, task in enumerate(TASKS):
        pos_nodes = int(y[:, col].sum().item())
        out[f"{task}_motif_count"] = int(len(motifs.get(task, [])))
        out[f"{task}_pos_nodes"] = int(pos_nodes)
        out[f"{task}_pos_rate"] = float(pos_nodes / max(n, 1))
        out[f"{task}_all_zero"] = int(pos_nodes == 0)
    return out


def community_support_rows(
    *,
    g: GraphData,
    split: str,
    node_ranges: Sequence[Tuple[int, int]],
    meta: Dict[str, object],
) -> List[Dict[str, object]]:
    rows = []
    edge_index = g.edge_index.detach().cpu()
    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    community_id = g.planted_community_id.detach().cpu().numpy().astype(np.int64)
    src_comm = community_id[src]
    dst_comm = community_id[dst]

    for cid, (start, end) in enumerate(node_ranges):
        nodes = torch.arange(start, end, dtype=torch.long)
        y = g.y[nodes].detach().cpu()

        intra_edges = int(((src_comm == cid) & (dst_comm == cid)).sum())
        boundary_edges = int(((src_comm == cid) ^ (dst_comm == cid)).sum())

        row = {
            "split": split,
            "community_id": int(cid),
            "node_start": int(start),
            "node_end_exclusive": int(end),
            "num_nodes": int(end - start),
            "intra_edges_raw": intra_edges,
            "boundary_edges_raw": boundary_edges,
        }

        for col, task in enumerate(TASKS):
            pos = int(y[:, col].sum().item())
            row[f"{task}_pos_nodes"] = pos
            row[f"{task}_pos_rate"] = float(pos / max(end - start, 1))

        rows.append(row)

    return rows


def print_global_audit(split: str, g: GraphData, motifs: Dict[str, List], meta: Dict[str, object]) -> None:
    print_header(f"GLOBAL BIG GRAPH AUDIT | split={split}")

    st = graph_stats(g)
    st.update(task_support_stats(g, motifs))
    st.update(
        {
            "intra_edges_raw": meta["intra_edges_raw"],
            "bridge_edges_raw": meta["bridge_edges_raw"],
            "bridge_edge_fraction_raw": meta["bridge_edge_fraction_raw"],
        }
    )

    for k, v in st.items():
        print(f"{k:35s}: {v}")

    assert int(st["num_nodes"]) == int(meta["total_nodes"])
    assert int(st["bridge_edges_raw"]) == int(meta["bridge_edges_raw"])
    assert g.x.size(0) == g.y.size(0) == g.num_nodes
    assert g.y.size(1) == len(TASKS), f"Expected {len(TASKS)} task columns, got {g.y.size(1)}"
    assert g.edge_index.size(1) == g.edge_attr.size(0), "edge_index / edge_attr mismatch"
    assert g.edge_attr.size(1) >= 4, "Expected edge_attr to contain timestamps + ports after add_ports()."

    for task in TASKS:
        assert int(st[f"{task}_pos_nodes"]) > 0, f"{split}: {task} has zero positives in global graph."

    print("PASS: global graph tensor shapes and label support are valid.")


def print_community_audit(split: str, rows: List[Dict[str, object]]) -> None:
    print_header(f"PER-COMMUNITY SUPPORT AUDIT | split={split}")
    df = pd.DataFrame(rows)
    show_cols = [
        "community_id",
        "num_nodes",
        "intra_edges_raw",
        "boundary_edges_raw",
    ]
    for task in TASKS:
        show_cols.extend([f"{task}_pos_nodes", f"{task}_pos_rate"])

    print(df[show_cols].to_string(index=False))

    for _, row in df.iterrows():
        cid = int(row["community_id"])
        for task in TASKS:
            pos = int(row[f"{task}_pos_nodes"])
            assert pos > 0, f"{split}: community {cid} has zero positives for {task}"

    print("PASS: every community has non-zero positive labels for every task.")


# =============================================================================
# saving global and community graphs
# =============================================================================

def make_community_subgraph(global_g: GraphData, start: int, end: int) -> GraphData:
    nodes = torch.arange(int(start), int(end), dtype=torch.long)

    # Use only timestamp columns and recompute ports inside this induced community graph.
    base_edge_attr = raw_timestamp_edge_attr(global_g)
    edge_index, edge_attr = subgraph(
        subset=nodes,
        edge_index=global_g.edge_index.detach().cpu(),
        edge_attr=base_edge_attr.detach().cpu(),
        relabel_nodes=True,
        num_nodes=int(global_g.num_nodes),
    )

    sg = GraphData(
        x=global_g.x[nodes].detach().cpu().clone(),
        y=global_g.y[nodes].detach().cpu().clone(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
        readout="node",
        num_nodes=int(len(nodes)),
    )
    sg = sg.add_ports()
    sg.original_node_start = int(start)
    sg.original_node_end_exclusive = int(end)
    return sg


def save_split_graphs_and_communities(
    *,
    split_graphs: Dict[str, GraphData],
    split_motifs: Dict[str, Dict[str, List]],
    split_meta: Dict[str, Dict[str, object]],
    split_community_graphs: Dict[str, Dict[int, GraphData]],
    split_community_motifs: Dict[str, Dict[int, Dict[str, List]]],
    global_out_dir: Path,
    community_root: Path,
    dataset_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_out_dir.mkdir(parents=True, exist_ok=True)
    community_dataset_root = community_root / dataset_id
    community_dataset_root.mkdir(parents=True, exist_ok=True)

    global_support_rows = []
    motif_rows = []
    community_support_all_rows = []
    base_client_rows = []

    for split in SPLITS:
        g = split_graphs[split]
        motifs = split_motifs[split]
        meta = split_meta[split]
        node_ranges = meta["node_ranges"]

        torch.save(g, global_out_dir / f"{split}.pt")

        motif_rows.append({"split": split, **{task: int(len(motifs[task])) for task in TASKS}})

        for task in TASKS:
            col = TASKS.index(task)
            pos = int(g.y[:, col].sum().item())
            global_support_rows.append(
                {
                    "split": split,
                    "task": task,
                    "motif_count": int(len(motifs[task])),
                    "pos_nodes": pos,
                    "pos_rate": float(pos / max(int(g.num_nodes), 1)),
                }
            )

        community_rows = community_support_rows(
            g=g,
            split=split,
            node_ranges=node_ranges,
            meta=meta,
        )
        community_support_all_rows.extend(community_rows)

        for cid, (start, end) in enumerate(node_ranges):
            comm_dir = community_dataset_root / f"community_{cid}"
            comm_dir.mkdir(parents=True, exist_ok=True)
            sg = split_community_graphs[split][int(cid)]
            torch.save(sg, comm_dir / f"{split}.pt")

    # Write global CSVs.
    global_support_df = pd.DataFrame(global_support_rows)
    motif_df = pd.DataFrame(motif_rows)
    community_support_df = pd.DataFrame(community_support_all_rows)

    global_support_df.to_csv(global_out_dir / "label_support.csv", index=False)
    motif_df.to_csv(global_out_dir / "motif_counts.csv", index=False)
    community_support_df.to_csv(global_out_dir / "community_support.csv", index=False)

    # Write community-level support CSVs and base client registry.
    for cid in sorted(community_support_df["community_id"].unique()):
        comm_dir = community_dataset_root / f"community_{cid}"
        sub = community_support_df[community_support_df["community_id"] == cid].copy()
        sub.to_csv(comm_dir / "community_support.csv", index=False)

        # Label support in the same split/task style as existing graph folders.
        label_rows = []
        for split in SPLITS:
            split_sub = sub[sub["split"] == split].iloc[0]
            for task in TASKS:
                label_rows.append(
                    {
                        "split": split,
                        "task": task,
                        "motif_count": int(
                            len(split_community_motifs[split][int(cid)].get(task, []))
                        ),
                        "pos_nodes": int(split_sub[f"{task}_pos_nodes"]),
                        "pos_rate": float(split_sub[f"{task}_pos_rate"]),
                    }
                )
        pd.DataFrame(label_rows).to_csv(comm_dir / "label_support.csv", index=False)

        # Base client registry row with train/val/test support columns.
        row = {
            "graph_id": int(BASE_GRAPH_ID_START + cid),
            "dataset_id": f"{dataset_id}_community{cid}",
            "data_dir": str(comm_dir),
            "n": int(sub[sub["split"] == "train"].iloc[0]["num_nodes"]),
            "type": "planted_chordal_community",
            "planted_community_id": int(cid),
            "base_dataset_id": dataset_id,
            "global_dataset_id": dataset_id,
            "global_data_dir": str(global_out_dir),
            "global_num_nodes": int(split_graphs["train"].num_nodes),
            "global_num_edges_raw": int(split_graphs["train"].edge_index.size(1)),
            "bridge_mode": BRIDGE_MODE,
            "label_semantics": LABEL_SEMANTICS,
        }

        for split in SPLITS:
            split_sub = sub[sub["split"] == split].iloc[0]
            row[f"{split}_num_nodes"] = int(split_sub["num_nodes"])
            row[f"{split}_num_edges_raw"] = int(split_sub["intra_edges_raw"])
            row[f"{split}_boundary_edges_raw"] = int(split_sub["boundary_edges_raw"])

            for task in TASKS:
                row[f"{split}_{task}_pos_nodes"] = int(split_sub[f"{task}_pos_nodes"])
                row[f"{split}_{task}_pos_rate"] = float(split_sub[f"{task}_pos_rate"])

        base_client_rows.append(row)

    base_client_df = pd.DataFrame(base_client_rows).sort_values("graph_id")
    base_client_df.to_csv(community_dataset_root / "community_base_clients.csv", index=False)

    return global_support_df, community_support_df, base_client_df


# =============================================================================
# q virtual clients
# =============================================================================

def make_q_virtual_clients(
    *,
    base_client_df: pd.DataFrame,
    q_values: Sequence[float],
    out_dir: Path,
    dataset_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    virtual_rows = []
    next_gid = VIRTUAL_GRAPH_ID_START

    for q in q_values:
        validate_q(q)
        off = q_other_share(q)

        for _, base in base_client_df.sort_values("planted_community_id").iterrows():
            cid = int(base["planted_community_id"])
            base_gid = int(base["graph_id"])
            base_did = str(base["dataset_id"])
            data_dir = str(base["data_dir"])

            for assigned_task in TASKS:
                row = base.copy().to_dict()
                row["graph_id"] = int(next_gid)
                row["dataset_id"] = f"{base_did}_q{q_tag(q)}_{assigned_task}"
                row["data_dir"] = data_dir

                row["base_graph_id"] = int(base_gid)
                row["base_dataset_id"] = base_did
                row["base_rank"] = int(cid)
                row["planted_community_id"] = int(cid)
                row["structural_cluster_id"] = int(cid)

                row["mask_mode"] = MASK_MODE
                row["assigned_task"] = assigned_task
                row["mask_task"] = assigned_task
                row["masked_tasks_json"] = safe_json([t for t in TASKS if t != assigned_task])

                row["q_value"] = float(q)
                row["q_iid"] = float(q_iid_value())
                row["q_other_share"] = float(off)
                row["q_assigned_share"] = float(q)
                row["q_allocation_mode"] = Q_ALLOCATION_MODE
                row["mask_fraction"] = float(q)
                row["mask_seed"] = int(next_gid)
                row["mask_apply_split"] = MASK_APPLY_SPLIT
                row["designed_heterogeneity"] = float(q)
                row["controlled_benchmark"] = CONTROLLED_BENCHMARK

                row["target_family"] = f"q_task_{assigned_task}"
                row["community_task_family"] = f"community{cid}_{assigned_task}"

                # Preserve true train support and overwrite train support with expected visible support.
                train_num_nodes = int(base["train_num_nodes"])

                for task in TASKS:
                    true_rate = float(base[f"train_{task}_pos_rate"])
                    true_nodes = int(base[f"train_{task}_pos_nodes"])
                    share = float(q if task == assigned_task else off)

                    # Use the exact same integer allocation convention as the real
                    # q label-mask loader. This avoids independent-rounding errors
                    # where visible metadata could sum to true_nodes +/- a few.
                    counts_for_label_task = q_counts_for_label_task(
                        true_nodes,
                        label_task=task,
                        q_value=float(q),
                    )
                    assigned_idx = TASKS.index(assigned_task)
                    visible_nodes = int(counts_for_label_task[assigned_idx])
                    visible_rate = float(visible_nodes / max(train_num_nodes, 1))

                    row[f"true_train_{task}_pos_rate"] = true_rate
                    row[f"true_train_{task}_pos_nodes"] = true_nodes
                    row[f"visible_train_{task}_pos_rate"] = visible_rate
                    row[f"visible_train_{task}_pos_nodes"] = visible_nodes
                    row[f"q_share_{task}"] = share

                    # Keep compatibility with existing heterogeneity utilities.
                    row[f"train_{task}_pos_rate"] = visible_rate
                    row[f"train_{task}_pos_nodes"] = visible_nodes
                    row[f"p_{task}"] = visible_rate

                p = np.asarray([row[f"p_{task}"] for task in TASKS], dtype=np.float64)
                row["p_sum"] = float(p.sum())
                row["p_mean"] = float(p.mean())
                row["p_min"] = float(p.min())
                row["p_max"] = float(p.max())
                row["p_spread"] = float(p.max() - p.min())
                row["dominant_task"] = TASKS[int(np.argmax(p))]

                virtual_rows.append(row)
                next_gid += 1

    virtual_df = pd.DataFrame(virtual_rows).reset_index(drop=True)

    manifest_rows = []
    for q, sub in virtual_df.groupby("q_value", sort=True):
        sub = sub.sort_values(["planted_community_id", "assigned_task", "graph_id"]).copy()

        graph_ids = [int(x) for x in sub["graph_id"].tolist()]
        dataset_ids = [str(x) for x in sub["dataset_id"].tolist()]
        subset_clients = "|".join(str(x) for x in graph_ids)

        membership = []
        graph_to_task = {}
        graph_to_community = {}
        family_to_graph_ids: Dict[str, List[int]] = {}
        community_to_graph_ids: Dict[str, List[int]] = {}

        for _, row in sub.iterrows():
            gid = int(row["graph_id"])
            task = str(row["assigned_task"])
            cid = int(row["planted_community_id"])
            family = f"q_task_{task}"

            graph_to_task[str(gid)] = task
            graph_to_community[str(gid)] = cid
            family_to_graph_ids.setdefault(family, []).append(gid)
            community_to_graph_ids.setdefault(str(cid), []).append(gid)

            membership.append(
                {
                    "graph_id": gid,
                    "dataset_id": str(row["dataset_id"]),
                    "base_graph_id": int(row["base_graph_id"]),
                    "base_dataset_id": str(row["base_dataset_id"]),
                    "planted_community_id": cid,
                    "assigned_task": task,
                    "mask_mode": MASK_MODE,
                    "q_value": float(q),
                    "q_assigned_share": float(q),
                    "q_other_share": float(q_other_share(q)),
                }
            )

        row = {
            "family": "planted_community_q_task_label_heterogeneity",
            "subset_id": f"{dataset_id}_q{q_tag(q)}_{len(graph_ids)}clients",
            "subset_size": int(len(graph_ids)),
            "subset_clients": subset_clients,
            "q_value": float(q),
            "q_iid": float(q_iid_value()),
            "q_other_share": float(q_other_share(q)),
            "q_assigned_share": float(q),
            "mask_fraction": float(q),
            "mask_mode": MASK_MODE,
            "mask_apply_split": MASK_APPLY_SPLIT,
            "controlled_benchmark": CONTROLLED_BENCHMARK,
            "num_structural_clusters": int(sub["planted_community_id"].nunique()),
            "clients_per_structural_cluster": int(len(TASKS)),
            "global_dataset_id": str(sub.iloc[0]["global_dataset_id"]),
            "global_data_dir": str(sub.iloc[0]["global_data_dir"]),
            "global_num_nodes": int(sub.iloc[0]["global_num_nodes"]),
            "global_num_edges_raw": int(sub.iloc[0]["global_num_edges_raw"]),
            "bridge_mode": str(sub.iloc[0]["bridge_mode"]),
            "label_semantics": str(sub.iloc[0]["label_semantics"]),
            "centralized_scope": "global",
            "base_graph_ids_json": safe_json(
                sorted({int(x) for x in sub["base_graph_id"].tolist()})
            ),
            "base_dataset_ids_json": safe_json(
                sorted({str(x) for x in sub["base_dataset_id"].tolist()})
            ),
            "graph_ids_json": safe_json(graph_ids),
            "dataset_ids_json": safe_json(dataset_ids),
            "membership_json": safe_json(membership),
            "graph_to_task_json": safe_json(graph_to_task),
            "graph_to_community_json": safe_json(graph_to_community),
            "family_to_graph_ids_json": safe_json(family_to_graph_ids),
            "community_to_graph_ids_json": safe_json(community_to_graph_ids),
            "family_counts_json": safe_json(
                sub["target_family"].value_counts().sort_index().to_dict()
            ),
        }

        for task in TASKS:
            row[f"{task}_pos_rate_std_across_clients"] = float(sub[f"p_{task}"].std(ddof=0))

        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows).sort_values("q_value").reset_index(drop=True)

    virtual_df.to_csv(out_dir / "cluster_generation_parameters.csv", index=False)
    manifest_df.to_csv(out_dir / "selected_subset.csv", index=False)

    return virtual_df, manifest_df


def audit_q_virtual_clients(virtual_df: pd.DataFrame, q_values: Sequence[float]) -> None:
    print_header("Q-VIRTUAL CLIENT REGISTRY AUDIT")

    expected = len(q_values) * len(TASKS)
    for q in q_values:
        sub_q = virtual_df[virtual_df["q_value"].astype(float) == float(q)]
        print_subheader(f"q={q:.6f} | rows={len(sub_q)}")
        print(
            sub_q[
                [
                    "graph_id",
                    "planted_community_id",
                    "assigned_task",
                    "data_dir",
                ]
            ].head(20).to_string(index=False)
        )

        for cid, group in sub_q.groupby("planted_community_id"):
            assert len(group) == len(TASKS), (
                f"q={q}: community {cid} should have {len(TASKS)} task-specialist clients, got {len(group)}"
            )
            assert set(group["assigned_task"]) == set(TASKS)

            print(f"\nq={q:.3f} community={cid} visible train positive-node matrix")
            mat = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)
            share = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)

            group = group.set_index("assigned_task").loc[TASKS]
            for assigned in TASKS:
                for task in TASKS:
                    mat.loc[assigned, task] = int(group.loc[assigned, f"train_{task}_pos_nodes"])
                    share.loc[assigned, task] = float(group.loc[assigned, f"q_share_{task}"])

            print("share matrix:")
            print(share.to_string())
            print("visible node matrix:")
            print(mat.to_string())

    print("\nPASS: q virtual client registry has the expected task-specialist structure.")


# =============================================================================
# main
# =============================================================================

def build_dataset_id(args, ds: List[float], deltas: List[float]) -> str:
    def f(x: float) -> str:
        return str(x).replace(".", "p")
    d_tag = "_".join(f(x) for x in ds)
    delta_tag = "_".join(f(x) for x in deltas)
    return (
        f"planted_chordal_c{len(ds)}_n{args.nodes_per_community}_"
        f"d{d_tag}_delta{delta_tag}_bridge{args.bridge_edges_per_pair}_seed{args.seed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-per-community", type=int, default=2000)
    parser.add_argument("--community-ds", type=str, default="3,3,3.5,3.5")
    parser.add_argument("--community-deltas", type=str, default="8,12,8,12")
    parser.add_argument("--bridge-edges-per-pair", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generator", type=str, default="chordal")
    parser.add_argument("--q-values", type=str, default="0.8")
    parser.add_argument("--global-out-root", type=str, default="./andrea/planted_community_data")
    parser.add_argument("--community-client-root", type=str, default="./andrea/planted_community_clients")
    parser.add_argument("--q-manifest-dir", type=str, default="./andrea/clustering_planted_community_q")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bridge-mode", choices=["forward_dag"], default="forward_dag")
    parser.add_argument("--no-q-clients", action="store_true")
    args = parser.parse_args()

    ds = parse_float_list(args.community_ds, "community-ds")
    deltas = parse_float_list(args.community_deltas, "community-deltas")
    q_values = parse_q_values(args.q_values)
    if args.bridge_mode != BRIDGE_MODE:
        raise ValueError(f"Only label-invariant bridge mode {BRIDGE_MODE!r} is supported.")

    if len(ds) != len(deltas):
        raise ValueError(
            f"--community-ds and --community-deltas must have same length; got {len(ds)} and {len(deltas)}"
        )

    total_nodes = int(args.nodes_per_community * len(ds))
    dataset_id = build_dataset_id(args, ds, deltas)

    global_out_dir = Path(args.global_out_root) / dataset_id
    community_root = Path(args.community_client_root)
    q_manifest_dir = Path(args.q_manifest_dir)

    cfg = GenerationConfig(
        dataset_id=dataset_id,
        nodes_per_community=int(args.nodes_per_community),
        community_ds=ds,
        community_deltas=deltas,
        bridge_edges_per_pair=int(args.bridge_edges_per_pair),
        seed=int(args.seed),
        generator=str(args.generator),
        q_values=q_values,
        global_out_dir=str(global_out_dir),
        community_client_root=str(community_root),
        q_manifest_dir=str(q_manifest_dir),
        overwrite=bool(args.overwrite),
        max_time=int(total_nodes),
        bridge_mode=str(args.bridge_mode),
        label_semantics=LABEL_SEMANTICS,
    )

    print_header("PLANTED COMMUNITY GRAPH DATA GENERATION CONFIG")
    print(json.dumps(asdict(cfg), indent=2))

    ensure_empty_or_overwrite(global_out_dir, args.overwrite)

    community_dataset_root = community_root / dataset_id
    ensure_empty_or_overwrite(community_dataset_root, args.overwrite)

    if not args.no_q_clients:
        ensure_empty_or_overwrite(q_manifest_dir, args.overwrite)

    split_graphs: Dict[str, GraphData] = {}
    split_motifs: Dict[str, Dict[str, List]] = {}
    split_meta: Dict[str, Dict[str, object]] = {}
    split_community_graphs: Dict[str, Dict[int, GraphData]] = {}
    split_community_motifs: Dict[str, Dict[int, Dict[str, List]]] = {}
    component_records_all: List[Dict] = []

    for split in SPLITS:
        (
            g,
            motifs,
            component_records,
            meta,
            community_graphs,
            community_motifs,
        ) = build_planted_big_graph_for_split(split=split, cfg=cfg)
        split_graphs[split] = g
        split_motifs[split] = motifs
        split_meta[split] = meta
        split_community_graphs[split] = community_graphs
        split_community_motifs[split] = community_motifs
        component_records_all.extend(component_records)

        print_global_audit(split, g, motifs, meta)
        comm_rows = community_support_rows(g=g, split=split, node_ranges=meta["node_ranges"], meta=meta)
        print_community_audit(split, comm_rows)

    global_support_df, community_support_df, base_client_df = save_split_graphs_and_communities(
        split_graphs=split_graphs,
        split_motifs=split_motifs,
        split_meta=split_meta,
        split_community_graphs=split_community_graphs,
        split_community_motifs=split_community_motifs,
        global_out_dir=global_out_dir,
        community_root=community_root,
        dataset_id=dataset_id,
    )

    # Write config and component records after successful save.
    with open(global_out_dir / "generation_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    pd.DataFrame(component_records_all).to_csv(global_out_dir / "component_generation_records.csv", index=False)
    base_client_df.to_csv(global_out_dir / "community_base_clients.csv", index=False)

    print_header("SAVED GLOBAL AND COMMUNITY DATA")
    print(f"global graph folder        : {global_out_dir}")
    print(f"community client root      : {community_root / dataset_id}")
    print(f"global train/val/test      : {[str(global_out_dir / (s + '.pt')) for s in SPLITS]}")
    print(f"base community client rows : {len(base_client_df)}")
    print(base_client_df[["graph_id", "dataset_id", "data_dir", "planted_community_id"]].to_string(index=False))

    if not args.no_q_clients:
        virtual_df, manifest_df = make_q_virtual_clients(
            base_client_df=base_client_df,
            q_values=q_values,
            out_dir=q_manifest_dir,
            dataset_id=dataset_id,
        )
        audit_q_virtual_clients(virtual_df, q_values)

        print_header("SAVED Q-MASKED VIRTUAL CLIENT REGISTRY")
        print(f"q registry folder          : {q_manifest_dir}")
        print(f"cluster_generation CSV     : {q_manifest_dir / 'cluster_generation_parameters.csv'}")
        print(f"selected_subset CSV        : {q_manifest_dir / 'selected_subset.csv'}")
        print(f"virtual client rows        : {len(virtual_df)}")
        print(f"selected subset rows       : {len(manifest_df)}")
        print(manifest_df[["subset_id", "q_value", "subset_size", "num_structural_clusters"]].to_string(index=False))

    print_header("FINAL SUCCESS")
    print("Data generation finished with exact local/global label invariance.")
    print("Next sanity check command:")
    print(f"  ls -lh {global_out_dir}")
    print(f"  find {community_root / dataset_id} -maxdepth 2 -type f | sort | head -40")
    if not args.no_q_clients:
        print(f"  head -5 {q_manifest_dir / 'selected_subset.csv'}")
        print(f"  head -5 {q_manifest_dir / 'cluster_generation_parameters.csv'}")


if __name__ == "__main__":
    main()
