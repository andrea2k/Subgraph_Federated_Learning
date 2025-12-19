import os
import csv
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch
from torch_geometric.utils import subgraph

from utils.gcn_utils import GraphData

from scripts.data.simulator import (
    to_lists,
    deg_in, deg_out, fan_in, fan_out
)


# --- Witness-returning motif detectors ---

def _canonical_cycle(cyc: Tuple[int, ...]) -> Tuple[int, ...]:
    """Canonicalize by rotation so smallest node id is first (direction preserved)."""
    m = min(cyc)
    mi = cyc.index(m)
    return cyc[mi:] + cyc[:mi]


def Cn_check_with_witness(k: int, max_instances: Optional[int] = None):
    """
    Detects directed cycles of length k, labels all nodes that participate 
    in at least one such cycle, and records the exact node tuples (witnesses) 
    for each cycle instance it finds.

    Returns a function f(data)->(labels, witnesses)
      labels: [N,1] float 0/1 if node participates in at least one k-cycle
      witnesses: list of k-tuples (each tuple is a directed cycle instance)
    """
    assert 2 <= k <= 6

    def _fn(data: "GraphData"):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)

        out_sets = [set(g_out[i].tolist()) for i in range(num_nodes)]

        labels = torch.zeros((num_nodes, 1), dtype=torch.float32)
        witnesses: List[Tuple[int, ...]] = []
        seen = set()

        # bounded DFS from each start node
        for start in range(num_nodes):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()

                if len(path) == k:
                    # close cycle: last -> start
                    if start in out_sets[node]:
                        cyc = tuple(path)
                        canon = _canonical_cycle(cyc)
                        if canon not in seen:
                            seen.add(canon)
                            witnesses.append(canon)
                            for u in canon:
                                labels[u, 0] = 1.0
                            if max_instances is not None and len(witnesses) >= max_instances:
                                return labels, witnesses
                    continue

                for nxt in out_sets[node]:
                    if nxt in path:
                        continue
                    if nxt == start and len(path) + 1 < k:
                        continue
                    stack.append((nxt, path + [nxt]))

        return labels, witnesses

    return _fn


def SG2_check_with_witness(max_instances: Optional[int] = None):
    """
    Witness-returning version of SG2_check.

    Returns:
      labels: [N,1] 1 if node i is a 'gather' node under SG2 logic
      witnesses: list of 4-tuples (s, j1, j2, i)
        where i is gather, j1/j2 are two distinct predecessors of i,
        and s is a shared predecessor (source) of both j1 and j2 (excluding i).
    """
    def _fn(data: "GraphData"):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)

        labels = torch.zeros((num_nodes, 1), dtype=torch.float32)
        witnesses: List[Tuple[int, int, int, int]] = []

        for i in range(num_nodes):
            preds = g_in[i].unique().tolist()
            if len(preds) < 2:
                continue

            src_to_mids: Dict[int, List[int]] = defaultdict(list)
            for j in preds:
                if j == i:
                    continue
                pj = g_in[j].tolist()
                for s in pj:
                    if s == i:
                        continue
                    src_to_mids[int(s)].append(int(j))

            # pick one witness per gather node i 
            found = False
            for s, mids in src_to_mids.items():
                mids_u = list(dict.fromkeys(mids))  # unique preserve order
                if len(mids_u) >= 2:
                    j1, j2 = mids_u[0], mids_u[1]
                    labels[i, 0] = 1.0
                    witnesses.append((int(s), int(j1), int(j2), int(i)))
                    found = True
                    if max_instances is not None and len(witnesses) >= max_instances:
                        return labels, witnesses
                    break
            if found:
                continue

        return labels, witnesses

    return _fn


def BP2_check_with_witness(max_instances: Optional[int] = None):
    """
    Witness-returning version of BP2_check.

    Returns:
      labels: [N,1] 1 if node i satisfies BP2 logic
      witnesses: list of 4-tuples (l1, l2, r, i)
        where i is the "right-side" gather-like node,
        l1,l2 are two distinct predecessors of i,
        and r is a node that both l1 and l2 point to (excluding i).
    """
    def _fn(data: "GraphData"):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        g_in, g_out = to_lists(edge_index, num_nodes)

        labels = torch.zeros((num_nodes, 1), dtype=torch.float32)
        witnesses: List[Tuple[int, int, int, int]] = []

        for i in range(num_nodes):
            lefts = g_in[i].unique().tolist()
            if len(lefts) < 2:
                continue

            right_to_lefts: Dict[int, List[int]] = defaultdict(list)
            for l in lefts:
                outs = g_out[int(l)].tolist()
                for r in outs:
                    if int(r) == i:
                        continue
                    right_to_lefts[int(r)].append(int(l))

            # one witness per i
            found = False
            for r, ls in right_to_lefts.items():
                ls_u = list(dict.fromkeys(ls))
                if len(ls_u) >= 2:
                    l1, l2 = ls_u[0], ls_u[1]
                    labels[i, 0] = 1.0
                    witnesses.append((int(l1), int(l2), int(r), int(i)))
                    found = True
                    if max_instances is not None and len(witnesses) >= max_instances:
                        return labels, witnesses
                    break

            if found:
                continue

        return labels, witnesses

    return _fn


# --- New labeling approach that builds y (inherits the existing approach) and collects witnesses --- 
# By collecting witnesses for the motif-based subtasks (cycles/SG/BP), we know which specific nodes formed each motif instance.

def define_subtasks_thresholds_and_witness_builders(
    cycle_max_instances_per_k: Optional[int] = None,
    sg_max_instances: Optional[int] = None,
    bp_max_instances: Optional[int] = None,
):
    """
    Same task ordering with define_subtasks_and_thresholds() method of Beni, 
    but for motif tasks we use witness-returning builders.
    For degree/fan tasks we reuse the existing functions and thresholds.
    """
    # Regular label-only tasks
    label_only_funcs = [
        deg_in, deg_out, fan_in, fan_out
    ]
    label_only_thresholds = [3, 3, 3, 3]
    label_only_names = ["deg_in>3", "deg_out>3", "fan_in>3", "fan_out>3"]

    # Witness-returning motif tasks (cycles 2..6, SG2, BP2)
    motif_builders = [
        ("cycle2", Cn_check_with_witness(2, max_instances=cycle_max_instances_per_k)),
        ("cycle3", Cn_check_with_witness(3, max_instances=cycle_max_instances_per_k)),
        ("cycle4", Cn_check_with_witness(4, max_instances=cycle_max_instances_per_k)),
        ("cycle5", Cn_check_with_witness(5, max_instances=cycle_max_instances_per_k)),
        ("cycle6", Cn_check_with_witness(6, max_instances=cycle_max_instances_per_k)),
        ("scatter_gather", SG2_check_with_witness(max_instances=sg_max_instances)),
        ("biclique", BP2_check_with_witness(max_instances=bp_max_instances)),
    ]

    names = label_only_names + [n for (n, _) in motif_builders]
    thresholds = label_only_thresholds + [None] * len(motif_builders)
    return label_only_funcs, label_only_thresholds, motif_builders, names, thresholds


def set_y_with_labels_and_witnesses(
    data: "GraphData",
    label_only_funcs,
    label_only_thresholds,
    motif_builders,
):
    """
    Produces:
      data.y: [N, 11] float 0/1
      witnesses: dict with keys:
        "cycle2".."cycle6" -> List[Tuple[int,...]]
        "scatter_gather" -> List[Tuple[int,int,int,int]]
        "biclique" -> List[Tuple[int,int,int,int]]
    """
    ys = []
    witnesses: Dict[str, List[Tuple[int, ...]]] = {}

    # label-only tasks
    for f, th in zip(label_only_funcs, label_only_thresholds):
        y = f(data, th)  
        ys.append(y.float())

    # motif tasks (label + witnesses)
    for name, builder in motif_builders:
        y, w = builder(data)
        ys.append(y.float())
        witnesses[name] = w

    data.y = torch.cat(ys, dim=1)
    return data, witnesses


# --- Federated assignment based on witnesses and building client graphs ---

def assign_clients_from_witnesses(
    num_nodes: int,
    witnesses: Dict[str, List[Tuple[int, ...]]],
    num_clients: int,
    seed: int,
):
    """
    One client per node. Priority:
      cycles2..6 -> scatter_gather -> biclique -> fill remaining nodes to balance client sizes
    Constraint:
      within each witness instance, try to assign each node to a different client (best-effort).
    Conflict:
      first assignment wins; later instances do not move already assigned nodes.
    """
    assert num_clients >= 2
    rng = random.Random(seed)

    node_to_client = torch.full((num_nodes,), -1, dtype=torch.long)
    sizes = [0] * num_clients

    def _pick_distinct_clients(used: set, m: int) -> List[int]:
        available = [c for c in range(num_clients) if c not in used]
        rng.shuffle(available)
        if len(available) >= m:
            return available[:m]
        extra = []
        while len(extra) < (m - len(available)):
            extra.append(rng.randrange(num_clients))
        return available + extra

    def _assign_instance(nodes: List[int]):
        used = set()
        for u in nodes:
            cu = int(node_to_client[u].item())
            if cu != -1:
                used.add(cu)

        unassigned = [u for u in nodes if node_to_client[u].item() == -1]
        choices = _pick_distinct_clients(used, len(unassigned))

        for u, c in zip(unassigned, choices):
            node_to_client[u] = c
            sizes[c] += 1

    # Priority ordering
    for k in ["cycle2", "cycle3", "cycle4", "cycle5", "cycle6"]:
        for cyc in witnesses.get(k, []):
            _assign_instance(list(map(int, cyc)))

    for inst in witnesses.get("scatter_gather", []):
        _assign_instance(list(map(int, inst)))

    for inst in witnesses.get("biclique", []):
        _assign_instance(list(map(int, inst)))

    # Fill remaining nodes to balance client sizes
    remaining = [i for i in range(num_nodes) if node_to_client[i].item() == -1]
    rng.shuffle(remaining)
    for u in remaining:
        c = int(min(range(num_clients), key=lambda x: sizes[x]))
        node_to_client[u] = c
        sizes[c] += 1

    return node_to_client


def build_client_graphs(
    data: "GraphData",
    node_to_client: torch.Tensor,
    relabel_nodes: bool = True,
):
    """
    Returns list of GraphData, one per client, induced by that client's nodes.
    """
    num_clients = int(node_to_client.max().item()) + 1
    client_graphs: List["GraphData"] = []

    for c in range(num_clients):
        node_idx = torch.where(node_to_client == c)[0]
        if node_idx.numel() == 0:
            # here we skip empty clients
            continue

        x = data.x[node_idx].clone()
        y = data.y[node_idx].clone()

        eidx, eattr = subgraph(
            node_idx,
            data.edge_index,
            data.edge_attr,
            relabel_nodes=relabel_nodes,
            num_nodes=data.num_nodes,
        )

        gd = GraphData(
            x=x,
            y=y,
            edge_index=eidx,
            edge_attr=eattr,
            readout=data.readout,
        )
        client_graphs.append(gd)

    return client_graphs


def save_federated_clients(
    split_dir: str,
    global_data: "GraphData",
    node_to_client: torch.Tensor,
):
    os.makedirs(split_dir, exist_ok=True)

    # mapping
    torch.save(node_to_client.cpu(), os.path.join(split_dir, "node_to_client.pt"))

    # per-client graphs
    clients_out = os.path.join(split_dir, "clients")
    os.makedirs(clients_out, exist_ok=True)

    num_clients = int(node_to_client.max().item()) + 1

    client_stats = []  # list of (client_id, num_nodes, num_edges)

    for c in range(num_clients):
        node_idx = torch.where(node_to_client == c)[0]
        num_nodes = int(node_idx.numel())
        if num_nodes == 0:
            client_stats.append((c, 0, 0))
            continue

        # induced subgraph for this client's nodes
        eidx, eattr = subgraph(
            node_idx,
            global_data.edge_index,
            global_data.edge_attr,
            relabel_nodes=True,
            num_nodes=global_data.num_nodes,
        )
        num_edges = int(eidx.size(1))

        gd = GraphData(
            x=global_data.x[node_idx].clone(),
            y=global_data.y[node_idx].clone(),
            edge_index=eidx,
            edge_attr=eattr,
            readout=global_data.readout,
        )
        torch.save(gd, os.path.join(clients_out, f"client_{c:04d}.pt"))

        client_stats.append((c, num_nodes, num_edges))

    # sizes
    with open(os.path.join(split_dir, "client_sizes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client_id", "num_nodes", "num_edges"])
        w.writerows(client_stats)
