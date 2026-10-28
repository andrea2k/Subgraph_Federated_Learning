"""Cycle-task definitions and node labels computed from motif witnesses."""

import torch
from fedptha.data.motifs import (
    build_in_out,
    cycles_C2,
    cycles_C3,
    cycles_C4,
    cycles_C5,
    cycles_C6,
)

TASK_FUNCS = {
    "cycle2": lambda adj: cycles_C2(adj["out_set"], adj["in_set"]),
    "cycle3": lambda adj: cycles_C3(adj["out_set"], adj["in_set"]),
    "cycle4": lambda adj: cycles_C4(adj["out_set"], adj["in_set"]),
    "cycle5": lambda adj: cycles_C5(adj["out_set"], adj["in_set"]),
    "cycle6": lambda adj: cycles_C6(adj["out_set"], adj["in_set"]),
}

TASKS = list(TASK_FUNCS.keys())

SPLITS = ["train", "val", "test"]


def nodes_to_label_from_witness(task_name, w):
    # degree/fan tasks are singleton witnesses: (node,)
    if task_name in {"deg_in_three", "deg_out_three", "fan_in_three", "fan_out_three"}:
        return w

    # cycle tasks: every node on the cycle is positive
    if task_name.startswith("cycle"):
        return w

    # scatter-gather: only the sink node is positive
    # witness format: (source, j1, j2, sink)
    if task_name == "scatter_gather":
        return (w[-1],)

    # biclique: both sink-side/right-side nodes are positive
    # witness format: (l1, l2, r1, r2)
    if task_name == "biclique":
        return (w[2], w[3])

    raise ValueError(f"Unknown task name: {task_name}")


def set_y_and_get_motifs(g, task_funcs=TASK_FUNCS):
    edge_index = g.edge_index
    num_nodes = int(g.num_nodes)

    out_list, in_list, out_set, in_set = build_in_out(edge_index, num_nodes)

    adj = {
        "out_list": out_list,
        "in_list": in_list,
        "out_set": out_set,
        "in_set": in_set,
    }

    tasks = list(task_funcs.keys())
    y = torch.zeros((num_nodes, len(tasks)), dtype=torch.float32)

    motifs_tuple = {}

    for col, task_name in enumerate(tasks):
        motifs = task_funcs[task_name](adj)
        motifs_tuple[task_name] = motifs

        for w in motifs:
            for u in nodes_to_label_from_witness(task_name, w):
                y[int(u), col] = 1.0

    g.y = y
    g.num_classes = y.shape[1]
    return g, motifs_tuple
