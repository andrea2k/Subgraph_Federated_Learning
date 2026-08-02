import os
import csv
from itertools import product
import pandas as pd
import torch
import numpy as np

from utils.seed import set_seed, derive_seed
from scripts.data.simulator import GraphSimulator

from utils.witness_funcs import (
    build_in_out,
    fan_in_three,
    fan_out_three,
    deg_in_three,
    deg_out_three,
    cycles_C2,
    cycles_C3,
    cycles_C4,
    cycles_C5,
    cycles_C6,
    SG2,
    BP2,
)

BASE_SEED = 0

DATA_ROOT = "./andrea/cluster_data"
GRAPH_PARAM_CSV = "./andrea/clustering/cluster_generation_parameters.csv"

# DATA_ROOT = "./andrea/cluster_data_benchmarkA"
# GRAPH_PARAM_CSV = "./andrea/clustering_benchmarkA/cluster_generation_parameters.csv"

N_POOL = [3800, 3900, 4000, 4100, 4200]
D_POOL = [
    3,
    4,
    5,
]
R_POOL = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    10.5,
    11.0,
    11.5,
]
GEN = ["chordal"]

REPEATS_PER_CELL = 4

# BASE_SEED = 0
# DATA_ROOT = "./andrea/big_graph_data"
# GRAPH_PARAM_CSV = "./andrea/big_graph_generation_parameters.csv"

# N_POOL = [8192]
# D_POOL = [6]
# R_POOL = [11.1]
# GEN = ["chordal"]

TASK_FUNCS = {
    # "deg_in_three": lambda adj: deg_in_three(adj["out_list"], adj["in_list"]),
    # "deg_out_three": lambda adj: deg_out_three(adj["out_list"], adj["in_list"]),
    # "fan_in_three": lambda adj: fan_in_three(adj["out_set"], adj["in_set"]),
    # "fan_out_three": lambda adj: fan_out_three(adj["out_set"], adj["in_set"]),
    "cycle2": lambda adj: cycles_C2(adj["out_set"], adj["in_set"]),
    "cycle3": lambda adj: cycles_C3(adj["out_set"], adj["in_set"]),
    "cycle4": lambda adj: cycles_C4(adj["out_set"], adj["in_set"]),
    "cycle5": lambda adj: cycles_C5(adj["out_set"], adj["in_set"]),
    "cycle6": lambda adj: cycles_C6(adj["out_set"], adj["in_set"]),
    # "scatter_gather": lambda adj: SG2(adj["out_set"], adj["in_set"]),
    # "biclique": lambda adj: BP2(adj["out_set"], adj["in_set"]),
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


# get the motifs tuples and mark the node's label based
# on wether the motifs tuples contains the node
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


# generate a graph based on parameters n, d, r, generator_type
def make_sim(n: int, d: int, r: float, generator: str, seed) -> GraphSimulator:
    return GraphSimulator(
        num_nodes=n,
        avg_degree=d,
        num_edges=None,
        network_type="type1",
        readout="node",
        node_feats=False,
        bidirectional=False,
        delta=r,
        num_graphs=1,
        generator=generator,
        seed=seed,
    )


# returns a dataset_id based on the generator-parameters
def dataset_id(n: int, d: int, r: float, generator: str, rep: int) -> str:
    return f"data_{int(n)}_{int(d)}_{r}_{generator}_rep{rep}"


def prefix_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) else 0.0


def graph_basic_stats(g) -> dict:
    edge_index = g.edge_index.detach().cpu()
    n = int(g.num_nodes)
    e_raw = int(edge_index.size(1))

    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    out_deg_raw = np.bincount(src, minlength=n).astype(np.int64)
    in_deg_raw = np.bincount(dst, minlength=n).astype(np.int64)
    _, _, out_set, in_set = build_in_out(edge_index, n)
    out_deg_unique = np.array([len(s) for s in out_set], dtype=np.int64)
    in_deg_unique = np.array([len(s) for s in in_set], dtype=np.int64)

    unique_edges = {(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())}
    e_unique = len(unique_edges)
    possible_edges = n * (n - 1)

    reciprocal_unique = sum((v, u) in unique_edges for (u, v) in unique_edges)
    reciprocity_unique = (reciprocal_unique / e_unique) if e_unique > 0 else 0.0
    duplicate_edge_fraction = (1.0 - (e_unique / e_raw)) if e_raw > 0 else 0.0

    return {
        "num_nodes": n,
        "num_edges_raw": e_raw,
        "num_edges_unique": e_unique,
        "density_raw": (e_raw / possible_edges) if possible_edges > 0 else 0.0,
        "density_unique": (e_unique / possible_edges) if possible_edges > 0 else 0.0,
        "duplicate_edge_fraction": duplicate_edge_fraction,
        "reciprocity_unique": reciprocity_unique,
        "mean_in_degree_raw": safe_mean(in_deg_raw),
        "mean_out_degree_raw": safe_mean(out_deg_raw),
        "std_in_degree_raw": safe_std(in_deg_raw),
        "std_out_degree_raw": safe_std(out_deg_raw),
        "max_in_degree_raw": int(in_deg_raw.max()) if len(in_deg_raw) else 0,
        "max_out_degree_raw": int(out_deg_raw.max()) if len(out_deg_raw) else 0,
        "mean_in_degree_unique": safe_mean(in_deg_unique),
        "mean_out_degree_unique": safe_mean(out_deg_unique),
        "std_in_degree_unique": safe_std(in_deg_unique),
        "std_out_degree_unique": safe_std(out_deg_unique),
        "max_in_degree_unique": int(in_deg_unique.max()) if len(in_deg_unique) else 0,
        "max_out_degree_unique": (
            int(out_deg_unique.max()) if len(out_deg_unique) else 0
        ),
    }


def split_task_stats(g, split_motifs: dict, tasks=TASKS) -> dict:
    y = g.y.detach().cpu().numpy()
    n = int(g.num_nodes)
    out = {}

    for col, task_name in enumerate(tasks):
        motif_count = int(len(split_motifs[task_name]))
        pos_nodes = int(y[:, col].sum())
        pos_rate = (pos_nodes / n) if n > 0 else 0.0

        out[f"{task_name}_motif_count"] = motif_count
        out[f"{task_name}_pos_nodes"] = pos_nodes
        out[f"{task_name}_pos_rate"] = pos_rate
        out[f"{task_name}_all_zero"] = int(pos_nodes == 0)

    return out


# get the motifs counts of a graph and write it to csv
def write_motif_counts_csv(path: str, split_motifs: dict, tasks=TASKS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split"] + list(tasks))
        for split_name in SPLITS:
            motifs = split_motifs[split_name]
            writer.writerow([split_name] + [len(motifs[t]) for t in tasks])


def write_label_support_csv(path: str, split_task_summary: dict, tasks=TASKS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "task", "motif_count", "pos_nodes", "pos_rate"])
        for split_name in SPLITS:
            for task_name in tasks:
                writer.writerow(
                    [
                        split_name,
                        task_name,
                        split_task_summary[split_name][f"{task_name}_motif_count"],
                        split_task_summary[split_name][f"{task_name}_pos_nodes"],
                        split_task_summary[split_name][f"{task_name}_pos_rate"],
                    ]
                )


def add_cross_split_summary(row: dict, tasks=TASKS) -> dict:
    for task_name in tasks:
        pos_rates = [row[f"{split_name}_{task_name}_pos_rate"] for split_name in SPLITS]
        pos_nodes = [
            row[f"{split_name}_{task_name}_pos_nodes"] for split_name in SPLITS
        ]
        motif_counts = [
            row[f"{split_name}_{task_name}_motif_count"] for split_name in SPLITS
        ]

        pos_nodes_mean = float(np.mean(pos_nodes))
        row[f"{task_name}_pos_nodes_mean"] = pos_nodes_mean
        row[f"{task_name}_pos_nodes_std"] = float(np.std(pos_nodes))
        row[f"{task_name}_pos_nodes_max_diff_from_mean"] = float(
            max(
                np.max(pos_nodes) - pos_nodes_mean,
                pos_nodes_mean - np.min(pos_nodes),
            )
        )

        pos_rate_mean = float(np.mean(pos_rates))
        row[f"{task_name}_pos_rate_mean"] = pos_rate_mean
        row[f"{task_name}_pos_rate_std"] = float(np.std(pos_rates))
        row[f"{task_name}_pos_rate_max_diff_from_mean"] = float(
            max(
                np.max(pos_rates) - pos_rate_mean,
                pos_rate_mean - np.min(pos_rates),
            )
        )

        row[f"{task_name}_all_splits_have_pos"] = int(min(pos_nodes) > 0)
        row[f"{task_name}_motif_count_mean"] = float(np.mean(motif_counts))
        row[f"{task_name}_motif_count_std"] = float(np.std(motif_counts))

    return row


import os


def graph_outputs_exist(out_dir_pt: str) -> bool:
    required_files = [
        os.path.join(out_dir_pt, "motif_counts.csv"),
        os.path.join(out_dir_pt, "label_support.csv"),
        *[os.path.join(out_dir_pt, f"{split}.pt") for split in SPLITS],
    ]
    return all(
        os.path.isfile(path) and os.path.getsize(path) > 0 for path in required_files
    )


def load_split_task_summary_from_csv(path: str, tasks=TASKS) -> dict:
    df = pd.read_csv(path)

    out = {}
    for split_name in SPLITS:
        sub = df[df["split"] == split_name].set_index("task")

        split_dict = {}
        for task_name in tasks:
            motif_count = int(sub.loc[task_name, "motif_count"])
            pos_nodes = int(sub.loc[task_name, "pos_nodes"])
            pos_rate = float(sub.loc[task_name, "pos_rate"])

            split_dict[f"{task_name}_motif_count"] = motif_count
            split_dict[f"{task_name}_pos_nodes"] = pos_nodes
            split_dict[f"{task_name}_pos_rate"] = pos_rate
            split_dict[f"{task_name}_all_zero"] = int(pos_nodes == 0)

        out[split_name] = split_dict

    return out


def load_split_basic_from_pt(out_dir_pt: str) -> dict:
    out = {}
    for split_name in SPLITS:
        g = torch.load(os.path.join(out_dir_pt, f"{split_name}.pt"), weights_only=False)
        out[split_name] = graph_basic_stats(g)
    return out


def build_registry_row(
    graph_id: int,
    n: int,
    d: int,
    r: float,
    generator: str,
    did: str,
    out_dir_pt: str,
    split_basic: dict,
    split_task_summary: dict,
) -> dict:
    split_seeds = {
        "train": derive_seed(BASE_SEED, f"train_g{graph_id}_{n}_{d}_{r}_{generator}"),
        "val": derive_seed(BASE_SEED, f"val_g{graph_id}_{n}_{d}_{r}_{generator}"),
        "test": derive_seed(BASE_SEED, f"test_g{graph_id}_{n}_{d}_{r}_{generator}"),
    }

    row = {
        "graph_id": graph_id,
        "dataset_id": did,
        "data_dir": out_dir_pt,
        "n": int(n),
        "d": int(d),
        "r": float(r),
        "type": generator,
        "seed_train": split_seeds["train"],
        "seed_val": split_seeds["val"],
        "seed_test": split_seeds["test"],
    }

    for split_name in SPLITS:
        row.update(prefix_keys(split_basic[split_name], f"{split_name}_"))
        row.update(prefix_keys(split_task_summary[split_name], f"{split_name}_"))

    row = add_cross_split_summary(row, TASKS)
    return row


def main():
    set_seed(BASE_SEED)
    rng = np.random.default_rng(BASE_SEED)

    # --- build all possible unique parameter combinations ---
    all_combos = list(product(N_POOL, D_POOL, R_POOL, GEN))

    NUM_GRAPHS = len(N_POOL) * len(D_POOL) * len(R_POOL) * len(GEN)

    if NUM_GRAPHS > len(all_combos):
        raise ValueError(
            f"NUM_GRAPHS={NUM_GRAPHS} > total combos={len(all_combos)}. Reduce NUM_GRAPHS or expand pools."
        )

    # sample without replacement to avoid duplicate graphs
    chosen_idx = rng.choice(len(all_combos), size=NUM_GRAPHS, replace=False)
    sampled_params = [all_combos[i] for i in chosen_idx]

    rows = []

    for graph_id, (n, d, r, generator) in enumerate(sampled_params):
        did = dataset_id(n, d, r, generator)
        out_dir_pt = os.path.join(DATA_ROOT, did)

        if graph_outputs_exist(out_dir_pt):
            print(f"Using existing data for {did}")

            split_basic = load_split_basic_from_pt(out_dir_pt)
            split_task_summary = load_split_task_summary_from_csv(
                os.path.join(out_dir_pt, "label_support.csv"),
                TASKS,
            )

            row = build_registry_row(
                graph_id=graph_id,
                n=n,
                d=d,
                r=r,
                generator=generator,
                did=did,
                out_dir_pt=out_dir_pt,
                split_basic=split_basic,
                split_task_summary=split_task_summary,
            )
            rows.append(row)
            continue

        split_seeds = {
            "train": derive_seed(
                BASE_SEED, f"train_g{graph_id}_{n}_{d}_{r}_{generator}"
            ),
            "val": derive_seed(BASE_SEED, f"val_g{graph_id}_{n}_{d}_{r}_{generator}"),
            "test": derive_seed(BASE_SEED, f"test_g{graph_id}_{n}_{d}_{r}_{generator}"),
        }

        split_graphs = {}
        split_motifs = {}
        split_basic = {}
        split_task_summary = {}

        for split_name in SPLITS:
            g = (
                make_sim(n, d, r, generator, split_seeds[split_name])
                .generate_pytorch_graph()
                .add_ports()
            )

            g, motifs = set_y_and_get_motifs(g, TASK_FUNCS)
            split_graphs[split_name] = g
            split_motifs[split_name] = motifs
            split_basic[split_name] = graph_basic_stats(g)
            split_task_summary[split_name] = split_task_stats(g, motifs, TASKS)

        os.makedirs(out_dir_pt, exist_ok=True)

        for split_name in SPLITS:
            torch.save(
                split_graphs[split_name], os.path.join(out_dir_pt, f"{split_name}.pt")
            )

        write_motif_counts_csv(
            os.path.join(out_dir_pt, "motif_counts.csv"),
            split_motifs,
            TASKS,
        )
        write_label_support_csv(
            os.path.join(out_dir_pt, "label_support.csv"),
            split_task_summary,
            TASKS,
        )

        row = build_registry_row(
            graph_id=graph_id,
            n=n,
            d=d,
            r=r,
            generator=generator,
            did=did,
            out_dir_pt=out_dir_pt,
            split_basic=split_basic,
            split_task_summary=split_task_summary,
        )
        rows.append(row)

        print(f"[{graph_id:04d}] DATA GENERATED -> {out_dir_pt}")

    df = pd.DataFrame(rows).sort_values(["graph_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(GRAPH_PARAM_CSV), exist_ok=True)
    df.to_csv(GRAPH_PARAM_CSV, index=False)
    print(f"{NUM_GRAPHS} DATA PARAMETERS STORED -> {GRAPH_PARAM_CSV}")


if __name__ == "__main__":
    main()
