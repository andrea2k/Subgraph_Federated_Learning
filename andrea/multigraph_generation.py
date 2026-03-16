import os
import csv
from itertools import product
import pandas as pd
import torch
import numpy as np

from utils.seed import set_seed, derive_seed
from scripts.data.simulator import GraphSimulator

from andrea.witness_funcs import (
    build_unique_in_out,
    cycles_C2,
    cycles_C3,
    cycles_C4,
    cycles_C5,
    cycles_C6,
    SG2,
    BP2,
)

BASE_SEED = 0
DATA_ROOT = "./andrea/test_data"
GRAPH_PARAM_CSV = "./andrea/test_generation_parameters.csv"

N_POOL = [1000, 1500, 2000]
D_POOL = [4, 5, 6]
R_POOL = [2.5, 3.0, 3.5, 4.0]
GEN = ["chordal", "barabasi", "watts"]

NUM_GRAPHS = 16

TASK_FUNCS = {
    "cycle2": lambda out_set, in_set: cycles_C2(out_set, in_set),
    "cycle3": lambda out_set, in_set: cycles_C3(out_set, in_set),
    "cycle4": lambda out_set, in_set: cycles_C4(out_set, in_set),
    "cycle5": lambda out_set, in_set: cycles_C5(out_set, in_set),
    "cycle6": lambda out_set, in_set: cycles_C6(out_set, in_set),
    "scatter_gather": lambda out_set, in_set: SG2(out_set, in_set),
    "biclique": lambda out_set, in_set: BP2(out_set, in_set),
}
TASKS = list(TASK_FUNCS.keys())


# get the motifs tuples and mark the node's label based
# on wether the motifs tuples contains the node
def set_y_and_get_motifs(g, task_funcs=TASK_FUNCS):
    edge_index = g.edge_index
    num_nodes = int(g.num_nodes)

    out_set, in_set = build_unique_in_out(edge_index, num_nodes)

    tasks = list(task_funcs.keys())
    y = torch.zeros((num_nodes, len(tasks)), dtype=torch.float32)

    motifs_tuple = {}

    for col, task_name in enumerate(tasks):
        motifs = task_funcs[task_name](out_set, in_set)  # list of tuples
        motifs_tuple[task_name] = motifs

        # mark all nodes in witness tuple
        for w in motifs:
            for u in w:
                y[int(u), col] = 1.0

    g.y = y
    g.num_classes = y.shape[1]
    return g, motifs_tuple


# get the motifs counts of a graph and store it in a csv file
def write_motif_counts_csv(path, split_motifs, tasks=TASKS):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split"] + tasks)
        for split_name in ["train", "val", "test"]:
            motifs = split_motifs[split_name]
            writer.writerow([split_name] + [len(motifs[t]) for t in tasks])


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
def dataset_id(n: int, d: int, r: float, generator: str) -> str:
    return f"data_{int(n)}_{int(d)}_{r}_{generator}"


def main():
    set_seed(BASE_SEED)
    rng = np.random.default_rng(BASE_SEED)

    # --- build all possible unique parameter combinations ---
    all_combos = list(product(N_POOL, D_POOL, R_POOL, GEN))
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

        # unique split seeds per graph_id
        split_seeds = {
            "train": derive_seed(
                BASE_SEED, f"train_g{graph_id}_{n}_{d}_{r}_{generator}"
            ),
            "val": derive_seed(BASE_SEED, f"val_g{graph_id}_{n}_{d}_{r}_{generator}"),
            "test": derive_seed(BASE_SEED, f"test_g{graph_id}_{n}_{d}_{r}_{generator}"),
        }

        # generate train/val/test
        tr = (
            make_sim(n, d, r, generator, split_seeds["train"])
            .generate_pytorch_graph()
            .add_ports()
        )
        tr, tr_motifs = set_y_and_get_motifs(tr, TASK_FUNCS)

        va = (
            make_sim(n, d, r, generator, split_seeds["val"])
            .generate_pytorch_graph()
            .add_ports()
        )
        va, va_motifs = set_y_and_get_motifs(va, TASK_FUNCS)

        te = (
            make_sim(n, d, r, generator, split_seeds["test"])
            .generate_pytorch_graph()
            .add_ports()
        )
        te, te_motifs = set_y_and_get_motifs(te, TASK_FUNCS)

        # write
        os.makedirs(out_dir_pt, exist_ok=True)
        torch.save(tr, os.path.join(out_dir_pt, "train.pt"))
        torch.save(va, os.path.join(out_dir_pt, "val.pt"))
        torch.save(te, os.path.join(out_dir_pt, "test.pt"))

        split_motifs = {"train": tr_motifs, "val": va_motifs, "test": te_motifs}
        write_motif_counts_csv(
            os.path.join(out_dir_pt, "motif_counts.csv"), split_motifs, TASKS
        )

        print(f"[{graph_id:02d}] DATA GENERATED -> {out_dir_pt}")

        # registry row
        rows.append(
            {
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
                "num_nodes_train": int(tr.num_nodes),
                "num_nodes_val": int(va.num_nodes),
                "num_nodes_test": int(te.num_nodes),
                "num_edges_train": int(tr.edge_index.size(1)),
                "num_edges_val": int(va.edge_index.size(1)),
                "num_edges_test": int(te.edge_index.size(1)),
            }
        )

    # write registry CSV (keep generation order)
    df = pd.DataFrame(rows).sort_values(["graph_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(GRAPH_PARAM_CSV), exist_ok=True)
    df.to_csv(GRAPH_PARAM_CSV, index=False)
    print(f"DATA PARAMETERS STORED -> {GRAPH_PARAM_CSV}")


if __name__ == "__main__":
    main()
