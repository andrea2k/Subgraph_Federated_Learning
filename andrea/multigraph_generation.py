import os
import csv
from itertools import product
import pandas as pd
import torch

from utils.seed import set_seed, derive_seed 
from scripts.data.simulator import GraphSimulator
from scripts.data.generate_synthetic import write_label_stats

from andrea.witness_funcs import (
    build_unique_in_out,
    cycles_C2, cycles_C3, cycles_C4, cycles_C5, cycles_C6,
    SG2, BP2,
)

TASK_FUNCS = {
    "cycle2": lambda out, out_set, in_set: cycles_C2(out, out_set),
    "cycle3": lambda out, out_set, in_set: cycles_C3(out, out_set),
    "cycle4": lambda out, out_set, in_set: cycles_C4(out, out_set),
    "cycle5": lambda out, out_set, in_set: cycles_C5(out, out_set),
    "cycle6": lambda out, out_set, in_set: cycles_C6(out, out_set),
    "scatter_gather": lambda out, out_set, in_set: SG2(out_set, in_set),
    "biclique": lambda out, out_set, in_set: BP2(out_set, in_set),
}
TASKS = list(TASK_FUNCS.keys())

def set_y_and_count_motifs(g, task_funcs=TASK_FUNCS):
    edge_index = g.edge_index
    num_nodes = int(g.num_nodes)

    out, out_set, inn, in_set = build_unique_in_out(edge_index, num_nodes)

    tasks = list(task_funcs.keys())
    y = torch.zeros((num_nodes, len(tasks)), dtype=torch.float32)

    counts = {}

    for col, task_name in enumerate(tasks):
        motifs = task_funcs[task_name](out, out_set, in_set)  # list of tuples
        counts[task_name] = len(motifs)

        # label_mode="all": mark all nodes in witness tuple
        for w in motifs:
            for u in w:
                y[int(u), col] = 1.0

    g.y = y
    g.num_classes = y.shape[1]
    return g, counts

def write_y_sums_csv(path, split_counts, tasks=TASKS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split"] + tasks)
        for split_name in ["tr", "va", "te"]:
            c = split_counts[split_name]
            writer.writerow([split_name] + [c[t] for t in tasks])

N_LIST = [1000,2000]              # nodes
D_LIST = [4]                 # average degree
R_LIST = [4.0]         # locality radius / delta
GEN_LIST = ["chordal"]   # generator

BASE_SEED = 0

def main():

    set_seed(BASE_SEED)
    
    rows = []
    for (n, d, r, generator) in product(N_LIST, D_LIST, R_LIST, GEN_LIST):
        split_seeds = {
        "train": derive_seed(BASE_SEED, f"train_{n}_{d}_{r}_{generator}"),
        "val":   derive_seed(BASE_SEED, f"val_{n}_{d}_{r}_{generator}"),
        "test":  derive_seed(BASE_SEED, f"test_{n}_{d}_{r}_{generator}"),
        }

        def make_sim():
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
            )
        
        set_seed(split_seeds["train"])
        tr = make_sim().generate_pytorch_graph().add_ports()
        tr, tr_counts = set_y_and_count_motifs(tr, TASK_FUNCS)

        set_seed(split_seeds["val"])
        va = make_sim().generate_pytorch_graph().add_ports()
        va, va_counts = set_y_and_count_motifs(va, TASK_FUNCS)

        set_seed(split_seeds["test"])
        te = make_sim().generate_pytorch_graph().add_ports()
        te, te_counts = set_y_and_count_motifs(te, TASK_FUNCS)

        out_dir_pt = f"./andrea/data/data_{n}_{d}_{r}_{generator}"
        os.makedirs(out_dir_pt, exist_ok=True)

        torch.save(tr, os.path.join(out_dir_pt, "train.pt"))
        torch.save(va, os.path.join(out_dir_pt, "val.pt"))
        torch.save(te, os.path.join(out_dir_pt, "test.pt"))

        split_counts = {"tr": tr_counts, "va": va_counts, "te": te_counts}
        write_y_sums_csv(f"{out_dir_pt}/y_sums.csv", split_counts, TASKS)
        
        row = {
            "n": n,
            "d": d,
            "r": r,
            "type": generator,
            "seed_tr": split_seeds["train"],
            "seed_va": split_seeds["val"],
            "seed_te": split_seeds["test"],
            "num_nodes_tr": int(tr.num_nodes),
            "num_nodes_va": int(va.num_nodes),
            "num_nodes_te": int(te.num_nodes),
            "num_edges_tr": int(tr.edge_index.size(1)),
            "num_edges_va": int(va.edge_index.size(1)),
            "num_edges_te": int(te.edge_index.size(1)),
        }
        
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["n", "d", "r", "type"]).reset_index(drop=True)
    out_dir = "./andrea"
    df.to_csv(os.path.join(out_dir, "data_parameters.csv"), index=False)

if __name__ == "__main__":
    main()