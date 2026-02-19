import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from utils.fed_partitioning import graphdata_to_pyg
from utils.graph_helpers import check_and_strip_self_loops

BASE_SEED = 0
GRAPH_PARAM_CSV = "./andrea/test_generation_parameters.csv"
DATA_ROOT = "./andrea/test_data"
OUT_DIR = "./andrea/client_features"

SPLITS = ["train", "val", "test"]

MOTIF_COLS = ["cycle2", "cycle3", "cycle4", "cycle5", "cycle6", "scatter_gather", "biclique"]

def dataset_id_from_row(row: pd.Series) -> str:
    if "dataset_id" in row:
        return str(row["dataset_id"])
    return f"data_{int(row['n'])}_{int(row['d'])}_{row['r']}_{row['type']}"

def load_global_data(dataset_id: str, split_name: str):
    path = os.path.join(DATA_ROOT, dataset_id, f"{split_name}.pt")
    g = torch.load(path, weights_only=False)
    return graphdata_to_pyg(g)

def load_global_motif_counts(dataset_id: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_ROOT, dataset_id, "motif_counts.csv")
    return pd.read_csv(csv_path)

def motif_vector_from_df(motif_counts_df: pd.DataFrame, split_name: str) -> np.ndarray:
    row = motif_counts_df[motif_counts_df["split"] == split_name]
    return row.iloc[0][MOTIF_COLS].to_numpy(dtype=np.float64)

# calculate edge density for directed grpah, i.e
# number of edges / possible edges (N*(N-1))
def node_edge_density(g) -> tuple[int, int, float]:
    n = int(g.num_nodes)
    e = int(g.edge_index.size(1))
    # no self-loops; multi-edges allowed and count in E
    denom = n * (n - 1)
    dens = (e / denom) if denom > 0 else 0.0
    return n, e, dens

# build vector of in-degrees and out-degrees for a directed graph
def in_out_degree_arrays(g) -> tuple[np.ndarray, np.ndarray]:
    edge_index = g.edge_index
    n = int(g.num_nodes)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    out_deg = torch.bincount(src, minlength=n).cpu().numpy().astype(np.int64)
    in_deg = torch.bincount(dst, minlength=n).cpu().numpy().astype(np.int64)
    return in_deg, out_deg

# build a histogram for counting the degree occurrence
def degree_hist_dense(deg: np.ndarray) -> Tuple[List[int], List[float]]:
    h = np.bincount(deg).astype(np.float64)  # length = local max_deg + 1
    return h.tolist()

# per-label count vectors
def label_counts_vec(g) -> np.ndarray:
    y = g.y
    if isinstance(y, torch.Tensor):
        y = y.detach()
    return y.sum(dim=0).cpu().numpy().astype(np.float64)

# first compute number of 1's per node, i.e how many positive labels a node has,
# then we build histogram where index "i" means: how many nodes have i positive labels
# for a client
def labelset_size_hist_counts(g) -> np.ndarray:
    y = g.y
    if isinstance(y, torch.Tensor):
        y = y.detach()
    C = int(y.size(1))
    ks = y.sum(dim=1).to(torch.long).cpu().numpy()
    h = np.bincount(ks, minlength=C + 1).astype(np.float64)
    return h

# Build a directed multi-label mixing matrix M[a,b] with size num_labels * num_labels.
# For index (a,b) with value "v", "a" and "b" is the label[i] and label[j] and v
# stands for fraction of directed edges going from label "a" to label "b".
def label_mixing_counts_vec(g) -> np.ndarray:
    edge_index = g.edge_index
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)

    y = g.y
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    y = (y > 0).to(torch.float32)
    C = int(y.size(1))

    # Precompute label indices per node (sparse lists)
    label_idx: List[torch.Tensor] = [torch.nonzero(y[i], as_tuple=False).view(-1) for i in range(int(g.num_nodes))]
    M = torch.zeros((C, C), dtype=torch.float64)

    # Loop over edges; multi-edges naturally contribute multiple times
    for u, v in zip(src.tolist(), dst.tolist()):
        Su = label_idx[u]
        Tv = label_idx[v]
        su = int(Su.numel())
        tv = int(Tv.numel())
        if su == 0 or tv == 0:
            continue
        w = 1.0 / (su * tv)
        for a in Su.tolist():
            for b in Tv.tolist():
                M[a, b] += w

    return M.reshape(-1).cpu().numpy().astype(np.float64)

def _write_df(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"WROTE -> {out_path}")

# store the metrics
def extract_for_split(graph_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    records: List[Dict] = []

    for _, row in graph_df.iterrows():
        did = dataset_id_from_row(row)
        g = check_and_strip_self_loops(load_global_data(did, split_name))

        # ---- quantity ----
        n, e, dens = node_edge_density(g)

        # ---- (in/out) ----
        in_d, out_d = in_out_degree_arrays(g)

        # ---- Motifs ----
        motif_df = load_global_motif_counts(did)
        motif_counts = motif_vector_from_df(motif_df, split_name)

        # ---- label prevalence ----
        lbl_counts = label_counts_vec(g)

        # ---- labelset-size ----
        lss_counts = labelset_size_hist_counts(g)

        # ---- label mixing matrix distance ----
        mix_counts = label_mixing_counts_vec(g)

        rec = {
            "client_id": int(row["graph_id"]) if "graph_id" in row else None,
            "dataset_id": did,
            "split": split_name,
            # generator params
            "type": str(row["type"]) if "type" in row else None,
            
            # n, d, r params
            "n_param": int(row["n"]) if "n" in row else None,
            "d_param": int(row["d"]) if "d" in row else None,
            "r_param": float(row["r"]) if "r" in row else None,

            # nodes, edges, density
            "num_nodes": int(n),
            "num_edges": int(e),
            "density": float(dens),

            # in-degree, out-degree
            "in_deg_hist_counts": degree_hist_dense(in_d),
            "out_deg_hist_counts": degree_hist_dense(out_d),

            # Motifs
            "motif_counts": np.asarray(motif_counts, dtype=np.float64).tolist(),

            # Labels
            "label_counts": np.asarray(lbl_counts, dtype=np.float64).tolist(),
            "labelset_size_counts": np.asarray(lss_counts, dtype=np.float64).tolist(),
            "label_mixing_counts": np.asarray(mix_counts, dtype=np.float64).tolist(),
        }
        records.append(rec)

    return pd.DataFrame.from_records(records)


def main():
    graph_df = pd.read_csv(GRAPH_PARAM_CSV)

    # Ensure we have a stable client_id
    if "graph_id" not in graph_df.columns:
        graph_df = graph_df.reset_index(drop=True)
        graph_df["graph_id"] = np.arange(len(graph_df), dtype=int)

    os.makedirs(OUT_DIR, exist_ok=True)

    meta = {
        "graph_param_csv": GRAPH_PARAM_CSV,
        "data_root": DATA_ROOT,
        "splits": SPLITS,
        "motif_cols": MOTIF_COLS,
        "notes": "Per-client cached features for fast subset heterogeneity. Arrays stored as list columns.",
    }

    for split in SPLITS:
        df = extract_for_split(graph_df, split)
        out_path = os.path.join(OUT_DIR, f"client_features_{split}.parquet")
        _write_df(df, out_path)
    meta["columns"] = list(df.columns)
    meta_path = os.path.join(OUT_DIR, "client_features_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"WROTE -> {meta_path}")

if __name__ == "__main__":
    main()
