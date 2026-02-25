import os
import json
import copy
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn

from utils.train_utils import load_datasets, ensure_node_features, train_epoch, evaluate_epoch
from utils.graph_helpers import check_and_strip_self_loops, build_hetero_neighbor_loader, build_full_eval_loader, max_port_cols
from utils.hetero import make_bidirected_hetero
from models.pna_reverse_mp import PNANetReverseMP, compute_directional_degree_hists
from utils.seed import set_seed

# -----------------------------
# Config / paths (edit as needed)
# -----------------------------
CSV_PATH = "./andrea/test_generation_parameters.csv"
CONFIG_PATH = "./configs/pna_configs.json"
CONFIG_KEY = "reverse_mp_with_port_and_ego"

# FedAvg controls (minimal defaults)
NUM_CLIENTS_USE = 2          # use first N clients from CSV (set None to use all)
ROUNDS = 3
CLIENT_FRACTION = 1.0        # 1.0 = all selected clients each round
LOCAL_EPOCHS = 1
SEED = 0

# Logging / eval controls
EVAL_EVERY = 1               # evaluate every N rounds
EVAL_ON = "val"              # "val" or "test"
EVAL_CLIENTS = 2             # evaluate on first M clients (cheap)

MODEL_LEVEL_KEYS = [
    "base_seed",
    "use_ego_ids",
    "use_port_ids",
    "use_mini_batch",
    "batch_size",
    "port_emb_dim",
    "num_epochs",
    "enable_cross_client_comm",
    "cross_client_mix_alpha",
]

def load_cfg(config_path: str, key: str) -> Dict:
    with open(config_path, "r") as f:
        all_cfg = json.load(f)
    cfg_obj = all_cfg[key]

    # Your JSON stores hyperparams under "default_hparams"
    h = cfg_obj["default_hparams"]
    # carry a few top-level flags (if you ever add them later)
    h["_config_name"] = cfg_obj.get("name", key)

    for k in MODEL_LEVEL_KEYS:
        if k in cfg_obj:
            h[k] = cfg_obj[k]
    return h

def load_client_from_dir(client_dir: str):
    train_g, val_g, test_g = load_datasets(
        log_dir=client_dir,
        train_data_file="train.pt",
        val_data_file="val.pt",
        test_data_file="test.pt",
    )
    return train_g, val_g, test_g

def make_model(cfg: Dict, x_dim: int, out_dim: int,
               deg_fwd_hist: torch.Tensor, deg_rev_hist: torch.Tensor,
               in_vocab: int, out_vocab: int) -> PNANetReverseMP:

    ego_dim = cfg["ego_dim"] if cfg.get("ego_dim", None) is not None else cfg["batch_size"]

    model = PNANetReverseMP(
        in_dim=x_dim,
        out_dim=out_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        deg_fwd_hist=deg_fwd_hist,
        deg_rev_hist=deg_rev_hist,
        ego_dim=ego_dim,
        in_port_vocab_size=in_vocab if cfg.get("use_port_ids", True) else None,
        out_port_vocab_size=out_vocab if cfg.get("use_port_ids", True) else None,
        port_emb_dim=cfg.get("port_emb_dim", 0) if cfg.get("use_port_ids", True) else None,
    )
    return model

def compute_global_port_vocab(*graph_lists: List) -> Tuple[int, int]:
    max_in, max_out = 0, 0
    for graphs in graph_lists:
        for g in graphs:
            mi, mo = max_port_cols(g)
            max_in = max(max_in, int(mi))
            max_out = max(max_out, int(mo))
    return max_in + 1, max_out + 1

def compute_global_degree_hists(train_homo_graphs: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple global hist: compute histogram on concatenated degrees across clients.
    We accumulate degrees per node within each graph, then append into a big list of degrees.
    """
    all_in_deg = []
    all_out_deg = []

    for g in train_homo_graphs:
        ei = g.edge_index
        n = g.num_nodes

        # in-degree: counts of destinations
        in_deg = torch.bincount(ei[1], minlength=n).cpu()
        out_deg = torch.bincount(ei[0], minlength=n).cpu()

        all_in_deg.append(in_deg)
        all_out_deg.append(out_deg)

    in_deg_cat = torch.cat(all_in_deg, dim=0)
    out_deg_cat = torch.cat(all_out_deg, dim=0)

    # build histograms like compute_directional_degree_hists does
    # deg_fwd_hist: hist of in-degrees
    # deg_rev_hist: hist of out-degrees (equivalently in-degrees on reversed)
    max_in = int(in_deg_cat.max().item()) if in_deg_cat.numel() else 0
    max_out = int(out_deg_cat.max().item()) if out_deg_cat.numel() else 0

    deg_fwd_hist = torch.bincount(in_deg_cat, minlength=max_in + 1).float()
    deg_rev_hist = torch.bincount(out_deg_cat, minlength=max_out + 1).float()
    return deg_fwd_hist, deg_rev_hist


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg(CONFIG_PATH, CONFIG_KEY)
    print("Loaded cfg:", cfg)

    # Load clients list from CSV
    df = pd.read_csv(CSV_PATH)
    if NUM_CLIENTS_USE is not None:
        df = df.head(NUM_CLIENTS_USE).reset_index(drop=True)

    client_dirs = df["data_dir"].tolist()
    client_ids = df["dataset_id"].tolist() if "dataset_id" in df.columns else [f"client_{i}" for i in range(len(df))]
    num_clients = len(client_dirs)
    print(f"Loaded {num_clients} clients from CSV.")

    # Load all client graphs (homo) + sanity check
    homo_train, homo_val, homo_test = [], [], []
    for cid, cdir in zip(client_ids, client_dirs):
        tr, va, te = load_client_from_dir(cdir)
        homo_train.append(tr); homo_val.append(va); homo_test.append(te)

    # Compute global shared shapes
    x_dim = int(homo_train[0].x.size(-1))
    out_dim = int(homo_train[0].y.size(-1))
    edge_dim = int(homo_train[0].edge_attr.size(-1))
    deg_fwd_hist, deg_rev_hist = compute_directional_degree_hists(
        edge_index=homo_train[0].edge_index,
        num_nodes=homo_train[0].num_nodes,
    )
    print("num nodes", homo_train[0].num_nodes)
    print("x dim", x_dim)
    print("y dim", out_dim)
    print("edge dim", edge_dim)
    print("deg_fwd_first_hist", deg_fwd_hist, "numel", deg_fwd_hist.numel())
    print("deg_rev_first_hist", deg_rev_hist, "numel", deg_rev_hist.numel())

    # Compute global port vocab (required if using port embeddings)
    if cfg.get("use_port_ids", True):
        in_vocab, out_vocab = compute_global_port_vocab(homo_train, homo_val, homo_test)
    else:
        in_vocab, out_vocab = 0, 0
    print(f"Global port vocab sizes: in={in_vocab}, out={out_vocab}")

    deg_fwd_hist, deg_rev_hist = compute_global_degree_hists(homo_train) 

    print("deg global fwd hist", deg_fwd_hist)
    print("deg global rev hist", deg_rev_hist)

    clients = []
    for i, cid in enumerate(client_ids):
        tr_h = make_bidirected_hetero(homo_train[i])
        va_h = make_bidirected_hetero(homo_val[i])
        te_h = make_bidirected_hetero(homo_test[i])
        clients.append({"id": cid, "train": tr_h, "val": va_h, "test": te_h})

    print(clients[0])

if __name__ == "__main__":
    main()