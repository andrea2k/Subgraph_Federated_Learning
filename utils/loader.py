# utils/loader.py
import os
import torch

def load_client_graphs(clients_dir, num_clients):
    out = []
    for cid in range(num_clients):
        p1 = os.path.join(clients_dir, f"client_{cid:04d}.pt")
        p2 = os.path.join(clients_dir, f"client_{cid}.pt")
        path = p1 if os.path.exists(p1) else p2
        out.append(torch.load(path, weights_only=False))
    return out
