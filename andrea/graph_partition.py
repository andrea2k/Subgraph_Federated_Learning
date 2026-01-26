import os
import csv

import pandas as pd
import torch

from utils.seed import set_seed, derive_seed
from utils.fed_partitioning import graphdata_to_pyg
from utils.fed_simulation import (
    louvain_label_imbalance_split,
    metis_label_imbalance_split,
    louvain_original_split,
    metis_original_split,
)
from andrea.multigraph_generation import (
    DATA_ROOT, 
    GRAPH_PARAM_CSV, 
    TASK_FUNCS, 
    TASKS, 
    set_y_and_count_motifs, 
)

BASE_SEED = 0
PARTITION_PARAM_CSV = "./andrea/graph_partition_parameters.csv"
FED_OUT_ROOT = "./andrea/subgraph_data"

SPLIT_METHODS = [
    "louvain_label_imbalance",
    "metis_label_imbalance",
    "louvain_original",
    "metis_original",
]
N_CLIENTS_LIST = [32]
LOUVAIN_RES_LIST = [1.0, 1.5]
ASSIGN_METHOD_LIST = ["zipf", "equal"]
ALPHA_LIST = [1.2]

SAVE_CLIENTS_PT = True

def METIS_NUM_COMS_LIST(n_clients):
    return [2 * n_clients, 3 * n_clients]

def build_split_configs():
    """Return list of cfg dicts. Each cfg becomes one split directory."""
    split_configs = []

    for method in SPLIT_METHODS:
        for n_clients in N_CLIENTS_LIST:

            if method == "louvain_label_imbalance":
                for res in LOUVAIN_RES_LIST:
                    seed_tag = f"{method}_res{res}_c{n_clients}"
                    split_configs.append({
                        "method": method,
                        "n_clients": n_clients,
                        "louvain_res": res,
                        "metis_num_coms": None,
                        "client_assignment": None,
                        "alpha": None,
                        "seed_tag": seed_tag,
                    })

            elif method == "metis_label_imbalance":
                for num_coms in METIS_NUM_COMS_LIST(n_clients):
                    seed_tag = f"{method}_com{num_coms}_c{n_clients}"
                    split_configs.append({
                        "method": method,
                        "n_clients": n_clients,
                        "louvain_res": None,
                        "metis_num_coms": num_coms,
                        "client_assignment": None,
                        "alpha": None,
                        "seed_tag": seed_tag,
                    })

            elif method == "louvain_original":
                for res in LOUVAIN_RES_LIST:
                    for assign in ASSIGN_METHOD_LIST:
                        if assign == "zipf":
                            for alpha in ALPHA_LIST:
                                seed_tag = f"{method}_res{res}_{assign}{alpha}_c{n_clients}"
                                split_configs.append({
                                    "method": method,
                                    "n_clients": n_clients,
                                    "louvain_res": res,
                                    "metis_num_coms": None,
                                    "client_assignment": assign,
                                    "alpha": alpha,
                                    "seed_tag": seed_tag,
                                })
                        else:
                            seed_tag = f"{method}_res{res}_{assign}_c{n_clients}"
                            split_configs.append({
                                "method": method,
                                "n_clients": n_clients,
                                "louvain_res": res,
                                "metis_num_coms": None,
                                "client_assignment": assign,
                                "alpha": None,
                                "seed_tag": seed_tag,
                            })

            elif method == "metis_original":
                for num_coms in METIS_NUM_COMS_LIST(n_clients):
                    for assign in ASSIGN_METHOD_LIST:
                        if assign == "zipf":
                            for alpha in ALPHA_LIST:
                                seed_tag = f"{method}_com{num_coms}_{assign}{alpha}_c{n_clients}"
                                split_configs.append({
                                    "method": method,
                                    "n_clients": n_clients,
                                    "louvain_res": None,
                                    "metis_num_coms": num_coms,
                                    "client_assignment": assign,
                                    "alpha": alpha,
                                    "seed_tag": seed_tag,
                                })
                        else:
                            seed_tag = f"{method}_com{num_coms}_{assign}_c{n_clients}"
                            split_configs.append({
                                "method": method,
                                "n_clients": n_clients,
                                "louvain_res": None,
                                "metis_num_coms": num_coms,
                                "client_assignment": assign,
                                "alpha": None,
                                "seed_tag": seed_tag,
                            })

    # make deterministic filesystem-safe split_id
    for cfg in split_configs:
        cfg["split_id"] = cfg["seed_tag"].replace(".", "p")

    return split_configs


def run_split_from_config(global_data, cfg, seed):
    method = cfg["method"]
    n_clients = int(cfg["n_clients"])

    if method == "louvain_label_imbalance":
        return louvain_label_imbalance_split(
            global_data,
            num_clients=n_clients,
            resolution=float(cfg["louvain_res"]),
            seed=seed,
            return_node_indices=False,
        )

    if method == "metis_label_imbalance":
        return metis_label_imbalance_split(
            global_data,
            num_clients=n_clients,
            metis_num_coms=int(cfg["metis_num_coms"]),
            seed=seed,
            return_node_indices=False,
        )

    if method == "louvain_original":
        return louvain_original_split(
            global_data,
            num_clients=n_clients,
            resolution=float(cfg["louvain_res"]),
            alpha=(None if cfg["alpha"] is None else float(cfg["alpha"])),
            client_assignment=str(cfg["client_assignment"]),
            seed=seed,
            return_node_indices=False,
        )

    if method == "metis_original":
        return metis_original_split(
            global_data,
            num_clients=n_clients,
            metis_num_coms=int(cfg["metis_num_coms"]),
            alpha=(None if cfg["alpha"] is None else float(cfg["alpha"])),
            client_assignment=str(cfg["client_assignment"]),
            seed=seed,
            return_node_indices=False,
        )

    raise ValueError(f"Unknown method: {method}")

def dataset_id_from_row(row):
    return f"data_{int(row['n'])}_{int(row['d'])}_{row['r']}_{row['type']}"

def load_global_data(dataset_id, split_name):
    path = os.path.join(DATA_ROOT, dataset_id, f"{split_name}.pt")
    g = torch.load(path, weights_only=False)
    return graphdata_to_pyg(g)

def sanity_check(global_data, clients):
    num_global = int(global_data.num_nodes)
    all_global_nodes = []
    for cid, c in enumerate(clients):
        if (c.num_nodes == 0):
            print(f"Client:{cid} has 0 nodes" )
        gm = c.global_map
        all_global_nodes.extend(gm.tolist())

    assert len(all_global_nodes) == num_global, (
        f"Total assigned nodes = {len(all_global_nodes)}, "
        f"but global_data has {num_global} nodes"
    )

    unique_nodes = set(all_global_nodes)
    assert len(unique_nodes) == num_global, (
        "Duplicate or missing global nodes detected across clients"
    )
    print("Sanity check passed: partition is valid.")

def save_split_dir(out_dir, clients, tasks=TASKS):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "motif_counts.csv")
    PRINTED = False
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id"] + TASKS) 
        for cid, client_data in enumerate(clients):
            g, counts = set_y_and_count_motifs(client_data, TASK_FUNCS)
            fname = f"client_{cid}.pt"
            fpath = os.path.join(out_dir, fname)
            torch.save(g, fpath)
            writer.writerow([cid] + [counts[t] for t in tasks])
    print(f"DATA PARTITION -> {out_dir} ({len(clients)} clients)")
    
def main():
    set_seed(BASE_SEED)

    graphs_df = pd.read_csv(GRAPH_PARAM_CSV).sort_values(["n", "d", "r", "type"]).reset_index(drop=True)
    split_configs = build_split_configs()

    rows = []

    for _, grow in graphs_df.iterrows():

        dataset_id = dataset_id_from_row(grow)

        for split_name in ["train", "val", "test"]:
            global_data = load_global_data(dataset_id, split_name)

            for cfg in split_configs:
                method = str(cfg["method"])
                split_id = cfg["split_id"]
                seed_tag = str(cfg["seed_tag"])

                seed = derive_seed(BASE_SEED, f"{dataset_id}_{split_name}_{seed_tag}")

                clients = run_split_from_config(global_data, cfg, seed)
                sanity_check(global_data, clients)
                
                out_dir = os.path.join(FED_OUT_ROOT, dataset_id, split_name, split_id)
                
                save_split_dir(out_dir, clients)

                row = {
                    "dataset_id": dataset_id,
                    "n": int(grow["n"]),
                    "d": int(grow["d"]),
                    "r": float(grow["r"]),
                    "type": str(grow["type"]),

                    "split": split_name,
                    "method": method,
                    "split_id": split_id,
                    "seed_tag": seed_tag,
                    "seed": int(seed),

                    "n_clients": int(cfg["n_clients"]),
                    "louvain_res": cfg.get("louvain_res", ""),
                    "metis_num_coms": cfg.get("metis_num_coms", ""),
                    "client_assignment": cfg.get("client_assignment", ""),
                    "alpha": cfg.get("alpha", ""),
                }

                rows.append(row)
                break
            break
        break
    df = pd.DataFrame(rows)

    sort_cols = ["dataset_id", "split", "method", "n_clients", "split_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(PARTITION_PARAM_CSV, index=False)
    print(f"PARTITION PARAMETERS STORED -> {PARTITION_PARAM_CSV}")

if __name__ == "__main__":
    main()