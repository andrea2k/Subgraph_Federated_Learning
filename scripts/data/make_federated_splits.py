import os
import json
import torch
from torch_geometric.data import Data

from utils.seed import set_seed, derive_seed
from utils.fed_partitioning import graphdata_to_pyg, get_subgraph_pyg_data
from utils.fed_simulation import louvain_label_imbalance_split, metis_label_imbalance_split, louvain_original_split, metis_original_split

CONFIG_PATH = "./configs/fed_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["federated_dataset_simulation"]

NUM_CLIENTS = CONFIG["num_clients"]
LOUVAIN_RESOLUTION = CONFIG["louvain_resolution"]
METIS_NUM_COMS = CONFIG["metis_num_coms"]  # taken to be greater than num of clients to ensure at least one community per client
BASE_SEED = CONFIG.get("base_seed", 0)


def main():
    # set a global seed
    set_seed(BASE_SEED)

    # derive different seeds for each splitting method
    # Each split has a distinct and reproducible seed
    split_seeds = {
        "louvain": derive_seed(BASE_SEED, "louvain"),
        "metis": derive_seed(BASE_SEED, "metis"),
        "louvain_imbalance": derive_seed(BASE_SEED, "louvain_imbalance"),
        "metis_imbalance": derive_seed(BASE_SEED, "metis_imbalance"),
    }

    # load synthetic graph and convert to PyG Data
    train_graphdata = torch.load("./data/train.pt", weights_only=False)
    global_data = graphdata_to_pyg(train_graphdata)

    # Louvain-based label imbalance split
    louvain_node_splits = louvain_label_imbalance_split(
        global_data,
        num_clients=NUM_CLIENTS,
        resolution=LOUVAIN_RESOLUTION,
        seed=split_seeds["louvain_imbalance"],
        return_node_indices=True,
    )

    # Metis-based label imbalance split
    metis_node_splits = metis_label_imbalance_split(
        global_data,
        num_clients=NUM_CLIENTS,
        metis_num_coms=METIS_NUM_COMS,
        seed=split_seeds["metis_imbalance"],
        return_node_indices=True,
    )

    louvain_orig_node_splits = louvain_original_split(
        global_data,
        num_clients=NUM_CLIENTS,
        resolution=LOUVAIN_RESOLUTION,
        seed=split_seeds["louvain"],
        alpha=1.5,              # skewness increases as alpha increases
        return_node_indices=True,
    )

    metis_orig_node_splits = metis_original_split(
        global_data,
        num_clients=NUM_CLIENTS,
        metis_num_coms=METIS_NUM_COMS,
        seed=split_seeds["metis"],
        alpha=1.5,
        return_node_indices=True,
    )

    # construct client subgraphs
    print("Constructing client subgraphs...")
    louvain_clients = [get_subgraph_pyg_data(global_data, node_idx) for node_idx in louvain_node_splits]
    metis_clients = [get_subgraph_pyg_data(global_data, node_idx) for node_idx in metis_node_splits]

    louvain_orig_clients = [get_subgraph_pyg_data(global_data, node_idx) for node_idx in louvain_orig_node_splits]
    metis_orig_clients = [get_subgraph_pyg_data(global_data, node_idx) for node_idx in metis_orig_node_splits]

    # save federated splits with label imbalance
    louvain_dir = "./data/fed_louvain_imbalance_splits"
    metis_dir = "./data/fed_metis_imbalance_splits"
    os.makedirs(louvain_dir, exist_ok=True)
    os.makedirs(metis_dir, exist_ok=True)

    for cid, data in enumerate(louvain_clients):
        torch.save(data, os.path.join(louvain_dir, f"client_{cid}.pt"))
    for cid, data in enumerate(metis_clients):
        torch.save(data, os.path.join(metis_dir, f"client_{cid}.pt"))

    # save original federated splits
    louvain_orig_dir = "./data/fed_louvain_splits"
    metis_orig_dir = "./data/fed_metis_splits"
    os.makedirs(louvain_orig_dir, exist_ok=True)
    os.makedirs(metis_orig_dir, exist_ok=True)

    for cid, data in enumerate(louvain_orig_clients):
        torch.save(data, os.path.join(louvain_orig_dir, f"client_{cid}.pt"))
    for cid, data in enumerate(metis_orig_clients):
        torch.save(data, os.path.join(metis_orig_dir, f"client_{cid}.pt"))

    # Log the seeds used
    out_dir = "./data/seeds"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "fed_seeds.txt"), "w") as f:
        for k, v in split_seeds.items():
            f.write(f"{k}:{v}\n")

    print("Done. Federated splits successfully generated.")

if __name__ == "__main__":
    main()
