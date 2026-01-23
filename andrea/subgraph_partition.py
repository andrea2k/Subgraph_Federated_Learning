import os
import json
import csv
import torch

import numpy as np
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
import pymetis as metis
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx


from utils.metrics import write_fed_split_sizes
from utils.seed import set_seed, derive_seed
from utils.fed_partitioning import graphdata_to_pyg, get_subgraph_pyg_data
from utils.fed_simulation import louvain_label_imbalance_split, metis_label_imbalance_split, louvain_original_split, metis_original_split
from andrea.multigraph_generation import TASK_FUNCS, TASKS

BASE_SEED = 0

def main():

    n_clients = 32
    louvain_res = 1.0
    num_coms = 32
    skewness = 1.5
    
    set_seed(BASE_SEED)

    split_seeds = {
    "louvain": derive_seed(BASE_SEED, "louvain"),
    "metis": derive_seed(BASE_SEED, "metis"),
    "louvain_imbalance": derive_seed(BASE_SEED, "louvain_imbalance"),
    "metis_imbalance": derive_seed(BASE_SEED, "metis_imbalance"),
    }

    # load synthetic graph and convert to PyG Data
    global_data = torch.load("./andrea/data/data_2000_4_4.0_chordal/train.pt", weights_only=False)
    global_data = graphdata_to_pyg(global_data)
    num_classes = global_data.num_classes

    print(global_data)
    print(global_data.y.shape)

    G = to_networkx(global_data, to_undirected=True)
    comms = nx.algorithms.community.louvain_communities(
        G, resolution=louvain_res, seed=split_seeds["louvain_imbalance"]
    )
    communities = {}
    test = 0
    for com_id, list_nodes in enumerate(comms):
        if com_id not in communities:
            communities[com_id] = {
                "nodes": [],
                "label_distribution": np.zeros(num_classes, dtype=float),
            }
        for node_id in list_nodes:
            communities[com_id]["nodes"].append(node_id)
            label_vec = global_data.y[node_id].numpy()
            communities[com_id]["label_distribution"] += label_vec

    num_communities = len(communities)
    print(num_communities)
    label_total = np.zeros(num_classes, dtype=float)
    for i in range(num_communities):
        label_total += communities[i]["label_distribution"]
    print(label_total)
    # com_assignments = np.empty(global_data.num_nodes, dtype=np.int32)
    # for cid, nodes in enumerate(comms):
    #     for u in nodes:
    #         com_assignments[int(u)] = cid
    # # build per-community label distributions (vectors of length num_classes)
    # communities = {}
    # for node_id, com_id in enumerate(com_assignments):
    #     com_id = int(com_id)
    #     if com_id not in communities:
    #         communities[com_id] = {
    #             "nodes": [],
    #             "label_distribution": np.zeros(num_classes, dtype=float),
    #         }
    #     communities[com_id]["nodes"].append(node_id)
    #     # accumulate the label vector (multi-task)
    #     label_vec = global_data.y[node_id].cpu().numpy()  # shape [num_classes]
    #     communities[com_id]["label_distribution"] += label_vec

    # num_communities = len(communities)

    # # normalize label distributions and create clustering features
    # clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    # for com_id in communities.keys():
    #     dist = communities[com_id]["label_distribution"]
    #     total = dist.sum()
    #     if total > 0:
    #         dist = dist / total
    #     clustering_data[com_id, :] = dist

    # # kMeans over communities by label distribution
    # kmeans = KMeans(n_clusters=num_clients, n_init="auto", random_state=seed)
    # clustering_labels = kmeans.fit_predict(clustering_data)  # community -> client

    # # aggregate communities into clients
    # client_indices = {cid: [] for cid in range(num_clients)}
    # for com_id in range(num_communities):
    #     client_id = int(clustering_labels[com_id])
    #     client_indices[client_id] += communities[com_id]["nodes"]

    # # if the user only wants the node indices, return them without computing local subgraphs
    # if return_node_indices:
    #     return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # # else, build local subgraphs
    # local_data = []
    # for client_id in range(num_clients):
    #     node_list = sorted(client_indices[client_id])
    #     local_subgraph = get_subgraph_pyg_data(global_data, node_list)
    #     if local_subgraph.edge_index.numel() == 0:
    #         # TODO: if no edges, we can add random edges or leave as empty
    #         pass
    #     local_data.append(local_subgraph)

    # train_graphdata = torch.load("./andrea/data/data_1000_4_4.0_chordal/train.pt", weights_only=False)
    # global_data = graphdata_to_pyg(train_graphdata)
    
    # louvain_split = louvain_original_split(
    #     global_data,
    #     num_clients=n_clients,
    #     resolution=louvain_res,
    #     seed=split_seeds["louvain"],
    #     alpha=skewness,              # skewness increases as alpha increases
    #     client_assignment="zipf",
    #     return_node_indices=True,
    # )

    # print(louvain_split)

    # louvain_split_label_imb = louvain_label_imbalance_split(
    #     global_data,
    #     num_clients=n_clients,
    #     resolution=louvain_res,
    #     seed=split_seeds["louvain_imbalance"],
    #     return_node_indices=True,
    # )


import sys, numpy, scipy, sknetwork

if __name__ == "__main__":
    main()