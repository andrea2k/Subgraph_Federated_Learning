import numpy as np
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
import pymetis as metis

from utils.fed_partitioning import get_subgraph_pyg_data, zipf_assign_communities_to_clients, equal_assign_communities_to_clients

"""
The implementation of the Label Imbalance Split (LIS) simulation strategy  
for the Metis-based and Louvain-based data partitioning techniques is adapted 
from the GitHub repository of the "OpenFGL: A Comprehensive Benchmark for 
Federated Graph Learning" paper by Li et al., (2024). 

In the original OpenFGL implementation, at the label distribution clustering step, 
both partitioning techniques assume that each node has a single class label. 
However, in the global graph generated for this work, each node has 11 binary labels.
To address this, the original code is modified to account for this multi-label scenario.
"""

def metis_original_split(global_data: Data,
                         num_clients: int,
                         metis_num_coms: int,
                         seed: int | None = None,
                         alpha: float = 1.2,
                         client_assignment: str = "zipf",
                         return_node_indices: bool = False):
    """
    Metis-based original subgraph-FL split (structure-only, no label handling)
    with Zipf-skewed client sizes.

    Supports two client-assignment strategies:
      - 'zipf'   : Zipf-skewed client sizes (realistic, heavy-tailed)
      - 'equal'  : Approximately equal-sized clients (controlled setting)
    """
    print(f"Conducting subgraph-FL Metis (original, {client_assignment}-assigned) simulation...")

    # convert to NetworkX and partition with Metis
    graph_nx = to_networkx(global_data, to_undirected=True)
    _, membership = metis.part_graph(metis_num_coms, graph_nx)
    membership = np.array(membership)  # shape [num_nodes], value = com_id

    # build structural communities
    communities: dict[int, list[int]] = {com_id: [] for com_id in range(metis_num_coms)}
    for node_id, com_id in enumerate(membership):
        com_id = int(com_id)
        communities[com_id].append(node_id)

    # assign communities to clients according to the chosen strategy
    if client_assignment == "zipf":
        client_indices = zipf_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )
    elif client_assignment == "equal":
        # alpha is ignored in the equal-sized case
        client_indices = equal_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown client_assignment='{client_assignment}'. Expected 'zipf' or 'equal'.")

    # if the user only wants the node indices, return them without computing local subgraphs
    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # else, build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        if local_subgraph.edge_index.numel() == 0:
            # TODO: if no edges, we can add random edges or leave as empty
            pass
        local_data.append(local_subgraph)

    return local_data


def louvain_original_split(global_data: Data,
                           num_clients: int,
                           resolution: float = 1.0,
                           seed: int | None = None,
                           alpha: float = 1.2,
                           client_assignment: str = "zipf",
                           return_node_indices: bool = False):
    """
    Louvain-based original subgraph-FL split (structure-only, no label handling)
    with Zipf-skewed client sizes.

    Supports two client-assignment strategies:
      - 'zipf'   : Zipf-skewed client sizes (realistic, heavy-tailed)
      - 'equal'  : Approximately equal-sized clients (controlled setting)
    """
    print(f"Conducting subgraph-FL Louvain (original, {client_assignment}-assigned) simulation...")

    num_nodes = global_data.num_nodes

    # Louvain communities on the adjacency
    adj_csr = to_scipy_sparse_matrix(global_data.edge_index, num_nodes=num_nodes)
    louvain = Louvain(
        modularity="newman",
        resolution=resolution,
        return_aggregate=True,
        random_state=seed,
    )
    com_assignments = louvain.fit_predict(adj_csr)  # community ID per node

    # build structural communities (no labels involved)
    communities: dict[int, list[int]] = {}
    for node_id, com_id in enumerate(com_assignments):
        com_id = int(com_id)
        if com_id not in communities:
            communities[com_id] = []
        communities[com_id].append(node_id)

    # assign communities to clients according to the chosen strategy
    if client_assignment == "zipf":
        client_indices = zipf_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )
    elif client_assignment == "equal":
        # alpha is ignored in the equal-sized case
        client_indices = equal_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown client_assignment='{client_assignment}'. Expected 'zipf' or 'equal'.")

    # if the user only wants the node indices, return them without computing local subgraphs
    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # else, build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        if local_subgraph.edge_index.numel() == 0:
            # TODO: if no edges, we can add random edges or leave as empty
            pass
        local_data.append(local_subgraph)

    return local_data


def metis_label_imbalance_split(global_data: Data,
                                num_clients: int,
                                metis_num_coms: int,
                                seed: int | None = None,
                                return_node_indices: bool = False):
    """
    Metis-based Label Imbalance Split for a single big graph
    with multi-task labels y in {0,1}^{N x T}.

    Args:
        global_data: PyG Data for the full graph.
        num_clients: number of federated clients.
        metis_num_coms: number of Metis communities (often > num_clients).

    Returns:
        If return_node_indices is False (default):
            list[Data]: one subgraph per client.
        If return_node_indices is True:
            list[list[int]]: node indices per client.
    """
    print("Conducting subgraph-FL metis+ (label-imbalance) simulation...")

    num_classes = global_data.num_classes

    # convert to NetworkX and partition with Metis
    graph_nx = to_networkx(global_data, to_undirected=True)
    _, membership = metis.part_graph(metis_num_coms, graph_nx)
    membership = np.array(membership)  # shape [num_nodes], value = com_id

    # build communities with label distributions (multi-task)
    communities = {
        com_id: {
            "nodes": [],
            "label_distribution": np.zeros(num_classes, dtype=float),
        }
        for com_id in range(metis_num_coms)
    }

    for node_id, com_id in enumerate(membership):
        com_id = int(com_id)
        communities[com_id]["nodes"].append(node_id)
        label_vec = global_data.y[node_id].cpu().numpy()
        communities[com_id]["label_distribution"] += label_vec

    num_communities = len(communities)

    # normalize and create clustering features
    clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    for com_id in communities.keys():
        dist = communities[com_id]["label_distribution"]
        total = dist.sum()
        if total > 0:
            dist = dist / total
        clustering_data[com_id, :] = dist

    # kMeans: communities -> clients
    kmeans = KMeans(n_clusters=num_clients, n_init="auto", random_state=seed)
    clustering_labels = kmeans.fit_predict(clustering_data)

    client_indices = {cid: [] for cid in range(num_clients)}
    for com_id in range(num_communities):
        client_id = int(clustering_labels[com_id])
        client_indices[client_id] += communities[com_id]["nodes"]

    # if the user only wants the node indices, return them without computing local subgraphs
    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # else, build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        if local_subgraph.edge_index.numel() == 0:
            # TODO: if no edges, we can add random edges or leave as empty
            pass
        local_data.append(local_subgraph)

    return local_data


def louvain_label_imbalance_split(global_data: Data,
                                  num_clients: int,
                                  resolution: float = 1.0,
                                  seed: int | None = None,
                                  return_node_indices: bool = False):
    """
    Louvain-based Label Imbalance Split for a single big graph
    with multi-task labels y in {0,1}^{N x T}.

    Returns:
        If return_node_indices is False (default):
            list[Data]: one subgraph per client.
        If return_node_indices is True:
            list[list[int]]: node indices per client.
    """
    print("Conducting subgraph-FL louvain+ (label-imbalance) simulation...")

    num_nodes = global_data.num_nodes
    num_classes = global_data.num_classes  # == num_tasks

    # louvain communities on the adjacency
    adj_csr = to_scipy_sparse_matrix(global_data.edge_index, num_nodes=num_nodes)
    louvain = Louvain(modularity="newman",
                      resolution=resolution,
                      return_aggregate=True,
                      random_state=seed)
    com_assignments = louvain.fit_predict(adj_csr)  # community ID per node

    # build per-community label distributions (vectors of length num_classes)
    communities = {}
    for node_id, com_id in enumerate(com_assignments):
        com_id = int(com_id)
        if com_id not in communities:
            communities[com_id] = {
                "nodes": [],
                "label_distribution": np.zeros(num_classes, dtype=float),
            }
        communities[com_id]["nodes"].append(node_id)
        # accumulate the label vector (multi-task)
        label_vec = global_data.y[node_id].cpu().numpy()  # shape [num_classes]
        communities[com_id]["label_distribution"] += label_vec

    num_communities = len(communities)

    # normalize label distributions and create clustering features
    clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    for com_id in communities.keys():
        dist = communities[com_id]["label_distribution"]
        total = dist.sum()
        if total > 0:
            dist = dist / total
        clustering_data[com_id, :] = dist

    # kMeans over communities by label distribution
    kmeans = KMeans(n_clusters=num_clients, n_init="auto", random_state=seed)
    clustering_labels = kmeans.fit_predict(clustering_data)  # community -> client

    # aggregate communities into clients
    client_indices = {cid: [] for cid in range(num_clients)}
    for com_id in range(num_communities):
        client_id = int(clustering_labels[com_id])
        client_indices[client_id] += communities[com_id]["nodes"]

    # if the user only wants the node indices, return them without computing local subgraphs
    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # else, build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        if local_subgraph.edge_index.numel() == 0:
            # TODO: if no edges, we can add random edges or leave as empty
            pass
        local_data.append(local_subgraph)

    return local_data
