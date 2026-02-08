import numpy as np
import torch

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from utils.fed_partitioning import zipf_assign_communities_to_clients, equal_assign_communities_to_clients
from andrea.get_subgraph import get_subgraph_pyg_data
from andrea.multigraph_generation import (
    TASK_FUNCS, 
    set_y_and_get_motifs, 
)
from sklearn.cluster import KMeans

def motifs_only_src_dst(motifs):
    protected_src = []
    protected_dst = []

    for name, motif_list in motifs.items():
        if name == "cycle2":
            for (u, v) in motif_list:
                protected_src += [u, v]
                protected_dst += [v, u]

        elif name in {"cycle3", "cycle4", "cycle5", "cycle6"}:
            for cyc in motif_list:
                k = len(cyc)
                for i in range(k):
                    protected_src.append(cyc[i])
                    protected_dst.append(cyc[(i + 1) % k])

        elif name == "scatter_gather":
            # (source, j1, j2, sink)
            for (source, j1, j2, sink) in motif_list:
                protected_src += [source, source, j1, j2]
                protected_dst += [j1,     j2,     sink, sink]

        elif name == "biclique":
            # (l1, l2, r1, r2)
            for (l1, l2, r1, r2) in motif_list:
                protected_src += [l1, l1, l2, l2]
                protected_dst += [r1, r2, r1, r2]

    return protected_src, protected_dst

def motifs_only_filter(num_nodes, edge_index, protected_src, protected_dst):
    src = edge_index[0]
    dst = edge_index[1]

    edge_hash = src * num_nodes + dst

    prot_src_t = torch.tensor(protected_src, dtype=src.dtype, device=edge_hash.device)
    prot_dst_t = torch.tensor(protected_dst, dtype=dst.dtype, device=edge_hash.device)
    prot_hash = (prot_src_t * num_nodes + prot_dst_t).unique()

    protected_mask = torch.isin(edge_hash, prot_hash)

    return protected_mask

def motifs_only_graph(graph_data, motifs_filter):
    new_graph = graph_data.clone()
    new_graph.edge_index = graph_data.edge_index[:, motifs_filter]
    if getattr(graph_data, "edge_attr", None) is not None:
        new_graph.edge_attr = graph_data.edge_attr[motifs_filter]
    return new_graph

def components_original_split(global_data: Data,
                         num_clients: int,
                         seed: int | None = None,
                         alpha: float = 1.2,
                         client_assignment: str = "zipf",
                         return_node_indices: bool = False):

    global_data, motifs = set_y_and_get_motifs(global_data, TASK_FUNCS)
    motifs_src, motifs_dst = motifs_only_src_dst(motifs)
    motifs_filter = motifs_only_filter(global_data.num_nodes, global_data.edge_index, motifs_src, motifs_dst)

    new_graph = motifs_only_graph(global_data, motifs_filter)
    new_graph, motifs = set_y_and_get_motifs(new_graph, TASK_FUNCS)

    G = to_networkx(new_graph, to_undirected=True)

    communities = {}
    for com_id, node_ids_set in enumerate(nx.connected_components(G)):
        communities[com_id] = list(node_ids_set)

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

def components_label_imbalance_split(
                        global_data: Data,
                        num_clients: int,
                        seed: int | None = None,
                        return_node_indices: bool = False,):
    
    global_data, motifs = set_y_and_get_motifs(global_data, TASK_FUNCS)
    motifs_src, motifs_dst = motifs_only_src_dst(motifs)
    motifs_filter = motifs_only_filter(global_data.num_nodes, global_data.edge_index, motifs_src, motifs_dst)

    new_graph = motifs_only_graph(global_data, motifs_filter)
    new_graph, _ = set_y_and_get_motifs(new_graph, TASK_FUNCS)

    num_classes = new_graph.num_classes
    G = to_networkx(new_graph, to_undirected=True)
    comp_list = list(nx.connected_components(G))
    num_communities = len(comp_list)

    communities = {
        com_id: {
            "nodes": list(comp_nodes),
            "label_distribution": np.zeros(num_classes, dtype=float),
        }
        for com_id, comp_nodes in enumerate(comp_list)
    }

    for com_id in range(num_communities):
        for node_id in communities[com_id]["nodes"]:
            label_vec = new_graph.y[node_id].cpu().numpy()
            communities[com_id]["label_distribution"] += label_vec
        
    clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    for com_id in range(num_communities):
        dist = communities[com_id]["label_distribution"]
        total = dist.sum()
        if total > 0:
            dist = dist / total
        clustering_data[com_id, :] = dist
    k = min(num_clients, num_communities)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    clustering_labels = kmeans.fit_predict(clustering_data)

    client_indices = {cid: [] for cid in range(num_clients)}
    for com_id in range(num_communities):
        client_id = int(clustering_labels[com_id])
        client_indices[client_id] += communities[com_id]["nodes"]

    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]
    
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(new_graph, node_list)
        local_data.append(local_subgraph)

    return local_data