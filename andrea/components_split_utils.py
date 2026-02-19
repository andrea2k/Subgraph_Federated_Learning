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

"""
components_split_utils.py

Client split utilities for "component-based" Subgraph-FL simulations.

High-level flow:
1) Compute motif witnesses and node labels y using set_y_and_get_motifs (TASK_FUNCS defines motifs).
2) Optionally filter edges down to motif edges only ("motifs-only graph").
3) Compute connected components (treating the motifs-only graph as undirected).
4) Assign components (communities) to clients via:
   - zipf: imbalanced assignment controlled by alpha
   - equal: roughly equal sized assignment
5) Build per-client induced PyG subgraphs using get_subgraph_pyg_data.

Two variants:
- components_original_split: assigns components directly (optionally zipf/equal).
- components_label_imbalance_split: clusters components by label distribution, then assigns clusters to clients using k-means.
"""

def motifs_only_src_dst(motifs):
    """
    Convert motif witnesses into explicit directed edge lists (src, dst) to protect/keep.

    This bridges motif witness tuples (cycles, scatter-gather, biclique) into the specific
    directed edges that realize the motif structure.

    Args:
        motifs: Dict[str, List[tuple]] mapping motif name -> list of witness tuples.

    Returns:
        protected_src: List[int] of source node IDs for protected edges.
        protected_dst: List[int] of destination node IDs for protected edges.
    """
    protected_src = []
    protected_dst = []

    for name, motif_list in motifs.items():
        if name == "cycle2":
            # Each witness is (u, v), representing u<->v, keep both directed edges
            for (u, v) in motif_list:
                protected_src += [u, v]
                protected_dst += [v, u]

        elif name in {"cycle3", "cycle4", "cycle5", "cycle6"}:
            # Each witness is an ordered cycle (n0, n1, ..., nk-1); keep edges ni -> n(i+1)
            for cyc in motif_list:
                k = len(cyc)
                for i in range(k):
                    protected_src.append(cyc[i])
                    protected_dst.append(cyc[(i + 1) % k])

        elif name == "scatter_gather":
            # Witness: (source, j1, j2, sink) with edges:
            # source->j1, source->j2, j1->sink, j2->sink
            for (source, j1, j2, sink) in motif_list:
                protected_src += [source, source, j1, j2]
                protected_dst += [j1,     j2,     sink, sink]

        elif name == "biclique":
            # Witness: (l1, l2, r1, r2) with edges:
            # l1->r1, l1->r2, l2->r1, l2->r2
            for (l1, l2, r1, r2) in motif_list:
                protected_src += [l1, l1, l2, l2]
                protected_dst += [r1, r2, r1, r2]

    return protected_src, protected_dst

def motifs_only_filter(num_nodes, edge_index, protected_src, protected_dst):
    """
    Build a boolean mask selecting only the motif-protected edges from edge_index.

    Implementation detail:
        Hash each directed edge (u->v) as u*num_nodes + v, then check membership against
        the set of protected edge hashes.

    Args:
        num_nodes: Number of nodes in the graph (used for hashing).
        edge_index: Tensor [2, E] of directed edges.
        protected_src: List[int] sources of protected edges.
        protected_dst: List[int] destinations of protected edges.

    Returns:
        protected_mask: Boolean tensor of shape [E] where True means keep this edge.
    """
    src = edge_index[0]
    dst = edge_index[1]

    edge_hash = src * num_nodes + dst

    prot_src_t = torch.tensor(protected_src, dtype=src.dtype, device=edge_hash.device)
    prot_dst_t = torch.tensor(protected_dst, dtype=dst.dtype, device=edge_hash.device)
    prot_hash = (prot_src_t * num_nodes + prot_dst_t).unique()

    protected_mask = torch.isin(edge_hash, prot_hash)

    return protected_mask

def motifs_only_graph(graph_data, motifs_filter):
    """
    Create a shallow clone of a PyG Data object containing only motif edges.

    Notes:
        - Node attributes (x, y, etc.) are preserved.
        - edge_index is filtered by motifs_filter.
        - edge_attr is filtered if present.

    Args:
        graph_data: PyG Data object.
        motifs_filter: Boolean mask over edges (shape [E]).

    Returns:
        new_graph: PyG Data object with filtered edges.
    """
    new_graph = graph_data.clone()
    new_graph.edge_index = graph_data.edge_index[:, motifs_filter]
    if getattr(graph_data, "edge_attr", None) is not None:
        new_graph.edge_attr = graph_data.edge_attr[motifs_filter]
    return new_graph

def components_original_split(
    global_data: Data,
    num_clients: int,
    seed: int | None = None,
    alpha: float = 1.2,
    client_assignment: str = "zipf",
    return_node_indices: bool = False,
):
    """
    Split a global graph into client subgraphs by connected components of the motifs-only graph.

    Steps:
        1) Compute motif labels/witnesses on the full graph.
        2) Filter down to motif edges only.
        3) Compute connected components on the motifs-only graph (undirected view).
        4) Assign components to clients using zipf or equal assignment.
        5) Return either node indices per client, or actual per-client PyG subgraphs.

    Args:
        global_data: Global PyG Data.
        num_clients: Number of clients to create.
        seed: RNG seed used inside assignment utilities.
        alpha: Zipf imbalance parameter (ignored for equal assignment).
        client_assignment: "zipf" or "equal".
        return_node_indices: If True, return node ID lists instead of subgraph Data objects.

    Returns:
        If return_node_indices:
            List[List[int]]: sorted node indices per client.
        Else:
            List[Data]: per-client induced subgraphs taken from global_data.
    """
    print(f"Conducting subgraph-FL Component (original, {client_assignment}-assigned) simulation...")

    # Compute motif-based multi-label y and witness tuples on the full graph
    global_data, motifs = set_y_and_get_motifs(global_data, TASK_FUNCS)

    # Convert witnesses into the directed edges that realize those motifs
    motifs_src, motifs_dst = motifs_only_src_dst(motifs)
    motifs_filter = motifs_only_filter(global_data.num_nodes, global_data.edge_index, motifs_src, motifs_dst)

    # Build a graph with only motif edges, then recompute y on that filtered graph for consistency
    new_graph = motifs_only_graph(global_data, motifs_filter)
    new_graph, motifs = set_y_and_get_motifs(new_graph, TASK_FUNCS)

    # Connected components are computed on an undirected view of the motifs-only graph
    G = to_networkx(new_graph, to_undirected=True)

    communities = {}
    for com_id, node_ids_set in enumerate(nx.connected_components(G)):
        communities[com_id] = list(node_ids_set)

    # Assign communities to clients according to chosen strategy
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
    
    # Build per-client induced subgraphs (on the ORIGINAL global_data node set)
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)

        # If a subgraph has no edges, we keep it as-is (caller may choose to handle this)
        if local_subgraph.edge_index.numel() == 0:
            pass

        local_data.append(local_subgraph)

    return local_data

def components_label_imbalance_split(
    global_data: Data,
    num_clients: int,
    seed: int | None = None,
    return_node_indices: bool = False,
):
    """
    Split by connected components, but *cluster components by label distribution* to induce label imbalance.

    Steps:
        1) Compute motifs and filter to a motifs-only graph (same as components_original_split).
        2) Compute connected components.
        3) For each component, compute its label distribution from node multi-label y.
        4) Run KMeans over component label distributions (k = min(num_clients, num_components)).
        5) Assign all components in a cluster to the same client ID.

    Args:
        global_data: Global PyG Data.
        num_clients: Number of clients.
        seed: Random seed (used by KMeans and potentially other utilities).
        return_node_indices: If True, return node ID lists instead of Data objects.

    Returns:
        If return_node_indices:
            List[List[int]]: sorted node indices per client.
        Else:
            List[Data]: per-client induced subgraphs taken from the motifs-only graph.
    """
    print(f"Conducting subgraph-FL Component (label-imbalance) simulation...")

    # Compute motif-based labels and witness tuples
    global_data, motifs = set_y_and_get_motifs(global_data, TASK_FUNCS)

    # Build motifs-only graph
    motifs_src, motifs_dst = motifs_only_src_dst(motifs)
    motifs_filter = motifs_only_filter(global_data.num_nodes, global_data.edge_index, motifs_src, motifs_dst)
    new_graph = motifs_only_graph(global_data, motifs_filter)

    # Recompute y on the motifs-only graph (ensures y matches filtered structure)
    new_graph, _ = set_y_and_get_motifs(new_graph, TASK_FUNCS)

    num_classes = new_graph.num_classes

    # Connected components (undirected view)
    G = to_networkx(new_graph, to_undirected=True)
    comp_list = list(nx.connected_components(G))
    num_communities = len(comp_list)

    # Build a per-community label count vector by summing node multi-label vectors
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

    # Normalize to distributions for clustering      
    clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    for com_id in range(num_communities):
        dist = communities[com_id]["label_distribution"]
        total = dist.sum()
        if total > 0:
            dist = dist / total
        clustering_data[com_id, :] = dist

    # Cluster components; if fewer components than clients, reduce k accordingly
    k = min(num_clients, num_communities)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    clustering_labels = kmeans.fit_predict(clustering_data)

    # Assign communities to clients by cluster ID (cluster IDs in [0, k-1])
    client_indices = {cid: [] for cid in range(num_clients)}
    for com_id in range(num_communities):
        client_id = int(clustering_labels[com_id])
        client_indices[client_id] += communities[com_id]["nodes"]

    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]
    
    # Build per-client subgraphs from the motifs-only graph
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(new_graph, node_list)
        local_data.append(local_subgraph)

    return local_data