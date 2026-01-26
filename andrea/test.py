import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import csv
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from andrea.multigraph_generation import (
    DATA_ROOT, 
    GRAPH_PARAM_CSV, 
    TASK_FUNCS, 
    TASKS, 
    set_y_and_count_motifs, 
)

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

def plot_pyg_data(
    data,
    title="graph",
    original_node_ids=None,   # list/1D tensor length = data.num_nodes (maps local->global)
    show_edge_attr=True,
    seed=42
):
    """
    Plot a torch_geometric.data.Data using NetworkX.
    - If original_node_ids is provided, nodes are labeled "global_id (y=..)".
      Otherwise nodes are labeled "local_id (y=..)".
    - Draws edge_attr as labels if present and show_edge_attr=True.
    """
    # Convert to NetworkX
    G = to_networkx(
        data,
        to_undirected=False,     # keep direction
        remove_self_loops=True
    )

    # Layout
    pos = nx.spring_layout(G, seed=seed)

    # Node labels
    labels = {}
    y = data.y.detach().cpu().tolist() if getattr(data, "y", None) is not None else [None] * data.num_nodes

    for n in G.nodes():
        if original_node_ids is None:
            labels[n] = f"{n}\n(y={y[n]})"
        else:
            gid = int(original_node_ids[n])
            labels[n] = f"{gid}\n(y={y[n]})"

    # Draw
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # Edge labels (edge_attr)
    if show_edge_attr and getattr(data, "edge_attr", None) is not None:
        # edge_attr is aligned with data.edge_index order. Build mapping (u,v)->value
        edge_attr = data.edge_attr.detach().cpu()
        edge_index = data.edge_index.detach().cpu()

        edge_labels = {}
        for k in range(edge_index.size(1)):
            u = int(edge_index[0, k])
            v = int(edge_index[1, k])
            val = edge_attr[k].view(-1).tolist()
            # if scalar, show just the number
            edge_labels[(u, v)] = val[0] if len(val) == 1 else val

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
def get_subgraph_pyg_data(global_data: Data, node_list):
    """
    Extract an induced subgraph from `global_data` on the given `node_list`.

    - Relabels node indices to be 0..(sub_num_nodes-1)
    - Slices x, y, edge_attr consistently
    """

    node_idx = torch.tensor(node_list, dtype=torch.long)

    # extract the subgraph and relabel nodes to be 0..(sub_num_nodes-1)
    sub_edge_index, sub_edge_attr = subgraph(
        node_idx,
        global_data.edge_index,
        edge_attr=getattr(global_data, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=global_data.num_nodes,
    )

    # build new data object
    sub_data = Data()
    sub_data.edge_index = sub_edge_index

    if sub_edge_attr is not None:
        sub_data.edge_attr = sub_edge_attr

    # x / y sliced in the same order as node_idx
    if hasattr(global_data, "x") and global_data.x is not None:
        sub_data.x = global_data.x[node_idx]

    if hasattr(global_data, "y") and global_data.y is not None:
        sub_data.y = global_data.y[node_idx]

    # explicitly set num_nodes to be safe
    sub_data.num_nodes = node_idx.numel()

    return sub_data

num_nodes = 5

# Directed complete graph (no self-loops): all i->j for i!=j
src, dst = [], []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and j > i:
            src.append(i)
            dst.append(j)

edge_index = torch.tensor([src, dst], dtype=torch.long)  # shape [2, 20]
edge_attr  = torch.ones(edge_index.size(1), 1)           # all weights = 1, shape [20, 1]

x = torch.ones(num_nodes, 1)                             # node features = 1, shape [5, 1]
y = 7 * torch.arange(num_nodes)                          # y[i] = 7*i, shape [5]

global_data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y, num_nodes=num_nodes)

_, counts = set_y_and_count_motifs(global_data,TASK_FUNCS)

plot_pyg_data(global_data, title="global_data")

print(counts)
print(global_data.y)

node_list = [1, 3, 4]   # choose any subset you like
sub_data = get_subgraph_pyg_data(global_data, node_list)

print(sub_data)
print(sub_data.y)

_, counts = set_y_and_count_motifs(sub_data,TASK_FUNCS)

plot_pyg_data(sub_data, title="subgraph nodes [1,3,4]", original_node_ids=node_list)

print(counts)
# node_list = [0, 2]   # choose any subset you like
# sub_data = get_subgraph_pyg_data(global_data, node_list)

# plot_pyg_data(sub_data, title="subgraph nodes [0,2]", original_node_ids=node_list)

# _, counts = set_y_and_count_motifs(sub_data,TASK_FUNCS)
# print(counts)
# print(sub_data.y)