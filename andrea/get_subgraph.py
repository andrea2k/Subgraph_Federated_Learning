import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def get_subgraph_pyg_data(global_data: Data, node_list):
    node_idx = torch.as_tensor(node_list, dtype=torch.long, device=global_data.edge_index.device)

    sub_edge_index, sub_edge_attr = subgraph(
        subset=node_idx,
        edge_index=global_data.edge_index,
        edge_attr=getattr(global_data, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=global_data.num_nodes,
    )

    sub_data = Data()
    sub_data.edge_index = sub_edge_index
    if sub_edge_attr is not None:
        sub_data.edge_attr = sub_edge_attr

    if getattr(global_data, "x", None) is not None:
        sub_data.x = global_data.x[node_idx]
    if getattr(global_data, "y", None) is not None:
        sub_data.y = global_data.y[node_idx]

    sub_data.num_nodes = node_idx.numel()

    # local id -> global id (as a tensor; faster than dict)
    sub_data.global_map = node_idx.detach().cpu()

    return sub_data