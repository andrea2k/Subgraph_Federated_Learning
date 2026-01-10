#!/usr/bin/env python3
from torch_geometric.data import HeteroData
import torch

def make_bidirected_hetero(data, *, keep_node_x=True, keep_node_y=True):
    """
    Convert a homogeneous, directed Data into a HeteroData with:
      node type: 'n'
      edge types:
        ('n','fwd','n'): original edges (u -> v)
        ('n','rev','n'): reversed edges (v -> u)

    Edge attributes:
      Expects data.edge_attr[..., -2:] == [in_port, out_port] (LONG).
      For 'fwd' we keep [in_port, out_port] as-is.
      For 'rev' we SWAP them.
    """
    assert hasattr(data, "edge_index"), "Data must have an edge_index"

    hd = HeteroData()
    num_nodes = data.num_nodes

    # node features / labels
    hd['n'].num_nodes = num_nodes

    # Copy node features/labels/masks
    if keep_node_x and getattr(data, 'x', None) is not None:
        hd['n'].x = data.x
    if keep_node_y and getattr(data, 'y', None) is not None:
        hd['n'].y = data.y
    for key in ['train_mask', 'val_mask', 'test_mask']:
        if getattr(data, key, None) is not None:
            hd['n'][key] = getattr(data, key)

    # Preserve federated ghost/owned metadata on nodes
    if hasattr(data, "owned_mask") and data.owned_mask is not None:
        # ensure bool dtype (some pipelines may store uint8/long)
        hd['n'].owned_mask = data.owned_mask.bool()
    if hasattr(data, "global_nid") and data.global_nid is not None:
        hd['n'].global_nid = data.global_nid.long()
    if hasattr(data, "client_id"):
        # store as a python int attribute; optional
        hd['n'].client_id = int(getattr(data, "client_id"))

    # Edges
    ei = data.edge_index
    hd[('n','fwd','n')].edge_index = ei                         # u -> v
    hd[('n','rev','n')].edge_index = torch.flip(ei, dims=[0])   # v -> u

    # Ports
    if getattr(data, 'edge_attr', None) is None:
        raise ValueError("Expected data.edge_attr with [.., in_port, out_port] appended. Got None.")

    E, F = data.edge_attr.shape
    if F < 2:
        raise ValueError(f"edge_attr must contain at least 2 columns for ports; got shape {data.edge_attr.shape}.")

    in_col, out_col = F - 2, F - 1
    in_ports  = data.edge_attr[:, in_col].long()   # [E]
    out_ports = data.edge_attr[:, out_col].long()  # [E]

    # Forward relation keeps [in_port, out_port] as-is:
    fwd_edge_attr = torch.stack([in_ports, out_ports], dim=-1)  # [E, 2] longs
    hd[('n','fwd','n')].edge_attr = fwd_edge_attr

    # Reverse relation swaps them:
    rev_edge_attr = torch.stack([out_ports, in_ports], dim=-1)  # [E, 2] longs
    hd[('n','rev','n')].edge_attr = rev_edge_attr

    return hd
