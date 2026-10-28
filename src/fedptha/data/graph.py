"""
This script provides utility functions for GNN data preprocessing, loss functions, and creating induced subgraphs based
on a chosen list of node indices
"""

from typing import Union
import logging, torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
from torch_geometric.transforms import BaseTransform


def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = (
        torch.zeros((data.edge_index.shape[1], 1))
        if data.timestamps is None
        else data.timestamps.reshape((-1, 1))
    )
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u, v, t in edges:
        u, v, t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out


def to_adj_edges_with_times(data):
    num_nodes = data.num_nodes
    timestamps = (
        torch.zeros((data.edge_index.shape[1], 1))
        if data.timestamps is None
        else data.timestamps.reshape((-1, 1))
    )
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u, v, t) in enumerate(edges):
        u, v, t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out


def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1:
            continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:, [0]], return_index=True, axis=0)
        nbs_unique = a[np.sort(idx)][:, 0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u, v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports


def time_deltas(data, adj_edges_list):
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1:
            continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i + 1, -1] - a[i, -1] for i in range(a.shape[0] - 1)]
        tds = np.hstack((a[:, 0].reshape(-1, 1), np.array(a_tds).reshape(-1, 1)))
        for i, td in tds:
            time_deltas[i] = td
    return time_deltas


class GraphData(Data):
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        readout: str = "node",
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = "ce"
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps
        elif edge_attr is not None:
            self.timestamps = edge_attr[:, 0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = (
            [ports(self.edge_index.flipud(), adj_list_out)] if reverse_ports else []
        )
        self.edge_attr = torch.cat([self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self

    def set_y(self, function, threshholds=None):
        if isinstance(function, list):
            if threshholds is None:
                threshholds = [None] * len(function)
            ys = []
            for i, (f, thresh) in enumerate(zip(function, threshholds)):
                if f is not None:
                    logging.info(f"Calculating feat {i}: {f.__name__}")
                    ys.append(f(self, thresh))
            y = torch.cat(ys, dim=1)
        else:
            y = function(self, threshholds)
        self.y = y
        if len(y.unique()) > 2:
            self.loss_fn = "mse"
        elif y.shape[1] > 1:
            self.loss_fn = "bce_multi"
        else:
            self.loss_fn = "ce"
        logging.info(f"y.shape = {y.shape}")
        logging.info(f"self.loss_fn = {self.loss_fn}")
        return y


class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch."""

    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        ids[: data.batch_size] = 1
        data.x = torch.cat([x, ids], dim=1)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
