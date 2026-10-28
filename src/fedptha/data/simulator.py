"""Random circulant multigraph construction used by the thesis benchmark."""

import logging
import random
import networkx as nx
import numpy as np
import torch
from .graph import GraphData


class GraphSimulator(object):
    def __init__(
        self,
        num_nodes: int,
        avg_degree: int = None,
        num_edges: int = None,
        max_time: int = None,
        network_type: str = "type1",
        readout: str = "node",
        node_feats=False,
        bidirectional=False,
        delta=None,
        num_graphs: int = None,
        generator: str = "chordal",
        seed: int = None,
    ):
        self.seed = seed

        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        self.torch_gen = torch.Generator().manual_seed(seed)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        # some default values
        self.num_edges = (
            int(num_nodes * avg_degree / 2) if num_edges is None else num_edges
        )
        self.avg_degree = (
            (num_edges / num_nodes) * 2 if avg_degree is None else avg_degree
        )
        self.max_time = num_nodes if max_time is None else max_time
        self.network_type = network_type
        self.readout = readout
        self.node_feats = node_feats
        self.bidirectional = bidirectional
        self.delta = delta
        if generator == "barabasi":
            self.generator = self.barabasi_albert
        elif generator == "chordal":
            self.generator = self.random_chordal_graph
        elif generator == "random":
            self.generator = self.random_generator
        elif generator == "watts":
            self.generator = self.watts_strogatz_graph

    def random_chordal_graph(self, i=0):
        avg_degree = self.avg_degree
        if i > 0:
            avg_degree = np.random.poisson(avg_degree)
        logging.info(f"Component {i} avg_degree = {avg_degree}")
        delta = self.delta if self.delta is not None else avg_degree * 0.5
        num_edges = int(self.num_nodes * avg_degree / 2)
        src = torch.randint(
            0, self.num_nodes, (1, 2 * num_edges), generator=self.torch_gen
        )
        dst = (
            torch.normal(
                mean=src.float(),
                std=torch.full_like(src.float(), float(delta)),
                generator=self.torch_gen,
            )
            .round()
            .long()
            % self.num_nodes
        )
        edge_index = torch.cat([src, dst], dim=0)
        edge_index = edge_index[:, torch.where(edge_index[0] != edge_index[1])[0]]
        while edge_index.shape[1] < num_edges:
            src = torch.randint(
                0, self.num_nodes, (1, 2 * num_edges), generator=self.torch_gen
            )
            dst = (
                torch.normal(
                    mean=src.float(),
                    std=torch.full_like(src.float(), float(delta)),
                    generator=self.torch_gen,
                )
                .round()
                .long()
                % self.num_nodes
            )

            edge_index_tmp = torch.cat([src, dst], dim=0)
            edge_index_tmp = edge_index_tmp[
                :, torch.where(edge_index_tmp[0] != edge_index_tmp[1])[0]
            ]
            edge_index = torch.cat([edge_index, edge_index_tmp], dim=1)

        edge_index = edge_index[:, :num_edges]
        return edge_index

    def watts_strogatz_graph(self, i=0, p=None):
        n = self.num_nodes
        avg_degree = self.avg_degree
        if i > 0:
            avg_degree = self.np_rng.poisson(avg_degree)
        k = round((avg_degree + 0.5))
        logging.info(f"Component {i} avg_degree = {k}")
        d = self.delta
        if p is None:
            p = (8 * d - k) / (n - k)
        p = max(0, min(1, p))
        nx_graph = nx.watts_strogatz_graph(n, k, p, seed=self.seed)
        nx_graph = nx.DiGraph(
            [tuple(self.py_rng.sample(list(e), 2)) for e in nx_graph.edges()]
        )
        data = from_networkx(nx_graph)
        edge_index = data.edge_index
        return edge_index

    def erdos_renyi_graph(self, i=0):
        edge_index = torch.randint(
            0, self.num_nodes, (2, 2 * self.num_edges), generator=self.torch_gen
        )
        edge_index = edge_index[:, torch.where(edge_index[0] != edge_index[1])[0]]
        while edge_index.shape[1] < self.num_edges:
            edge_index_tmp = torch.randint(
                0, self.num_nodes, (2, 2 * self.num_edges), generator=self.torch_gen
            )
            edge_index_tmp = edge_index_tmp[:, edge_index_tmp[0] != edge_index_tmp[1]]
            edge_index = torch.cat([edge_index, edge_index_tmp], dim=1)

        edge_index = edge_index[:, : self.num_edges]
        return edge_index

    def barabasi_albert(self, i=0):
        n = self.num_nodes
        m = round(self.avg_degree / 2)
        if i > 0:
            m = self.np_rng.poisson(m)
        print(self.avg_degree)
        print(m)
        logging.info(f"Component {i} avg_degree = {m}")
        nx_graph = nx.barabasi_albert_graph(n, m, seed=self.seed)
        nx_graph = nx.DiGraph(
            [tuple(self.py_rng.sample(list(e), 2)) for e in nx_graph.edges()]
        )
        data = from_networkx(nx_graph)
        edge_index = data.edge_index
        return edge_index

    def random_generator(self, i=0):
        generators = [self.random_chordal_graph, self.barabasi_albert]
        generator = generators[int(self.np_rng.integers(0, len(generators)))]
        logging.info(f"generator = {generator.__name__}")
        return generator(i)

    def generate_edge_indices(self):
        edge_index = torch.zeros((2, 0)).long()
        max_node = torch.Tensor([0]).long()
        for i in range(self.num_graphs):
            edge_index_tmp = self.generator(i)
            edge_index = torch.cat((edge_index, edge_index_tmp + max_node), dim=1)
            max_node = edge_index.max() + 1
        return edge_index

    def generate_pytorch_graph(self):
        edge_index = self.generate_edge_indices()
        total_nodes = int(self.num_nodes)
        total_edges = int(edge_index.shape[1])
        timestamps = torch.randint(
            0, self.max_time, (total_edges, 1), generator=self.torch_gen
        )
        timestamps2 = torch.randint(
            0, self.max_time, (total_edges, 1), generator=self.torch_gen
        )
        if self.bidirectional:
            edge_index = torch.cat([edge_index, edge_index.flipud()], dim=1)
            timestamps = torch.cat([timestamps, timestamps], dim=0)
            timestamps2 = torch.cat([timestamps2, timestamps2], dim=0)
        # edge_attr = timestamps.float()
        edge_attr = torch.cat(
            [timestamps.reshape((-1, 1)).float(), timestamps2.reshape((-1, 1)).float()],
            dim=1,
        )
        num_samples = total_nodes if self.readout == "node" else total_edges
        x = torch.ones((total_nodes, 1))
        y = torch.zeros((num_samples, 1)).long()
        return GraphData(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            readout=self.readout,
            num_nodes=total_nodes,
        )
