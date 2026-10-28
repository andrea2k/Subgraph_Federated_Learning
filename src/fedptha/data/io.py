"""Load the original benchmark graphs and their train/validation/test splits."""

import pickle
import torch


class GraphUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils.gcn_utils" and name == "GraphData":
            from .graph import GraphData

            return GraphData
        return super().find_class(module, name)


class LegacyGraphPickle:
    __name__ = "pickle"
    Unpickler = GraphUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def load_graph(path):
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=LegacyGraphPickle
    )


def load_datasets(
    log_dir,
    train_data_file="train.pt",
    val_data_file="val.pt",
    test_data_file="test.pt",
):
    from fedptha.paths import absolute

    directory = absolute(log_dir)
    return tuple(
        load_graph(directory / name)
        for name in (train_data_file, val_data_file, test_data_file)
    )
