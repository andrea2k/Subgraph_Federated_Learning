# flcore/fedprox/client.py
import torch
from fed_algo.base import BaseClient

class FedProxClient(BaseClient):
    """
    FedProxClient is a client implementation for the Federated Proximal (FedProx) framework, 
    introduced in the paper "Federated Optimization in Heterogeneous Networks." This client 
    handles local training with a custom loss function that includes a proximal term, 
    designed to address the challenges of heterogeneity in federated learning environments.

    Attributes:
        None
    """
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedProxClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedProxClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
    def execute(self):
        """
        1) Sync local model with server weights
        2) Cache a frozen snapshot of global parameters (for proximal term)
        3) Train locally with FedProx penalty
        """
        # 1) Sync local model with global model
        with torch.no_grad():
            global_weights = self.message_pool["server"]["weight"]
            for local_param, global_param in zip(self.task.model.parameters(), global_weights):
                local_param.data.copy_(global_param.to(self.device))

        # 2) Snapshot global params for proximal penalty (detach + clone)
        global_params = [p.detach().clone() for p in self.task.model.parameters()]

        # 3) Local training with FedProx
        mu = float(getattr(self.args, "fedprox_mu", 1e-3))
        self.task.train(global_params=global_params, fedprox_mu=mu)

    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": [p.data.detach().cpu().clone() for p in self.task.model.parameters()],
        }
