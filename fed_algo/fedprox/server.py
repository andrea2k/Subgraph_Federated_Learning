# flcore/fedprox/server.py
import torch
from fed_algo.base import BaseServer

class FedProxServer(BaseServer):
    """
    FedProxServer is a server implementation for the Federated Proximal (FedProx) framework, 
    introduced in the paper "Federated Optimization in Heterogeneous Networks." This server 
    is responsible for aggregating model parameters from multiple clients and updating the 
    global model based on these aggregated parameters.

    Attributes:
        None
    """
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedProxServer.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedProxServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        """
        Executes the server-side operations for aggregating model parameters from clients. 
        The global model is updated as a weighted average of the model parameters from 
        sampled clients, where the weights are proportional to the number of samples 
        each client used during training.
        """
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot_samples = sum(
                self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled_clients
            )

            for it, cid in enumerate(sampled_clients):
                weight = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot_samples

                client_weights = self.message_pool[f"client_{cid}"]["weight"]
                for local_w, global_param in zip(client_weights, self.task.model.parameters()):
                    local_w = local_w.to(self.device)
                    if it == 0:
                        global_param.data.copy_(weight * local_w)
                    else:
                        global_param.data.add_(weight * local_w)
        
    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters. 
        This information is used by the clients to synchronize their local models with the global model.

        The message includes:
            - weight: The updated global model parameters.
        """
        self.message_pool["server"] = {
            "weight": [p.data.detach().cpu().clone() for p in self.task.model.parameters()]
        }