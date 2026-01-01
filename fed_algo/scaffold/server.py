# fed_algo/scaffold/server.py
import torch
from fed_algo.base import BaseServer

class ScaffoldServer(BaseServer):
    """
    ScaffoldServer implements the server-side logic for the SCAFFOLD algorithm in Federated Learning.
    SCAFFOLD aims to reduce the variance caused by client drift by introducing control variates (local and global control variables)
    that adjust the client updates during training.

    Attributes:
        global_control (list[torch.Tensor]): A list of tensors representing the global control variates for each parameter in the model.
    """
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the ScaffoldServer with the provided arguments, global data, and device.
        
        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): The global dataset used for training (if applicable).
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the server and the clients.
            device (torch.device): Device to run the computations on (CPU or GPU).
        """
        super().__init__(args, global_data, data_dir, message_pool, device)
        self.global_control = [torch.zeros_like(p.data, device=self.device) for p in self.task.model.parameters()]

    @torch.no_grad()
    def update_global_control(self):
        """
        Updates the global control variates by averaging the local control variates from the sampled clients.
        This step is crucial for mitigating the variance caused by client drift in the federated learning process.
        """
        sampled_clients = self.message_pool["sampled_clients"]
        m = len(sampled_clients)

        # OpenFGL-style: global_control = average(local_controls)
        new_global = [torch.zeros_like(c) for c in self.global_control]
        for cid in sampled_clients:
            local_controls = self.message_pool[f"client_{cid}"]["local_control"]
            for i, lc in enumerate(local_controls):
                new_global[i].add_(lc.to(self.device) / m)

        for i in range(len(self.global_control)):
            self.global_control[i].data.copy_(new_global[i].data)

    def execute(self):
        """
        Executes the aggregation of client updates by averaging the local model parameters from the sampled clients.
        It also updates the global control variates based on the local control variates received from the clients.
        """
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot_samples = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled_clients)

            for it, cid in enumerate(sampled_clients):
                w = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot_samples
                client_weights = self.message_pool[f"client_{cid}"]["weight"]

                for local_w, global_param in zip(client_weights, self.task.model.parameters()):
                    local_w = local_w.to(self.device)
                    if it == 0:
                        global_param.data.copy_(w * local_w)
                    else:
                        global_param.data.add_(w * local_w)

        self.update_global_control()

    def send_message(self):
        """
        Sends the updated global model parameters and global control variates to the clients after the aggregation step.
        This information is used by the clients to adjust their local updates in the next round.
        """
        self.message_pool["server"] = {
            "weight": [p.data.detach().cpu().clone() for p in self.task.model.parameters()],
            "global_control": [c.detach().cpu().clone() for c in self.global_control],
        }
