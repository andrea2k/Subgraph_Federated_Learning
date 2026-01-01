# fed_algo/scaffold/client.py
import torch
from fed_algo.base import BaseClient

class ScaffoldClient(BaseClient):
    """
    ScaffoldClient implements the client-side logic for the SCAFFOLD algorithm in Federated Learning.
    SCAFFOLD aims to reduce the variance caused by client drift by introducing control variates (local and global control variables)
    that adjust the client updates during training.

    Attributes:
        local_control (list[torch.Tensor]): A list of tensors representing the local control variates for each parameter in the model.
    """
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the ScaffoldClient with the provided arguments, client ID, data, and device.
        
        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the client and the server.
            device (torch.device): Device to run the computations on (CPU or GPU).
        """
        super().__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_control = [torch.zeros_like(p.data, device=self.device) for p in self.task.model.parameters()]

    def step_preprocess(self):
        """
        Modifies the gradients of the model parameters by adding the difference between the global and local control variates.
        """
        global_control = self.message_pool["server"]["global_control"]
        for p, local_control, global_control in zip(self.task.model.parameters(), self.local_control, global_control):
            if p.grad is None:
                continue
            p.grad.data.add_(global_control.to(self.device) - local_control)

    @torch.no_grad()
    def update_local_control(self):
        """
        Updates the local control variates based on the difference between the global and local model parameters
        after training. This adjustment is crucial for reducing the variance in the federated learning process.
        """
        # OpenFGL-style update
        global_control = self.message_pool["server"]["global_control"]
        global_weight  = self.message_pool["server"]["weight"]

        local_epochs = getattr(self.args, "local_epochs", getattr(self.args, "num_epochs", 1))
        lr = self.task.optimizer.param_groups[0]["lr"]

        for i, (w_local, w_global, c) in enumerate(zip(self.task.model.parameters(), global_weight, global_control)):
            w_global = w_global.to(self.device)
            c = c.to(self.device)

            self.local_control[i].data = self.local_control[i].data - c.data + (w_global.data - w_local.data) / (local_epochs * lr)

    def execute(self):
        """
        Executes the local training process for the client. It involves updating the local model with the global model
        parameters and applying the control variates to adjust the gradients before training.
        """ 
        # sync local params from server
        with torch.no_grad():
            for lp, gp in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                lp.data.copy_(gp.to(self.device))

        # set step preprocess and train
        self.task.step_preprocess = self.step_preprocess
        self.task.train()
        self.task.step_preprocess = None

        # update local control
        self.update_local_control()

    def send_message(self):
        """
        Sends the updated model parameters and local control variates to the server after local training is completed.
        This information is used by the server to update the global model and control variates for the next round.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": [p.data.detach().cpu().clone() for p in self.task.model.parameters()],
            "local_control": [c.detach().cpu().clone() for c in self.local_control],
        }
