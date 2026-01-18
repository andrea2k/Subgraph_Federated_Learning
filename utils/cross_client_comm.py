import torch
from collections import defaultdict

class CrossClientComm:
    """
    Simulated cross-client embedding bus.

    - global_owner: dict[int -> int], maps global node id -> owner client id
    - embeddings[layer][global_nid] = 1D tensor (embedding at that layer)
    """

    def __init__(self, global_owner: dict[int, int]):
        self.global_owner = global_owner
        self.embeddings = defaultdict(dict)  # layer -> {global_nid: emb}

    def reset_round(self):
        """Call once at the start of each FedAvg round, if desired."""
        self.embeddings.clear()

    def push_owned(
        self,
        *,
        client_id: int,
        layer: int,
        global_nids: torch.Tensor,   # [N]
        node_embs: torch.Tensor,     # [N, D]
        owned_mask: torch.Tensor,    # [N] bool
    ):
        owned_idx = torch.where(owned_mask)[0]
        for idx in owned_idx.tolist():
            gid = int(global_nids[idx].item())
            owner = self.global_owner.get(gid, None)
            if owner is None or owner != client_id:
                # Ignore ownership mismatch
                continue
            self.embeddings[layer][gid] = node_embs[idx].detach().cpu()

    def pull_ghost_and_merge(
        self,
        *,
        layer: int,
        global_nids: torch.Tensor,    # [N]
        owned_mask: torch.Tensor,     # [N] bool
        local_embs: torch.Tensor,     # [N, D] on device
    ) -> torch.Tensor:
        """
        Returns updated embeddings where ghost nodes (owned_mask = False)
        are replaced with the owner's embeddings if available.
        """
        if layer not in self.embeddings:
            return local_embs

        emb_table = self.embeddings[layer]
        out = local_embs

        ghost_idx = torch.where(~owned_mask)[0]
        for idx in ghost_idx.tolist():
            gid = int(global_nids[idx].item())
            if gid in emb_table:
                out[idx] = emb_table[gid].to(local_embs.device)

        return out
