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
        mix_alpha: float = 1.0,       
    ) -> torch.Tensor:
        """
        Returns updated embeddings where ghost nodes (owned_mask = False)
        are blended with the owner's embeddings if available:
        h_ghost <- (1 - alpha) * h_local + alpha * h_owner.
        """
        if layer not in self.embeddings:
            return local_embs

        emb_table = self.embeddings[layer]
        out = local_embs

        if mix_alpha == 0.0:
            # no mixing, keep local embeddings
            return out

        ghost_idx = torch.where(~owned_mask)[0]
        for idx in ghost_idx.tolist():
            gid = int(global_nids[idx].item())
            if gid in emb_table:
                owner_emb = emb_table[gid].to(local_embs.device)
                if mix_alpha >= 1.0:
                    # behaves like hard overwrite
                    out[idx] = owner_emb
                else:
                    out[idx] = (1.0 - mix_alpha) * out[idx] + mix_alpha * owner_emb

        return out
