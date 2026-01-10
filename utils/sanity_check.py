# utils/sanity_check.py
import torch

def sanity_check_client_graphs(client_graphs, node_to_client, *, max_print=3):
    """
    Validates owned/ghost invariants for each saved client graph.
    Call once after loading client_graphs.
    """
    node_to_client = node_to_client.long().cpu()

    bad = 0
    for cid, gd in enumerate(client_graphs):
        # required attributes
        assert hasattr(gd, "owned_mask"), f"[cid={cid}] missing owned_mask"
        assert hasattr(gd, "global_nid"), f"[cid={cid}] missing global_nid"

        owned_mask = gd.owned_mask
        global_nid = gd.global_nid

        assert owned_mask.dtype == torch.bool, f"[cid={cid}] owned_mask must be bool, got {owned_mask.dtype}"
        assert global_nid.dtype in (torch.int64, torch.long), f"[cid={cid}] global_nid must be long"
        assert owned_mask.numel() == gd.num_nodes, f"[cid={cid}] owned_mask size mismatch"
        assert global_nid.numel() == gd.num_nodes, f"[cid={cid}] global_nid size mismatch"

        # owned nodes must belong to this cid in node_to_client (in global space)
        owned_global = global_nid[owned_mask].cpu()
        if owned_global.numel() > 0:
            assigned = node_to_client[owned_global]
            ok = (assigned == cid).all().item()
            if not ok:
                bad += 1
                wrong = owned_global[assigned != cid][:10].tolist()
                raise AssertionError(
                    f"[cid={cid}] owned nodes include globals not assigned to cid. Examples: {wrong}"
                )

        # edge invariant: every edge must touch an owned endpoint (in local space)
        src, dst = gd.edge_index[0], gd.edge_index[1]
        touches_owned = owned_mask[src] | owned_mask[dst]
        if not bool(touches_owned.all().item()):
            bad += 1
            idx = torch.where(~touches_owned)[0][:10].tolist()
            raise AssertionError(f"[cid={cid}] found edges not incident to owned nodes. Edge idx examples: {idx}")

        # optional stats
        cross = int((owned_mask[src] ^ owned_mask[dst]).sum().item())
        if cid < max_print:
            print(
                f"[SANITY cid={cid}] nodes={gd.num_nodes} owned={int(owned_mask.sum())} "
                f"ghost={gd.num_nodes - int(owned_mask.sum())} edges={gd.edge_index.size(1)} "
                f"cross_edges={cross}"
            )

    print(f"[SANITY] All client graphs passed. ({len(client_graphs)} clients)")
