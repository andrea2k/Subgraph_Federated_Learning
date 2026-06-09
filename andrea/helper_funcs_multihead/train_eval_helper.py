from __future__ import annotations

import torch
from torch import nn
from typing import Any
from typing import Dict
from torch_geometric.loader import NeighborLoader


def build_model_tag(
    mcw, num_layers, lr, weight_decay, dropout, hidden_dim, use_ego_ids, batch_size
):
    return (
        f"mcw{mcw}"
        f"_layers{num_layers}"
        f"_lr{lr}"
        f"_wd{weight_decay}"
        f"_do{dropout}"
        f"_hd{hidden_dim}"
        f"_ego{use_ego_ids}"
        f"_bs{batch_size}"
    )


def unpack_batch_edges(batch):
    edge_in = {
        ("n", "fwd", "n"): batch[("n", "fwd", "n")].edge_index,
        ("n", "rev", "n"): batch[("n", "rev", "n")].edge_index,
    }

    edge_attr_dict = {}
    for rel in [("n", "fwd", "n"), ("n", "rev", "n")]:
        if "edge_attr" in batch[rel]:
            ea = batch[rel].edge_attr
            if ea.dtype != torch.long:
                ea = ea.long()
            edge_attr_dict[rel] = ea

    return edge_in, edge_attr_dict


def build_hetero_neighbor_loader(
    hetero_data,
    batch_size: int,
    neighbors_per_hop: Any,
    device=None,
    shuffle=True,
):

    num_neighbors = {
        ("n", "fwd", "n"): neighbors_per_hop,
        ("n", "rev", "n"): neighbors_per_hop,
    }

    use_cuda = device is not None and device.type == "cuda"

    return NeighborLoader(
        hetero_data,
        num_neighbors=num_neighbors,
        input_nodes=("n", torch.arange(hetero_data["n"].num_nodes)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        filter_per_worker=True,
    )


def augment_batch_x_with_ego(batch, use_ego_ids: bool, ego_dim: int):
    """
    Add a 1-dimensional binary ego indicator to node features.
    """
    x = batch["n"].x
    y = batch["n"].y
    B = int(batch["n"].batch_size)

    if use_ego_ids:
        ego = torch.zeros((x.size(0), ego_dim), device=x.device, dtype=x.dtype)
        ego[:B, 0] = 1.0
        x = torch.cat([x, ego], dim=-1)

    x_in = {"n": x}
    y_seed = y[:B]
    return x_in, y_seed, B


def get_label_stats_from_graph(graph):
    y = graph["n"].y.float()
    if "label_mask" in graph["n"]:
        mask = graph["n"].label_mask.float()
    else:
        mask = torch.ones_like(y)
    pos_cnt = (y * mask).sum(dim=0)
    total_cnt = mask.sum(dim=0)
    neg_cnt = total_cnt - pos_cnt
    pos_weight = neg_cnt / torch.clamp(pos_cnt, min=1.0)
    return pos_cnt, neg_cnt, pos_weight


def build_criterion_for_client(graph, cfg, device):
    mcw = cfg.get("minority_class_weight", None)

    if mcw is None:
        return nn.BCEWithLogitsLoss(reduction="none")

    if mcw == "auto":
        pos_cnt, neg_cnt, pos_weight = get_label_stats_from_graph(graph)
        pos_weight = pos_weight.to(device)
        print("[criterion] BCEWithLogitsLoss with AUTO pos_weight")
        print("  pos_cnt   :", pos_cnt)
        print("  neg_cnt   :", neg_cnt)
        print("  pos_weight:", pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.full((graph["n"].y.size(1),), float(mcw), device=device),
        reduction="none",
    )


def get_seed_label_mask(batch, B: int):
    if "label_mask" not in batch["n"]:
        return None
    return batch["n"].label_mask[:B].float()


def masked_loss_from_logits(criterion, logits, labels, label_mask=None):
    raw_loss = criterion(logits, labels.float())

    if raw_loss.ndim == 0:
        return raw_loss

    if label_mask is None:
        return raw_loss.mean()

    label_mask = label_mask.to(raw_loss.device).float()
    denom = label_mask.sum().clamp_min(1.0)

    return (raw_loss * label_mask).sum() / denom


def compute_minority_f1_score_per_task(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    y = labels.bool()

    N, C = y.shape
    f1_scores = torch.zeros(C, dtype=torch.float32, device=logits.device)
    epsilon = 1e-12

    for c in range(C):
        y_c = y[:, c]

        pos = y_c.sum()
        neg = y_c.numel() - pos
        minority_is_one = pos <= neg

        if minority_is_one:
            y_pos = y_c
            pred_pos = preds[:, c]
        else:
            y_pos = ~y_c
            pred_pos = ~preds[:, c]

        true_pos = (y_pos & pred_pos).sum().float()
        false_pos = ((~y_pos) & pred_pos).sum().float()
        false_neg = (y_pos & (~pred_pos)).sum().float()

        precision = true_pos / (true_pos + false_pos + epsilon)
        recall = true_pos / (true_pos + false_neg + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores[c] = f1

    return f1_scores


def compute_positive_f1_score_per_task(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    y = labels.bool()

    N, C = y.shape
    f1_scores = torch.zeros(C, dtype=torch.float32, device=logits.device)
    epsilon = 1e-12

    for c in range(C):
        y_c = y[:, c]
        pred_c = preds[:, c]

        tp = (pred_c & y_c).sum().float()
        fp = (pred_c & (~y_c)).sum().float()
        fn = ((~pred_c) & y_c).sum().float()

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores[c] = f1

    return f1_scores


def train_epoch_neighbor(
    model, loader, optimizer, criterion, device, use_ego_ids: bool, ego_dim: int
):
    model.train()
    total_loss = 0.0
    total_count = 0
    i = 0
    for batch in loader:
        i += 1
        if i % 10 == 0:
            print("train-batch:", i)

        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)

        edge_in, edge_attr_dict = unpack_batch_edges(batch)

        out = model(
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
        )

        out_seed = out[:B]

        label_mask_seed = get_seed_label_mask(batch, B)
        loss = masked_loss_from_logits(
            criterion,
            out_seed,
            y_seed.float(),
            label_mask_seed,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_count += B

    return total_loss / max(total_count, 1)


def train_epoch_neighbor_fedprox(
    model,
    global_params: Dict[str, torch.Tensor],
    loader,
    optimizer,
    criterion,
    device,
    use_ego_ids: bool,
    ego_dim: int,
    fedprox_mu: float,
    aggregated: bool,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    i = 0

    for batch in loader:
        i += 1
        if i % 10 == 0:
            print("train-batch:", i)

        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)

        out = model(
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
        )

        out_seed = out[:B]

        label_mask_seed = get_seed_label_mask(batch, B)
        base_loss = masked_loss_from_logits(
            criterion,
            out_seed,
            y_seed.float(),
            label_mask_seed,
        )

        if aggregated:
            prox_reg = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                prox_reg = prox_reg + torch.sum((param - global_params[name]) ** 2)
            loss = base_loss + 0.5 * fedprox_mu * prox_reg
        else:
            loss = base_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_count += B

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    criterion,
    device,
    use_ego_ids: bool,
    ego_dim: int,
    threshold: float = 0.5,
):
    """
    Single-pass evaluation:
    - fixed threshold=0.5
    - no threshold tuning
    - no separate logits evaluation pipeline
    """
    model.eval()

    total_loss = 0.0
    total_count = 0
    total_pairs = 0
    correct_pairs = 0

    all_logits = []
    all_labels = []

    i = 0
    for batch in loader:
        i += 1
        if i % 10 == 0:
            print("eval-batch:", i)
        batch = batch.to(device)

        x_in, y_seed, B = augment_batch_x_with_ego(batch, use_ego_ids, ego_dim)
        edge_in, edge_attr_dict = unpack_batch_edges(batch)

        out = model(
            x_in,
            edge_in,
            edge_attr_dict=edge_attr_dict,
            device=device,
        )

        out_used = out[:B]
        y_used = y_seed

        loss = masked_loss_from_logits(
            criterion,
            out_used,
            y_used.float(),
            label_mask=None,
        )

        total_loss += loss.item() * B
        total_count += B

        preds = torch.sigmoid(out_used) > threshold
        correct_pairs += (preds == y_used.bool()).sum().item()
        total_pairs += y_used.numel()

        all_logits.append(out_used.detach().cpu())
        all_labels.append(y_used.detach().cpu())

    avg_loss = total_loss / max(total_count, 1)
    pair_acc = correct_pairs / max(total_pairs, 1)

    logits = torch.cat(all_logits, dim=0) if len(all_logits) else torch.empty((0,))
    labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.empty((0,))

    minority_f1_per_task = (
        compute_minority_f1_score_per_task(logits, labels, threshold=threshold)
        .detach()
        .cpu()
        .float()
    )

    positive_f1_per_task = (
        compute_positive_f1_score_per_task(logits, labels, threshold=threshold)
        .detach()
        .cpu()
        .float()
    )

    macro_minority_f1 = float(minority_f1_per_task.mean().item())
    macro_pos_f1 = float(positive_f1_per_task.mean().item())

    pos_cnt = labels.sum(dim=0).to(torch.long).tolist()
    pos_rate = labels.float().mean(dim=0).tolist()

    # keep tp/fp/tn/fn if you still want detailed logging
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    y_bool = labels.bool()

    tp_task, fp_task, tn_task, fn_task = [], [], [], []
    prec_task, rec_task = [], []

    eps = 1e-12
    C = labels.size(1)
    for c in range(C):
        p = preds[:, c]
        y = y_bool[:, c]

        tp = int((p & y).sum().item())
        fp = int((p & (~y)).sum().item())
        tn = int(((~p) & (~y)).sum().item())
        fn = int(((~p) & y).sum().item())

        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)

        tp_task.append(tp)
        fp_task.append(fp)
        tn_task.append(tn)
        fn_task.append(fn)
        prec_task.append(float(precision))
        rec_task.append(float(recall))

    # micro-F1 over positive class
    tp_micro = (preds & y_bool).sum().item()
    fp_micro = (preds & (~y_bool)).sum().item()
    fn_micro = ((~preds) & y_bool).sum().item()

    micro_prec = tp_micro / max(tp_micro + fp_micro, eps)
    micro_rec = tp_micro / max(tp_micro + fn_micro, eps)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, eps)

    subset_acc = (preds == y_bool).all(dim=1).float().mean().item()

    return {
        "scalar": {
            "loss": float(avg_loss),
            "pair_acc": float(pair_acc),
            "subset_acc": float(subset_acc),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_pos_f1),  # for backward compatibility
            "macro_pos_f1": float(macro_pos_f1),
            "macro_minority_f1": float(macro_minority_f1),
        },
        "per_task": {
            "tp": tp_task,
            "fp": fp_task,
            "tn": tn_task,
            "fn": fn_task,
            "precision": prec_task,
            "recall": rec_task,
            "f1": positive_f1_per_task.tolist(),  # backward compatibility
            "positive_f1": positive_f1_per_task.tolist(),
            "minority_f1": minority_f1_per_task.tolist(),
        },
        "counts": {
            "num_nodes": int(labels.size(0)),
            "pos_cnt": [int(x) for x in pos_cnt],
            "pos_rate": [float(x) for x in pos_rate],
        },
    }
