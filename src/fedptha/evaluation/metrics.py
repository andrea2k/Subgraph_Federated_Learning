"""The exact PR-AUC and F1 definitions used by the published experiments."""

import torch


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


def pr_auc_binary(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute true trapezoidal area under the binary precision-recall curve.

    Semantics match ``precision_recall_curve`` followed by
    ``auc(recall, precision)`` with all distinct score thresholds retained.
    Tied prediction scores are handled as a single threshold, so the result
    depends only on thresholded prediction sets rather than on arbitrary order
    within a tie.

    This is the metric stored in the new pipeline's ``pr_auc``,
    ``micro_pr_auc`` and ``macro_pr_auc`` fields.

    Returns NaN when no positive label is present, because a recall axis is not
    defined for that evaluated set.
    """
    scores = scores.detach().flatten().float().cpu()
    labels = labels.detach().flatten().float().cpu()

    if scores.numel() == 0:
        return float("nan")
    if scores.numel() != labels.numel():
        raise ValueError(
            f"scores/labels length mismatch: {scores.numel()} vs {labels.numel()}"
        )

    positives = labels > 0.5
    num_pos = int(positives.sum().item())
    if num_pos == 0:
        return float("nan")

    # Descending score order.  We evaluate the curve only after the final item
    # of each tied-score block, which is equivalent to sweeping distinct
    # decision thresholds.
    order = torch.argsort(scores, descending=True)
    scores_sorted = scores[order]
    y_sorted = positives[order].to(torch.float64)

    tp_cum = torch.cumsum(y_sorted, dim=0)
    n = int(scores_sorted.numel())
    if n == 1:
        threshold_idx = torch.tensor([0], dtype=torch.long)
    else:
        distinct_end = torch.nonzero(
            scores_sorted[:-1] != scores_sorted[1:],
            as_tuple=False,
        ).flatten()
        threshold_idx = torch.cat(
            [distinct_end, torch.tensor([n - 1], dtype=torch.long)],
            dim=0,
        )

    tp = tp_cum[threshold_idx]
    predicted_positive = threshold_idx.to(torch.float64) + 1.0
    fp = predicted_positive - tp

    precision = tp / (tp + fp).clamp_min(1e-15)
    recall = tp / float(num_pos)

    # Precision-recall libraries conventionally return the curve from recall=1
    # down to recall=0 and append (recall=0, precision=1).  Integrate the same
    # piecewise-linear curve, but use positive recall widths explicitly.
    precision_curve = torch.cat(
        [precision.flip(0), torch.ones(1, dtype=torch.float64)], dim=0
    )
    recall_curve = torch.cat(
        [recall.flip(0), torch.zeros(1, dtype=torch.float64)], dim=0
    )

    recall_width = recall_curve[:-1] - recall_curve[1:]
    trapezoid_height = 0.5 * (precision_curve[:-1] + precision_curve[1:])
    area = torch.sum(recall_width * trapezoid_height)
    return float(area.item())


def nanmean_float(values) -> float:
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float32)
    if tensor.numel() == 0 or bool(torch.isnan(tensor).all().item()):
        return float("nan")
    return float(torch.nanmean(tensor).item())
