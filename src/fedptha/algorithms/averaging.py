from __future__ import annotations
from fedptha.training.runtime import RunPaths, create_run_paths
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from fedptha.data.clients import ClientData, format_seconds
from fedptha.training.steps import (
    build_criterion_for_client,
    evaluate_loader,
    train_epoch_neighbor,
    train_epoch_neighbor_fedprox,
)
from fedptha.training.random import set_seed
from fedptha.training.runtime import (
    RunPaths,
    _subset_clients_str,
    append_eval_rows,
    append_mean_eval_row,
    append_train_row,
    build_client_loaders,
    build_model_context,
    build_runtime_clients,
    fedavg_state_dict,
    init_best_states,
    make_model,
    make_optimizer,
    tag_recent_eval_rows,
    update_best_state,
    weighted_scalar_summary,
)


def run_local_experiment(
    client: ClientData,
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    device: torch.device,
) -> None:
    """Train once and select by Macro PR-AUC only.

    One training trajectory is produced. During validation we keep one best
    in-memory state for each validation view (full and visible), both selected
    by Macro true PR-AUC. Final VV/VO/OO evaluations are written to CSV only;
    no .pt checkpoint is persisted.
    """
    set_seed(seed)
    ctx = build_model_context([client], cfg)
    train_graph = client.train_h
    train_loader, val_loader, test_loader = build_client_loaders(client, cfg, device)
    criterion = build_criterion_for_client(train_graph, cfg, device)
    model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    rows: List[Dict] = []
    best_by_eval_mode = init_best_states(model, key_name="epoch")
    total_epochs = int(rounds * local_epochs)
    run_start_time = time.perf_counter()
    for local_epoch in range(1, total_epochs + 1):
        train_loss = train_epoch_neighbor(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_ego_ids=ctx["use_ego_ids"],
            ego_dim=ctx["ego_dim"],
        )
        append_train_row(
            rows,
            run_type="local",
            algorithm="local",
            subset_id=None,
            subset_clients=None,
            seed=seed,
            graph_id=str(client.graph_id),
            dataset_id=str(client.dataset_id),
            phase="train_epoch",
            split="train",
            train_loss=train_loss,
            num_nodes=int(train_graph["n"].num_nodes),
            local_epoch=local_epoch,
        )
        rows[-1]["selection_metrics"] = "macro_pr_auc"
        print("local_epoch:", local_epoch, "train_loss:", train_loss)
        for eval_mask_mode in ("full", "visible"):
            val_metrics = evaluate_loader(
                model,
                val_loader,
                criterion,
                device,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
                threshold=0.5,
                eval_mask_mode=eval_mask_mode,
            )
            append_eval_rows(
                rows,
                run_type="local",
                algorithm="local",
                subset_id=None,
                subset_clients=None,
                seed=seed,
                phase=f"val_epoch_{eval_mask_mode}",
                split="val",
                graph_id=str(client.graph_id),
                dataset_id=str(client.dataset_id),
                metrics=val_metrics,
                local_epoch=local_epoch,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
            tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            print(
                "local_epoch:",
                local_epoch,
                "eval_mask_mode:",
                eval_mask_mode,
                "val_loss:",
                val_metrics["scalar"]["loss"],
            )
            print(
                "val_macro_minority_f1:", val_metrics["scalar"].get("macro_minority_f1")
            )
            print("val_positive_f1:", val_metrics["per_task"].get("positive_f1"))
            print("val_pr_auc:", val_metrics["per_task"].get("pr_auc"))
            update_best_state(
                best_by_eval_mode=best_by_eval_mode,
                eval_mask_mode=eval_mask_mode,
                metrics=val_metrics,
                model=model,
                step_value=local_epoch,
                key_name="epoch",
            )
    run_elapsed = time.perf_counter() - run_start_time
    print(f"run time: {format_seconds(run_elapsed)}")
    print("=======================================================")
    print("=======================================================")
    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
            "splits": ["test"],
        },
    ]
    split_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]
        selected_record = best_by_eval_mode[selected_by]
        selected_state = selected_record["state"]
        selected_epoch = selected_record["epoch"]
        best_value = selected_record["value"]
        model.load_state_dict(selected_state, strict=True)
        for split_name in job["splits"]:
            metrics = evaluate_loader(
                model,
                split_loaders[split_name],
                criterion,
                device,
                use_ego_ids=ctx["use_ego_ids"],
                ego_dim=ctx["ego_dim"],
                threshold=0.5,
                eval_mask_mode=eval_mask_mode,
            )
            append_eval_rows(
                rows,
                run_type="local",
                algorithm="local",
                subset_id=None,
                subset_clients=None,
                seed=seed,
                phase=f"best_{selection_protocol}_{split_name}",
                split=split_name,
                graph_id=str(client.graph_id),
                dataset_id=str(client.dataset_id),
                metrics=metrics,
                local_epoch=selected_epoch,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=selection_protocol,
                selected_by=selected_by,
            )
            tag_recent_eval_rows(
                rows,
                selection_metrics="macro_pr_auc",
                selector_metric="macro_pr_auc",
                selector_direction="max",
                selected_by_eval_mode=selected_by,
                selected_epoch=selected_epoch,
                selected_round=None,
                best_val_metric_value=best_value,
            )
    out_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("LOCAL MACRO-PR-AUC SELECTION SMOKE CHECK")
    print("=" * 100)
    print("csv path:", run_paths.csv_path)
    print("selection_metrics:", ("macro_pr_auc",))
    print(
        "best epochs:",
        {mode: best_by_eval_mode[mode]["epoch"] for mode in ["full", "visible"]},
    )
    print(
        "best values:",
        {mode: best_by_eval_mode[mode]["value"] for mode in ["full", "visible"]},
    )
    final_rows = out_df[
        out_df["task"].isna()
        & (out_df["split"] == "test")
        & out_df["selection_protocol"].isin(
            ["oracle_full", "realistic_visible", "realistic_selection_oracle"]
        )
    ]
    print("final test protocol rows:")
    preview_cols = [
        "phase",
        "selector_metric",
        "eval_mask_mode",
        "selection_protocol",
        "selected_by",
        "graph_id",
        "eval_loss",
        "micro_f1",
        "macro_pos_f1",
        "micro_pr_auc",
        "macro_pr_auc",
        "visible_pair_rate",
    ]
    preview_cols = [c for c in preview_cols if c in final_rows.columns]
    print(final_rows[preview_cols].to_string(index=False))
    print("=" * 100)
    out_df.to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_local(
    client: ClientData,
    cfg: Dict,
    seed: int,
    local_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    device: torch.device,
) -> RunPaths:
    cfg = dict(cfg)
    cfg["selection_tag"] = "macro_pr_auc"
    run_paths = create_run_paths(local_root)
    if run_paths.csv_path.exists():
        return run_paths
    run_local_experiment(
        client,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        device=device,
    )
    return run_paths


def run_fedavg_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
) -> None:
    """FedAvg training once, selecting one checkpoint per validation view by Macro PR-AUC."""
    if not subset_clients:
        raise ValueError("subset_clients is empty")
    set_seed(seed)
    subset_id = _subset_clients_str(subset_clients)
    subset_clients_str = subset_id
    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)
    global_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    rows: List[Dict] = []
    best_by_eval_mode = init_best_states(global_model, key_name="round")
    num_clients = len(runtime_clients)
    generator = torch.Generator().manual_seed(seed)
    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()
        num_selected = max(1, int(round(client_fraction * num_clients)))
        if num_selected >= num_clients:
            selected_indices = list(range(num_clients))
        else:
            permutation = torch.randperm(num_clients, generator=generator).tolist()
            selected_indices = permutation[:num_selected]
        local_states = []
        local_weights = []
        for idx in selected_indices:
            runtime = runtime_clients[idx]
            local_model = make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            local_model.load_state_dict(global_model.state_dict(), strict=True)
            optimizer = make_optimizer(local_model, cfg)
            for local_epoch_idx in range(1, local_epochs + 1):
                train_loss = train_epoch_neighbor(
                    local_model,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                )
                append_train_row(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="train_epoch",
                    split="train",
                    train_loss=train_loss,
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )
                rows[-1]["selection_metrics"] = "macro_pr_auc"
            for eval_mask_mode in ("full", "visible"):
                post_metrics = evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                append_eval_rows(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"val_epoch_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=post_metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
                tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            local_states.append(
                {
                    key: value.detach().cpu().clone()
                    for (key, value) in local_model.state_dict().items()
                }
            )
            local_weights.append(runtime.num_train_nodes)
        aggregated_state = fedavg_state_dict(local_states, local_weights)
        global_model.load_state_dict(aggregated_state, strict=True)
        for eval_mask_mode in ("full", "visible"):
            round_eval_infos = []
            for runtime in runtime_clients:
                metrics = evaluate_loader(
                    global_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                round_eval_infos.append(metrics)
                print("fed eval on client:", runtime.client.graph_id)
                print(
                    "fedavg round:",
                    round_idx,
                    "eval_mask_mode:",
                    eval_mask_mode,
                    "val_loss:",
                    metrics["scalar"]["loss"],
                )
                print(
                    "val_macro_minority_f1:", metrics["scalar"].get("macro_minority_f1")
                )
                print("val_positive_f1:", metrics["per_task"].get("positive_f1"))
                print("val_pr_auc:", metrics["per_task"].get("pr_auc"))
                append_eval_rows(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"global_val_client_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
                tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            append_mean_eval_row(
                rows,
                run_type="fedavg",
                algorithm="fedavg",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase=f"global_val_mean_{eval_mask_mode}",
                split="val",
                round_idx=round_idx,
                eval_infos=round_eval_infos,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
            rows[-1]["selection_metrics"] = "macro_pr_auc"
            mean_scalar = weighted_scalar_summary(round_eval_infos)
            update_best_state(
                best_by_eval_mode=best_by_eval_mode,
                eval_mask_mode=eval_mask_mode,
                metrics={"scalar": mean_scalar},
                model=global_model,
                step_value=round_idx,
                key_name="round",
            )
        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")
    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
            "splits": ["test"],
        },
    ]
    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]
        selected_record = best_by_eval_mode[selected_by]
        selected_state = selected_record["state"]
        selected_round = selected_record["round"]
        best_value = selected_record["value"]
        global_model.load_state_dict(selected_state, strict=True)
        for split_name in job["splits"]:
            for runtime in runtime_clients:
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]
                metrics = evaluate_loader(
                    global_model,
                    split_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                append_eval_rows(
                    rows,
                    run_type="fedavg",
                    algorithm="fedavg",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{selection_protocol}_{split_name}",
                    split=split_name,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=selected_round,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=selection_protocol,
                    selected_by=selected_by,
                )
                tag_recent_eval_rows(
                    rows,
                    selection_metrics="macro_pr_auc",
                    selector_metric="macro_pr_auc",
                    selector_direction="max",
                    selected_by_eval_mode=selected_by,
                    selected_epoch=None,
                    selected_round=selected_round,
                    best_val_metric_value=best_value,
                )
    out_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("FEDAVG MACRO-PR-AUC SELECTION SMOKE CHECK")
    print("=" * 100)
    print("csv path:", run_paths.csv_path)
    print("selection_metrics:", ("macro_pr_auc",))
    print(
        "best rounds:",
        {mode: best_by_eval_mode[mode]["round"] for mode in ["full", "visible"]},
    )
    print(
        "best values:",
        {mode: best_by_eval_mode[mode]["value"] for mode in ["full", "visible"]},
    )
    final_rows = out_df[
        out_df["task"].isna()
        & (out_df["split"] == "test")
        & out_df["selection_protocol"].isin(
            ["oracle_full", "realistic_visible", "realistic_selection_oracle"]
        )
    ]
    print("final test protocol rows:")
    preview_cols = [
        "phase",
        "selector_metric",
        "eval_mask_mode",
        "selection_protocol",
        "selected_by",
        "graph_id",
        "eval_loss",
        "micro_f1",
        "macro_pos_f1",
        "micro_pr_auc",
        "macro_pr_auc",
        "visible_pair_rate",
    ]
    preview_cols = [c for c in preview_cols if c in final_rows.columns]
    print(final_rows[preview_cols].to_string(index=False))
    print("=" * 100)
    out_df.to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_fedavg(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    fedavg_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    device: torch.device,
) -> RunPaths:
    """Train FedAvg and evaluate the best states for each validation view."""
    cfg = dict(cfg)
    cfg["selection_tag"] = "macro_pr_auc"
    subset_id = _subset_clients_str(subset_clients)
    run_paths = create_run_paths(fedavg_root)
    if run_paths.csv_path.exists():
        return run_paths
    run_fedavg_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        device=device,
    )
    return run_paths


def run_fedprox_experiment(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    run_paths: RunPaths,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    fedprox_mu: float,
    device: torch.device,
) -> None:
    """FedProx training once, selecting one checkpoint per validation view by Macro PR-AUC.

    This mirrors run_fedavg_experiment(...) but uses the FedProx proximal term
    during local training. It writes one CSV containing all selected-checkpoint
    performances and never writes .pt checkpoints in the final PR-AUC protocol.
    """
    if not subset_clients:
        raise ValueError("subset_clients is empty")
    set_seed(seed)
    subset_id = _subset_clients_str(subset_clients)
    subset_clients_str = subset_id
    ctx = build_model_context(subset_clients, cfg)
    runtime_clients = build_runtime_clients(subset_clients, cfg, device)
    global_model = make_model(
        cfg,
        ctx["x_dim"],
        ctx["out_dim"],
        ctx["deg_fwd_hist"],
        ctx["deg_rev_hist"],
        ctx["ego_dim"],
        ctx["in_vocab"],
        ctx["out_vocab"],
    ).to(device)
    rows: List[Dict] = []
    best_by_eval_mode = init_best_states(global_model, key_name="round")
    num_clients = len(runtime_clients)
    generator = torch.Generator().manual_seed(seed)
    aggregated = True
    for round_idx in range(1, rounds + 1):
        run_start_time = time.perf_counter()
        num_selected = max(1, int(round(client_fraction * num_clients)))
        if num_selected >= num_clients:
            selected_indices = list(range(num_clients))
        else:
            permutation = torch.randperm(num_clients, generator=generator).tolist()
            selected_indices = permutation[:num_selected]
        local_states = []
        local_weights = []
        for idx in selected_indices:
            runtime = runtime_clients[idx]
            local_model = make_model(
                cfg,
                ctx["x_dim"],
                ctx["out_dim"],
                ctx["deg_fwd_hist"],
                ctx["deg_rev_hist"],
                ctx["ego_dim"],
                ctx["in_vocab"],
                ctx["out_vocab"],
            ).to(device)
            local_model.load_state_dict(global_model.state_dict(), strict=True)
            global_params = {
                name: param.detach().clone()
                for (name, param) in local_model.named_parameters()
            }
            optimizer = make_optimizer(local_model, cfg)
            for local_epoch_idx in range(1, local_epochs + 1):
                train_loss = train_epoch_neighbor_fedprox(
                    local_model,
                    global_params,
                    runtime.train_loader,
                    optimizer,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    fedprox_mu=fedprox_mu,
                    aggregated=aggregated,
                )
                append_train_row(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    phase="train_epoch",
                    split="train",
                    train_loss=train_loss,
                    num_nodes=runtime.num_train_nodes,
                    round_idx=round_idx,
                    local_epoch=local_epoch_idx,
                )
                rows[-1]["selection_metrics"] = "macro_pr_auc"
            for eval_mask_mode in ("full", "visible"):
                post_metrics = evaluate_loader(
                    local_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                append_eval_rows(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"val_epoch_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=post_metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
                tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            local_states.append(
                {
                    key: value.detach().cpu().clone()
                    for (key, value) in local_model.state_dict().items()
                }
            )
            local_weights.append(runtime.num_train_nodes)
        aggregated_state = fedavg_state_dict(local_states, local_weights)
        global_model.load_state_dict(aggregated_state, strict=True)
        aggregated = True
        for eval_mask_mode in ("full", "visible"):
            round_eval_infos = []
            for runtime in runtime_clients:
                metrics = evaluate_loader(
                    global_model,
                    runtime.val_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                round_eval_infos.append(metrics)
                print("fed eval on client:", runtime.client.graph_id)
                print(
                    "fedprox round:",
                    round_idx,
                    "eval_mask_mode:",
                    eval_mask_mode,
                    "val_loss:",
                    metrics["scalar"]["loss"],
                )
                print(
                    "val_macro_minority_f1:", metrics["scalar"].get("macro_minority_f1")
                )
                print("val_positive_f1:", metrics["per_task"].get("positive_f1"))
                print("val_pr_auc:", metrics["per_task"].get("pr_auc"))
                append_eval_rows(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"global_val_client_{eval_mask_mode}",
                    split="val",
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=round_idx,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=None,
                    selected_by=None,
                )
                tag_recent_eval_rows(rows, selection_metrics="macro_pr_auc")
            append_mean_eval_row(
                rows,
                run_type="fedprox",
                algorithm="fedprox",
                subset_id=subset_id,
                subset_clients=subset_clients_str,
                seed=seed,
                phase=f"global_val_mean_{eval_mask_mode}",
                split="val",
                round_idx=round_idx,
                eval_infos=round_eval_infos,
                eval_mask_mode=eval_mask_mode,
                selection_protocol=None,
                selected_by=None,
            )
            rows[-1]["selection_metrics"] = "macro_pr_auc"
            mean_scalar = weighted_scalar_summary(round_eval_infos)
            update_best_state(
                best_by_eval_mode=best_by_eval_mode,
                eval_mask_mode=eval_mask_mode,
                metrics={"scalar": mean_scalar},
                model=global_model,
                step_value=round_idx,
                key_name="round",
            )
        run_elapsed = time.perf_counter() - run_start_time
        print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(f"==================== round {round_idx} / {rounds} ====================")
        print("=======================================================")
    final_eval_jobs = [
        {
            "selection_protocol": "oracle_full",
            "selected_by": "full",
            "eval_mask_mode": "full",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_visible",
            "selected_by": "visible",
            "eval_mask_mode": "visible",
            "splits": ["train", "val", "test"],
        },
        {
            "selection_protocol": "realistic_selection_oracle",
            "selected_by": "visible",
            "eval_mask_mode": "full",
            "splits": ["test"],
        },
    ]
    for job in final_eval_jobs:
        selection_protocol = job["selection_protocol"]
        selected_by = job["selected_by"]
        eval_mask_mode = job["eval_mask_mode"]
        selected_record = best_by_eval_mode[selected_by]
        selected_state = selected_record["state"]
        selected_round = selected_record["round"]
        best_value = selected_record["value"]
        global_model.load_state_dict(selected_state, strict=True)
        for split_name in job["splits"]:
            for runtime in runtime_clients:
                split_loader = {
                    "train": runtime.train_loader,
                    "val": runtime.val_loader,
                    "test": runtime.test_loader,
                }[split_name]
                metrics = evaluate_loader(
                    global_model,
                    split_loader,
                    runtime.criterion,
                    device,
                    use_ego_ids=ctx["use_ego_ids"],
                    ego_dim=ctx["ego_dim"],
                    threshold=0.5,
                    eval_mask_mode=eval_mask_mode,
                )
                append_eval_rows(
                    rows,
                    run_type="fedprox",
                    algorithm="fedprox",
                    subset_id=subset_id,
                    subset_clients=subset_clients_str,
                    seed=seed,
                    phase=f"best_{selection_protocol}_{split_name}",
                    split=split_name,
                    graph_id=str(runtime.client.graph_id),
                    dataset_id=str(runtime.client.dataset_id),
                    metrics=metrics,
                    round_idx=selected_round,
                    eval_mask_mode=eval_mask_mode,
                    selection_protocol=selection_protocol,
                    selected_by=selected_by,
                )
                tag_recent_eval_rows(
                    rows,
                    selection_metrics="macro_pr_auc",
                    selector_metric="macro_pr_auc",
                    selector_direction="max",
                    selected_by_eval_mode=selected_by,
                    selected_epoch=None,
                    selected_round=selected_round,
                    best_val_metric_value=best_value,
                )
    out_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("FEDPROX MACRO-PR-AUC SELECTION SMOKE CHECK")
    print("=" * 100)
    print("csv path:", run_paths.csv_path)
    print("selection_metrics:", ("macro_pr_auc",))
    print(
        "best rounds:",
        {mode: best_by_eval_mode[mode]["round"] for mode in ["full", "visible"]},
    )
    print(
        "best values:",
        {mode: best_by_eval_mode[mode]["value"] for mode in ["full", "visible"]},
    )
    final_rows = out_df[
        out_df["task"].isna()
        & (out_df["split"] == "test")
        & out_df["selection_protocol"].isin(
            ["oracle_full", "realistic_visible", "realistic_selection_oracle"]
        )
    ]
    print("final test protocol rows:")
    preview_cols = [
        "phase",
        "selector_metric",
        "eval_mask_mode",
        "selection_protocol",
        "selected_by",
        "graph_id",
        "eval_loss",
        "micro_f1",
        "macro_pos_f1",
        "micro_pr_auc",
        "macro_pr_auc",
        "visible_pair_rate",
    ]
    preview_cols = [c for c in preview_cols if c in final_rows.columns]
    print(final_rows[preview_cols].to_string(index=False))
    print("=" * 100)
    out_df.to_csv(run_paths.csv_path, index=False)
    print(f"saved csv -> {run_paths.csv_path}")


def run_fedprox(
    subset_clients: List[ClientData],
    cfg: Dict,
    seed: int,
    fedprox_root: str | Path,
    *,
    rounds: int,
    local_epochs: int,
    client_fraction: float,
    fedprox_mu: float,
    device: torch.device,
) -> RunPaths:
    """Run FedProx once using Macro PR-AUC as the only checkpoint selector."""
    cfg = dict(cfg)
    cfg["selection_tag"] = "macro_pr_auc"
    subset_id = _subset_clients_str(subset_clients)
    run_paths = create_run_paths(fedprox_root)
    if run_paths.csv_path.exists():
        return run_paths
    run_fedprox_experiment(
        subset_clients,
        cfg,
        seed,
        run_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        fedprox_mu=fedprox_mu,
        device=device,
    )
    return run_paths
