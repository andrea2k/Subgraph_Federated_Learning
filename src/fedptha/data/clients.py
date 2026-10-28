from __future__ import annotations

from fedptha.data.allocation import allocation_counts

import torch
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from fedptha.data.io import load_datasets
from fedptha.data.hetero import make_bidirected_hetero

from fedptha.data.tasks import TASKS


@dataclass
class ClientData:
    graph_id: str
    data_dir: str
    dataset_id: str
    train_g: object
    val_g: object
    test_g: object
    train_h: object
    val_h: object
    test_h: object
    mask_meta: Optional[dict] = None


def row_has_mask(row: pd.Series) -> bool:
    if "mask_mode" not in row:
        return False
    value = row.get("mask_mode")
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "none", "nan"}


def mask_meta_from_row(row: pd.Series) -> Optional[dict]:
    if not row_has_mask(row):
        return None

    assigned_task = row.get("assigned_task", row.get("mask_task", None))

    q_value = row.get("q_value", None)
    if q_value is None or pd.isna(q_value):
        q_value = row.get("mask_fraction", None)

    q_iid = row.get("q_iid", None)
    if q_iid is None or pd.isna(q_iid):
        q_iid = 1.0 / float(len(TASKS))

    q_other_share = row.get("q_other_share", None)
    if q_other_share is None or pd.isna(q_other_share):
        q_other_share = (
            None
            if q_value is None or pd.isna(q_value)
            else (1.0 - float(q_value)) / float(len(TASKS) - 1)
        )

    return {
        "mask_mode": str(row["mask_mode"]),
        "mask_task": str(row["mask_task"]),
        "assigned_task": (
            str(assigned_task)
            if assigned_task is not None and pd.notna(assigned_task)
            else str(row["mask_task"])
        ),
        # Registry field for the assigned task's positive-label share.
        "mask_fraction": (
            float(row["mask_fraction"])
            if "mask_fraction" in row and pd.notna(row["mask_fraction"])
            else None
        ),
        # New q-controlled metadata.
        "q_value": (
            float(q_value) if q_value is not None and pd.notna(q_value) else None
        ),
        "q_iid": float(q_iid),
        "q_other_share": (
            float(q_other_share)
            if q_other_share is not None and pd.notna(q_other_share)
            else None
        ),
        "q_assigned_share": (
            float(row.get("q_assigned_share", q_value))
            if q_value is not None and pd.notna(q_value)
            else None
        ),
        "q_allocation_mode": str(row.get("q_allocation_mode", "")),
        "mask_seed": int(row.get("mask_seed", 0)),
        "mask_apply_split": str(row.get("mask_apply_split", "train")),
        "base_graph_id": int(row.get("base_graph_id", row["graph_id"])),
        "controlled_benchmark": str(row.get("controlled_benchmark", "")),
    }


def attach_full_label_mask(g):
    if not hasattr(g, "label_mask"):
        g.label_mask = torch.ones_like(g.y, dtype=torch.float32)

    return g


def q_other_share(q_value: float) -> float:
    if len(TASKS) <= 1:
        raise ValueError("q allocation requires at least two tasks.")
    return (1.0 - float(q_value)) / float(len(TASKS) - 1)


def validate_q_value(q_value: float) -> None:
    q = float(q_value)
    iid_q = 1.0 / float(len(TASKS))

    if q < 0.0 or q > 1.0:
        raise ValueError(f"q_value must be in [0, 1], got {q}")

    if q + 1e-12 < iid_q:
        raise ValueError(
            f"q_value={q} is below IID q=1/T={iid_q}. "
            "For this benchmark, use q >= 1/T."
        )


def _q_group_seed(*, base_graph_id: int, q_value: float, task_idx: int) -> int:
    """
    Deterministic seed shared by all virtual clients in the same q group.

    Important:
    Do NOT use virtual graph_id or per-client mask_seed here, otherwise each
    client would shuffle positives differently and the chunks would not be
    disjoint.
    """
    q_code = int(round(float(q_value) * 1_000_000))
    seed = int(base_graph_id) * 1_000_003 + q_code * 9_176 + int(task_idx) * 1_009 + 17
    return int(seed % (2**31 - 1))


def _q_counts_for_label_task(
    num_positives: int, *, label_task: str, q_value: float
) -> List[int]:
    """
    Counts are ordered by TASKS, where each position means:
      TASKS[i] = assigned task of receiving client.
    """
    q = float(q_value)
    off = q_other_share(q)

    shares = [q if receiving_task == label_task else off for receiving_task in TASKS]

    return allocation_counts(int(num_positives), shares)


def _q_positive_chunk_for_client(
    positives: torch.Tensor,
    *,
    label_task: str,
    assigned_task: str,
    q_value: float,
    base_graph_id: int,
) -> torch.Tensor:
    """
    For a fixed label task, split all positive labels into deterministic,
    disjoint chunks across the task-specialized clients.

    The client whose assigned_task equals label_task receives q of the positives.
    Every other client receives (1-q)/(T-1).
    """
    if assigned_task not in TASKS:
        raise ValueError(f"Unknown assigned_task={assigned_task}. Expected {TASKS}")

    if label_task not in TASKS:
        raise ValueError(f"Unknown label_task={label_task}. Expected {TASKS}")

    n_pos = int(positives.numel())
    if n_pos == 0:
        return positives.new_empty((0,), dtype=torch.long)

    label_task_idx = TASKS.index(label_task)
    assigned_task_idx = TASKS.index(assigned_task)

    counts = _q_counts_for_label_task(
        n_pos,
        label_task=label_task,
        q_value=q_value,
    )

    start = int(sum(counts[:assigned_task_idx]))
    end = int(start + counts[assigned_task_idx])

    gen = torch.Generator().manual_seed(
        _q_group_seed(
            base_graph_id=int(base_graph_id),
            q_value=float(q_value),
            task_idx=int(label_task_idx),
        )
    )

    perm = torch.randperm(n_pos, generator=gen)
    shuffled = positives[perm]

    return shuffled[start:end]


def apply_q_task_label_allocation_mask(
    g,
    *,
    assigned_task: str,
    q_value: float,
    base_graph_id: int,
):
    """
    q-controlled label heterogeneity.

    Only positive labels are masked.

    For every label task t:
      - all positive labels for task t are hidden first;
      - then each virtual client receives its deterministic q chunk;
      - negatives remain visible.

    This means:
      label_mask = 1 for all negatives
      label_mask = 1 only for the positive labels allocated to this client
      label_mask = 0 for positive labels allocated to other clients
    """
    g = attach_full_label_mask(g)

    if assigned_task not in TASKS:
        raise ValueError(
            f"Unknown assigned_task={assigned_task}. Expected one of {TASKS}"
        )

    if q_value is None:
        raise ValueError("q_task_label_allocation requires q_value.")

    validate_q_value(float(q_value))

    y = g.y.float()

    # Reset to full visibility first. Then only hide positives.
    g.label_mask = torch.ones_like(y, dtype=torch.float32)

    for task_idx, label_task in enumerate(TASKS):
        positives = torch.where(y[:, task_idx] > 0.5)[0]

        if positives.numel() == 0:
            continue

        # Hide all positives for this task.
        g.label_mask[positives, task_idx] = 0.0

        # Reveal only this client's deterministic chunk.
        visible_chunk = _q_positive_chunk_for_client(
            positives,
            label_task=label_task,
            assigned_task=assigned_task,
            q_value=float(q_value),
            base_graph_id=int(base_graph_id),
        )

        if visible_chunk.numel() > 0:
            g.label_mask[visible_chunk, task_idx] = 1.0

    return g


def apply_mask_to_one_graph(graph, mask_meta):
    graph = attach_full_label_mask(graph)
    if mask_meta is None:
        return graph
    return apply_q_task_label_allocation_mask(
        graph,
        assigned_task=mask_meta["assigned_task"],
        q_value=mask_meta["q_value"],
        base_graph_id=mask_meta["base_graph_id"],
    )


def apply_mask_to_splits(
    train_g,
    val_g,
    test_g,
    mask_meta,
    *,
    mask_splits=("train",),
):
    """
    Attach label masks to train/val/test.

    mask_splits=("train",)
        Old setting:
        - train is masked
        - val/test remain full-visible

    mask_splits=("train", "val", "test")
        Realistic setting:
        - train is masked
        - val is masked
        - test is masked

    Full/oracle evaluation is still possible because labels are not removed;
    only label_mask changes.
    """
    train_g = attach_full_label_mask(train_g)
    val_g = attach_full_label_mask(val_g)
    test_g = attach_full_label_mask(test_g)

    if mask_meta is None:
        return train_g, val_g, test_g

    if "train" in mask_splits:
        train_g = apply_mask_to_one_graph(train_g, mask_meta)

    if "val" in mask_splits:
        val_g = apply_mask_to_one_graph(val_g, mask_meta)

    if "test" in mask_splits:
        test_g = apply_mask_to_one_graph(test_g, mask_meta)

    return train_g, val_g, test_g


def load_client_from_dir(client_dir: str):
    train_g, val_g, test_g = load_datasets(
        log_dir=client_dir,
        train_data_file="train.pt",
        val_data_file="val.pt",
        test_data_file="test.pt",
    )
    return train_g, val_g, test_g


def parse_subset_clients(s: str) -> List[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip() != ""]


def collect_needed_client_ids(chosen_df: pd.DataFrame) -> List[int]:
    needed_client_ids = set()
    for s in chosen_df["subset_clients"]:
        needed_client_ids.update(parse_subset_clients(s))
    return sorted(needed_client_ids)


def _json_list(value) -> list:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    return json.loads(str(value))


def _q_tag_from_row(row: pd.Series) -> str:
    q_value = row.get("q_value", row.get("mask_fraction", None))
    if q_value is None or pd.isna(q_value):
        return "unknown"
    return str(int(round(float(q_value) * 100)))


def resolve_global_centralized_source(
    subset_row: pd.Series,
    csv_path: str,
) -> tuple[str, str]:
    """Use the explicit global graph, or the shared five-client base graph."""
    if pd.notna(subset_row.get("global_data_dir")):
        return str(subset_row["global_data_dir"]), str(subset_row["global_dataset_id"])

    registry = pd.read_csv(csv_path)
    needed_ids = parse_subset_clients(str(subset_row["subset_clients"]))
    rows = registry[registry["graph_id"].isin(needed_ids)]
    if rows.empty or rows["data_dir"].nunique() != 1:
        raise ValueError("Centralized training requires one explicit global graph.")
    source = rows.iloc[0]
    return str(source["data_dir"]), str(source["base_dataset_id"])


def load_global_centralized_client_from_subset_row(
    subset_row: pd.Series,
    csv_path: str,
    *,
    verbose: bool = True,
) -> ClientData:
    """Load one unsplit global graph with complete labels for either benchmark.

    Five-client setup:
      the original selected base graph is the global graph.

    Twenty-client setup:
      the connected planted graph is loaded from global_data_dir; community
      graphs and virtual-client masks are never used for this baseline.
    """
    data_dir, global_dataset_id = resolve_global_centralized_source(
        subset_row,
        csv_path,
    )

    q_tag = _q_tag_from_row(subset_row)
    pseudo_graph_id = f"global_{global_dataset_id}_q{q_tag}"
    pseudo_dataset_id = f"{global_dataset_id}_global_centralized_q{q_tag}"

    if verbose:
        print("\n" + "=" * 100)
        print("LOAD GLOBAL-CENTRALIZED GRAPH")
        print("=" * 100)
        print("global_dataset_id:", global_dataset_id)
        print("pseudo_graph_id:", pseudo_graph_id)
        print("pseudo_dataset_id:", pseudo_dataset_id)
        print("global_data_dir:", data_dir)
        print("q_tag:", q_tag)
        print(
            "IMPORTANT: one unsplit graph, mask_meta=None, full labels on all splits."
        )
        print("=" * 100)

    train_g, val_g, test_g = load_client_from_dir(data_dir)
    train_g, val_g, test_g = apply_mask_to_splits(
        train_g,
        val_g,
        test_g,
        mask_meta=None,
    )

    train_h = make_bidirected_hetero(train_g)
    val_h = make_bidirected_hetero(val_g)
    test_h = make_bidirected_hetero(test_g)

    train_h["n"].label_mask = train_g.label_mask.float()
    val_h["n"].label_mask = val_g.label_mask.float()
    test_h["n"].label_mask = test_g.label_mask.float()

    return ClientData(
        graph_id=str(pseudo_graph_id),
        data_dir=str(data_dir),
        dataset_id=str(pseudo_dataset_id),
        train_g=train_g,
        val_g=val_g,
        test_g=test_g,
        train_h=train_h,
        val_h=val_h,
        test_h=test_h,
        mask_meta=None,
    )


def load_clients(
    chosen_df: pd.DataFrame,
    csv_path: Optional[str] = None,
    verbose: bool = True,
    *,
    mask_splits=("train",),
) -> Dict[int, ClientData]:
    if csv_path is None:
        df_needed = chosen_df
    else:
        needed_graph_ids = collect_needed_client_ids(chosen_df)
        df_data = pd.read_csv(csv_path).copy()
        df_needed = df_data[df_data["graph_id"].isin(needed_graph_ids)].copy()

        if verbose:
            print(f"\nNeed to load {len(df_needed)} unique clients.")

    id_to_client: Dict[int, ClientData] = {}
    for _, row in df_needed.iterrows():

        mask_meta = mask_meta_from_row(row)

        graph_id = int(row["graph_id"])
        data_dir = row["data_dir"]
        dataset_id = row["dataset_id"]

        train_g, val_g, test_g = load_client_from_dir(data_dir)

        train_g, val_g, test_g = apply_mask_to_splits(
            train_g,
            val_g,
            test_g,
            mask_meta,
            mask_splits=mask_splits,
        )

        train_h = make_bidirected_hetero(train_g)
        val_h = make_bidirected_hetero(val_g)
        test_h = make_bidirected_hetero(test_g)

        train_h["n"].label_mask = train_g.label_mask.float()
        val_h["n"].label_mask = val_g.label_mask.float()
        test_h["n"].label_mask = test_g.label_mask.float()

        id_to_client[graph_id] = ClientData(
            graph_id=str(graph_id),
            data_dir=data_dir,
            dataset_id=dataset_id,
            train_g=train_g,
            val_g=val_g,
            test_g=test_g,
            train_h=train_h,
            val_h=val_h,
            test_h=test_h,
            mask_meta=mask_meta,
        )

    return id_to_client


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
