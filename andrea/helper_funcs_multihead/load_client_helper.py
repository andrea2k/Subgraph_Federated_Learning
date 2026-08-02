from __future__ import annotations

import torch
import time
import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from utils.train_utils import load_datasets
from utils.hetero import make_bidirected_hetero

from andrea.multigraph_generation import TASKS


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
        # Backward-compatible old name.
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


def apply_positive_label_mask(g, *, task: str, fraction: float, seed: int):
    g = attach_full_label_mask(g)

    task_idx = TASKS.index(task)
    y = g.y.float()

    positives = torch.where(y[:, task_idx] > 0.5)[0]
    k = int(round(float(fraction) * positives.numel()))

    if k > 0:
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(positives.numel(), generator=gen)
        hidden = positives[perm[:k]]
        g.label_mask[hidden, task_idx] = 0.0

    return g


def apply_keep_assigned_task_mask_other_labels(
    g,
    *,
    assigned_task: str,
    fraction: float,
    seed: int,
):
    """
    Task-specialized masking.

    assigned_task:
        all labels remain visible.

    non-assigned tasks:
        fraction of positive labels AND fraction of negative labels are hidden.
    """
    g = attach_full_label_mask(g)

    if assigned_task not in TASKS:
        raise ValueError(
            f"Unknown assigned_task={assigned_task}. Expected one of {TASKS}"
        )

    fraction = float(fraction)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"mask fraction must be in [0, 1], got {fraction}")

    y = g.y.float()

    for task_idx, task in enumerate(TASKS):
        if task == assigned_task:
            continue

        if fraction <= 0.0:
            continue

        positives = torch.where(y[:, task_idx] > 0.5)[0]
        negatives = torch.where(y[:, task_idx] <= 0.5)[0]

        k_pos = int(round(fraction * positives.numel()))
        k_neg = int(round(fraction * negatives.numel()))

        hidden_parts = []

        if k_pos > 0:
            gen_pos = torch.Generator().manual_seed(int(seed) + 1009 * task_idx + 17)
            perm_pos = torch.randperm(positives.numel(), generator=gen_pos)
            hidden_parts.append(positives[perm_pos[:k_pos]])

        if k_neg > 0:
            gen_neg = torch.Generator().manual_seed(int(seed) + 1009 * task_idx + 29)
            perm_neg = torch.randperm(negatives.numel(), generator=gen_neg)
            hidden_parts.append(negatives[perm_neg[:k_neg]])

        if hidden_parts:
            hidden = torch.cat(hidden_parts, dim=0)
            g.label_mask[hidden, task_idx] = 0.0

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


def _allocation_counts(num_items: int, shares: Sequence[float]) -> List[int]:
    """
    Convert fractional shares into exact integer counts that sum to num_items.

    Example:
      num_items = 244
      shares = [0.5, 0.125, 0.125, 0.125, 0.125]
      raw = [122, 30.5, 30.5, 30.5, 30.5]

    This returns something like:
      [122, 31, 31, 30, 30]

    So no positive label is lost or duplicated.
    """
    n = int(num_items)
    raw = [float(s) * n for s in shares]
    counts = [int(math.floor(x)) for x in raw]

    remainder = n - int(sum(counts))
    if remainder < 0:
        raise ValueError(f"allocation counts overfull: n={n}, counts={counts}")

    # Add leftover items to largest fractional parts.
    frac_order = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - math.floor(raw[i]), -i),
        reverse=True,
    )

    for i in frac_order[:remainder]:
        counts[i] += 1

    if sum(counts) != n:
        raise AssertionError(f"counts do not sum to {n}: {counts}")

    return counts


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

    return _allocation_counts(int(num_positives), shares)


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


def apply_mask_to_one_graph(g, mask_meta):
    """
    Apply the same label-mask rule to one split graph.

    Important:
    - The graph labels y are NOT deleted.
    - We only change g.label_mask.
    - Full/oracle evaluation can still ignore label_mask and use all y.
    - Realistic/visible evaluation can use label_mask.
    """
    g = attach_full_label_mask(g)

    if mask_meta is None:
        return g

    mode = mask_meta["mask_mode"]

    if mode == "drop_positive_labels":
        return apply_positive_label_mask(
            g,
            task=mask_meta["mask_task"],
            fraction=mask_meta["mask_fraction"],
            seed=mask_meta["mask_seed"],
        )

    if mode == "keep_assigned_task_mask_other_labels":
        return apply_keep_assigned_task_mask_other_labels(
            g,
            assigned_task=mask_meta["assigned_task"],
            fraction=mask_meta["mask_fraction"],
            seed=mask_meta["mask_seed"],
        )

    if mode == "q_task_label_allocation":
        return apply_q_task_label_allocation_mask(
            g,
            assigned_task=mask_meta["assigned_task"],
            q_value=mask_meta["q_value"],
            base_graph_id=mask_meta["base_graph_id"],
        )

    raise ValueError(f"Unknown mask_mode: {mode}")


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


def _base_ids_from_selected_subset_row(row: pd.Series) -> List[int]:
    """
    Extract the original unmasked base graph ID from a selected_subset row.
    Preferred source: base_graph_ids_json.
    Fallback source: membership_json.
    """
    if "base_graph_ids_json" in row.index and pd.notna(row["base_graph_ids_json"]):
        base_ids = [int(x) for x in _json_list(row["base_graph_ids_json"])]
        if base_ids:
            return sorted(set(base_ids))

    if "membership_json" in row.index and pd.notna(row["membership_json"]):
        membership = json.loads(str(row["membership_json"]))
        base_ids = sorted({int(x["base_graph_id"]) for x in membership})
        if base_ids:
            return base_ids

    raise ValueError(
        "Could not find base graph IDs. Expected base_graph_ids_json or membership_json."
    )


def _nonempty_row_value(row: pd.Series, key: str):
    if key not in row.index:
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def _graph_dir_is_complete(path: str | Path) -> bool:
    p = Path(path)
    return all((p / f"{split}.pt").exists() for split in ("train", "val", "test"))


def _strip_community_suffix(dataset_id: str) -> str:
    return re.sub(r"_community\d+$", "", str(dataset_id))


def _infer_planted_global_dir(data_dir: str, global_dataset_id: str) -> Path:
    """Infer the planted global folder from a community client path.

    Expected pair:
      andrea/planted_community_clients/<dataset>/community_k
      andrea/planted_community_data/<dataset>
    """
    p = Path(str(data_dir))
    parts = list(p.parts)
    if "planted_community_clients" in parts:
        idx = parts.index("planted_community_clients")
        prefix = Path(*parts[:idx]) if idx > 0 else Path(".")
        return prefix / "planted_community_data" / str(global_dataset_id)
    return Path("./andrea/planted_community_data") / str(global_dataset_id)


def resolve_global_centralized_source(
    subset_row: pd.Series,
    csv_path: str,
) -> tuple[str, str]:
    """Resolve the unsplit, unmasked graph used by Global-Centralized.

    Resolution order:
      1. Explicit global_data_dir/global_dataset_id from selected_subset.csv.
      2. Explicit global_data_dir/global_dataset_id copied into registry rows.
      3. Five-client fallback: the one original base graph is already global.
      4. Legacy planted-community fallback inferred from community data paths.
    """
    df_data = pd.read_csv(csv_path).copy()

    explicit_dir = _nonempty_row_value(subset_row, "global_data_dir")
    explicit_id = _nonempty_row_value(subset_row, "global_dataset_id")
    if explicit_dir is not None:
        global_id = explicit_id or Path(explicit_dir).name
        return str(explicit_dir), str(global_id)

    needed_ids = parse_subset_clients(str(subset_row["subset_clients"]))
    client_rows = df_data[
        pd.to_numeric(df_data["graph_id"], errors="coerce").isin(needed_ids)
    ].copy()

    if not client_rows.empty and "global_data_dir" in client_rows.columns:
        dirs = [
            str(v).strip()
            for v in client_rows["global_data_dir"].dropna().tolist()
            if str(v).strip()
        ]
        unique_dirs = sorted(set(dirs))
        if len(unique_dirs) == 1:
            if "global_dataset_id" in client_rows.columns:
                ids = [
                    str(v).strip()
                    for v in client_rows["global_dataset_id"].dropna().tolist()
                    if str(v).strip()
                ]
                unique_ids = sorted(set(ids))
                global_id = unique_ids[0] if len(unique_ids) == 1 else Path(unique_dirs[0]).name
            else:
                global_id = Path(unique_dirs[0]).name
            return unique_dirs[0], global_id

    base_ids = _base_ids_from_selected_subset_row(subset_row)

    # Five-client setup: one physical base graph is the unsplit global graph.
    if len(base_ids) == 1:
        base_graph_id = int(base_ids[0])
        candidates = pd.DataFrame()
        if "base_graph_id" in df_data.columns:
            base_col = pd.to_numeric(df_data["base_graph_id"], errors="coerce")
            candidates = df_data[base_col.eq(base_graph_id)].copy()
        if candidates.empty:
            graph_col = pd.to_numeric(df_data["graph_id"], errors="coerce")
            candidates = df_data[graph_col.eq(base_graph_id)].copy()
        if candidates.empty:
            raise ValueError(
                f"Could not find base graph {base_graph_id} in registry {csv_path}."
            )
        source_row = candidates.iloc[0]
        data_dir = str(source_row["data_dir"])
        base_dataset_id = _nonempty_row_value(source_row, "base_dataset_id")
        if base_dataset_id is None:
            base_dataset_id = str(source_row["dataset_id"])
        return data_dir, str(base_dataset_id)

    # Legacy 20-client manifest fallback. New generation writes explicit fields,
    # but this allows old manifests to be audited and migrated.
    global_ids = []
    if "membership_json" in subset_row.index and pd.notna(subset_row["membership_json"]):
        membership = json.loads(str(subset_row["membership_json"]))
        for member in membership:
            base_id = member.get("base_dataset_id")
            if base_id is not None:
                global_ids.append(_strip_community_suffix(str(base_id)))

    if not global_ids and not client_rows.empty:
        id_col = "base_dataset_id" if "base_dataset_id" in client_rows.columns else "dataset_id"
        global_ids.extend(
            _strip_community_suffix(str(v)) for v in client_rows[id_col].dropna().tolist()
        )

    unique_global_ids = sorted(set(global_ids))
    if len(unique_global_ids) != 1:
        raise ValueError(
            "Could not resolve one global dataset ID for Global-Centralized. "
            f"Candidates={unique_global_ids}. Regenerate the manifest with global_data_dir."
        )

    global_dataset_id = unique_global_ids[0]
    if client_rows.empty:
        raise ValueError("Could not inspect client data paths for global-path inference.")
    inferred = _infer_planted_global_dir(
        str(client_rows.iloc[0]["data_dir"]),
        global_dataset_id,
    )
    return str(inferred), str(global_dataset_id)


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

    if not _graph_dir_is_complete(data_dir):
        raise FileNotFoundError(
            f"Global graph directory is incomplete: {data_dir}. "
            "Expected train.pt, val.pt, and test.pt."
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
        print("IMPORTANT: one unsplit graph, mask_meta=None, full labels on all splits.")
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


def audit_global_centralized_client(
    client: ClientData,
    *,
    strict: bool = True,
) -> None:
    """Assert that Global-Centralized is one fully supervised graph."""
    print("\n" + "=" * 100)
    print("AUDIT GLOBAL-CENTRALIZED CLIENT")
    print("=" * 100)
    print("graph_id:", client.graph_id)
    print("dataset_id:", client.dataset_id)
    print("data_dir:", client.data_dir)
    print("mask_meta:", client.mask_meta)

    _assert_or_warn(
        client.mask_meta is None,
        "global-centralized client should have mask_meta=None",
        strict=strict,
    )
    for split_name, graph in (
        ("train", client.train_g),
        ("val", client.val_g),
        ("test", client.test_g),
    ):
        _assert_or_warn(
            _mask_is_all_ones(graph),
            f"global-centralized {split_name} label_mask is not all ones",
            strict=strict,
        )
        _assert_or_warn(
            int(graph.y.size(1)) == len(TASKS),
            f"global-centralized {split_name} has {graph.y.size(1)} tasks, expected {len(TASKS)}",
            strict=strict,
        )
        print(
            f"{split_name}: nodes={int(graph.num_nodes)} "
            f"edges={int(graph.edge_index.size(1))} y={tuple(graph.y.shape)}"
        )

    counts = _mask_debug_counts(client.train_g)
    print("train visible_pos:", dict(zip(TASKS, counts["visible_pos"])))
    print("train hidden_pos :", dict(zip(TASKS, counts["hidden_pos"])))
    print("train hidden_neg :", dict(zip(TASKS, counts["hidden_neg"])))
    print("PASS global-centralized audit")


# Backward-compatible aliases. Existing imports continue to work, but semantics
# are now explicitly Global-Centralized rather than community-centralized.
def load_local_centralized_client_from_subset_row(
    subset_row: pd.Series,
    csv_path: str,
    *,
    verbose: bool = True,
) -> ClientData:
    return load_global_centralized_client_from_subset_row(
        subset_row,
        csv_path,
        verbose=verbose,
    )


def audit_local_centralized_client(client: ClientData, *, strict: bool = True) -> None:
    audit_global_centralized_client(client, strict=strict)


def _assert_or_warn(condition: bool, message: str, *, strict: bool) -> None:
    if condition:
        return
    if strict:
        raise AssertionError(message)
    print("WARNING:", message)


def _mask_is_all_ones(g) -> bool:
    if not hasattr(g, "label_mask"):
        return False
    return bool(torch.all(g.label_mask.float() == 1.0).item())


def _mask_debug_counts(g) -> Dict[str, List[int]]:
    y = g.y.float()
    mask = g.label_mask.float()

    pos = y > 0.5
    neg = ~pos
    visible = mask > 0.5
    hidden = ~visible

    return {
        "total_pos": pos.sum(dim=0).to(torch.long).tolist(),
        "visible_pos": (pos & visible).sum(dim=0).to(torch.long).tolist(),
        "hidden_pos": (pos & hidden).sum(dim=0).to(torch.long).tolist(),
        "total_neg": neg.sum(dim=0).to(torch.long).tolist(),
        "visible_neg": (neg & visible).sum(dim=0).to(torch.long).tolist(),
        "hidden_neg": (neg & hidden).sum(dim=0).to(torch.long).tolist(),
    }


def audit_q_positive_disjointness_for_split(
    subset_clients: List[ClientData],
    *,
    split_name: str,
    q_value: float,
    strict: bool = True,
    max_bad_examples: int = 20,
) -> None:
    """
    Check that for each task and each positive node-label event:

        sum_client label_mask[node, task] == 1

    This means:
      - no positive node-label event is duplicated across clients;
      - no positive node-label event is lost;
      - the q allocation exactly partitions the positive indices.

    Example:
      For cycle2, if node 123 is positive for cycle2, then exactly one
      virtual client should have label_mask[123, cycle2] = 1.
    """
    split_graphs = [getattr(client, f"{split_name}_g") for client in subset_clients]

    reference_y = split_graphs[0].y.float()
    n_nodes = int(reference_y.size(0))

    print("\n" + "=" * 100)
    print(f"Q POSITIVE DISJOINTNESS AUDIT | split={split_name}")
    print("=" * 100)
    print("q_value:", q_value)
    print("clients:", "|".join(str(c.graph_id) for c in subset_clients))

    # Check all virtual clients have same split graph size and same y.
    for client, g in zip(subset_clients, split_graphs):
        _assert_or_warn(
            int(g.y.size(0)) == n_nodes,
            (
                f"{split_name}: client {client.graph_id} has different num nodes: "
                f"{int(g.y.size(0))} vs reference {n_nodes}"
            ),
            strict=strict,
        )

        same_y = bool(torch.equal(g.y.float(), reference_y))
        _assert_or_warn(
            same_y,
            (
                f"{split_name}: client {client.graph_id} has different y tensor "
                "from reference client. Disjointness check assumes same base graph."
            ),
            strict=strict,
        )

    for task_idx, label_task in enumerate(TASKS):
        positives = torch.where(reference_y[:, task_idx] > 0.5)[0]
        n_pos = int(positives.numel())

        masks_for_task = []
        actual_by_assigned: Dict[str, int] = {}

        for client, g in zip(subset_clients, split_graphs):
            assigned_task = str((client.mask_meta or {}).get("assigned_task"))
            mask = g.label_mask.float()

            # This is the vector over positive nodes only.
            # Shape: [num_positive_nodes_for_this_task]
            mask_pos = mask[positives, task_idx] > 0.5

            masks_for_task.append(mask_pos)
            actual_by_assigned[assigned_task] = int(mask_pos.sum().item())

        if n_pos == 0:
            print(f"task={label_task}: no positives in split={split_name}")
            continue

        # Shape: [num_clients, num_positive_nodes]
        stacked = torch.stack(masks_for_task, dim=0)

        # For every positive node-label event, how many clients can see it?
        # Correct answer should be exactly 1 for every positive.
        per_positive_visible_count = stacked.sum(dim=0)

        lost_mask = per_positive_visible_count == 0
        duplicated_mask = per_positive_visible_count > 1
        ok_mask = per_positive_visible_count == 1

        lost_count = int(lost_mask.sum().item())
        duplicated_count = int(duplicated_mask.sum().item())
        ok_count = int(ok_mask.sum().item())

        min_seen = int(per_positive_visible_count.min().item())
        max_seen = int(per_positive_visible_count.max().item())
        visible_sum = int(stacked.sum().item())

        expected_counts = _q_counts_for_label_task(
            n_pos,
            label_task=label_task,
            q_value=q_value,
        )
        expected_by_assigned = {
            task: int(expected_counts[TASKS.index(task)]) for task in TASKS
        }

        print("\n" + "-" * 80)
        print(f"split={split_name} | task={label_task}")
        print("-" * 80)
        print("total positive node-label events :", n_pos)
        print("visible_sum across all clients    :", visible_sum)
        print("ok_count exactly once             :", ok_count)
        print("lost_count visible zero times     :", lost_count)
        print("duplicated_count visible >1 times :", duplicated_count)
        print("per_positive_min_seen             :", min_seen)
        print("per_positive_max_seen             :", max_seen)
        print("actual_by_assigned                :", actual_by_assigned)
        print("expected_by_assigned              :", expected_by_assigned)

        if lost_count > 0:
            lost_indices = positives[lost_mask][:max_bad_examples].tolist()
            print("EXAMPLE LOST POSITIVE NODE INDICES:", lost_indices)

        if duplicated_count > 0:
            duplicated_indices = positives[duplicated_mask][:max_bad_examples].tolist()
            print("EXAMPLE DUPLICATED POSITIVE NODE INDICES:", duplicated_indices)

            # Print which clients see the first duplicated examples.
            for node_idx in duplicated_indices:
                visible_clients = []
                for client, g in zip(subset_clients, split_graphs):
                    if bool(g.label_mask[int(node_idx), task_idx].float().item() > 0.5):
                        visible_clients.append(
                            f"{client.graph_id}/assigned={client.mask_meta.get('assigned_task')}"
                        )
                print(
                    f"duplicated node={node_idx}, task={label_task}, "
                    f"visible_clients={visible_clients}"
                )

        _assert_or_warn(
            lost_count == 0,
            (
                f"{split_name} task={label_task}: {lost_count} positive events "
                "are lost, i.e. visible in zero clients."
            ),
            strict=strict,
        )

        _assert_or_warn(
            duplicated_count == 0,
            (
                f"{split_name} task={label_task}: {duplicated_count} positive events "
                "are duplicated, i.e. visible in more than one client."
            ),
            strict=strict,
        )

        _assert_or_warn(
            visible_sum == n_pos,
            (
                f"{split_name} task={label_task}: visible_sum={visible_sum}, "
                f"but n_pos={n_pos}. Expected exact partition."
            ),
            strict=strict,
        )

        _assert_or_warn(
            actual_by_assigned == expected_by_assigned,
            (
                f"{split_name} task={label_task}: actual_by_assigned="
                f"{actual_by_assigned}, expected_by_assigned={expected_by_assigned}"
            ),
            strict=strict,
        )

    print("\nPASS Q POSITIVE DISJOINTNESS AUDIT | split=", split_name)
    print("=" * 100)


def print_client_debug_summary(
    client: ClientData,
    *,
    name: str = "",
) -> None:
    """
    Debug one loaded ClientData object.

    This checks:
      - graph/dataset identity
      - whether this is masked or unmasked
      - x/y shapes
      - visible/hidden labels per split
      - whether val/test masks are full visibility
    """
    title = f"CLIENT DEBUG SUMMARY | {name}" if name else "CLIENT DEBUG SUMMARY"

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print("graph_id     :", client.graph_id)
    print("dataset_id   :", client.dataset_id)
    print("data_dir     :", client.data_dir)
    print("mask_meta    :", client.mask_meta)

    if client.mask_meta is None:
        print("mask type    : UNMASKED / LOCAL-CENTRALIZED CANDIDATE")
    else:
        print("mask type    : MASKED / FULLY-LOCAL CLIENT")
        print("mask_mode    :", client.mask_meta.get("mask_mode"))
        print("assigned_task:", client.mask_meta.get("assigned_task"))
        print("q_value      :", client.mask_meta.get("q_value"))
        print("base_graph_id:", client.mask_meta.get("base_graph_id"))

    for split_name, g in [
        ("train", client.train_g),
        ("val", client.val_g),
        ("test", client.test_g),
    ]:
        counts = _mask_debug_counts(g)

        print("\n" + "-" * 80)
        print(f"{split_name.upper()} GRAPH")
        print("-" * 80)
        print("num_nodes          :", getattr(g, "num_nodes", None))
        print("x shape            :", tuple(g.x.shape))
        print("y shape            :", tuple(g.y.shape))
        print("label_mask shape   :", tuple(g.label_mask.shape))
        print("label_mask all ones:", _mask_is_all_ones(g))

        print("total_pos  :", dict(zip(TASKS, counts["total_pos"])))
        print("visible_pos:", dict(zip(TASKS, counts["visible_pos"])))
        print("hidden_pos :", dict(zip(TASKS, counts["hidden_pos"])))
        print("hidden_neg :", dict(zip(TASKS, counts["hidden_neg"])))

    print("=" * 100)


def audit_loaded_q_label_masks(
    chosen_df: pd.DataFrame,
    id_to_client: Dict[int, ClientData],
    *,
    strict: bool = True,
    max_groups: Optional[int] = None,
    mask_splits=("train",),
) -> None:
    """Audit real q masks for both the 5-client and 20-client benchmarks.

    The selected subset may contain either:
      - one physical graph with five task-specialist virtual clients; or
      - several physical community graphs, each with its own five virtual clients.

    Exact q disjointness is therefore checked independently inside every
    ``base_graph_id`` group. Positive labels from different physical graphs must
    never be compared as though they belonged to one shared node set.
    """
    print("\n" + "=" * 100)
    print("AUDIT LOADED Q LABEL MASKS")
    print("=" * 100)

    audited_subsets = 0
    audited_physical_groups = 0
    stop = False

    for row_idx, row in chosen_df.iterrows():
        subset_ids = parse_subset_clients(row["subset_clients"])
        subset_clients = [id_to_client[int(gid)] for gid in subset_ids]

        modes = {(client.mask_meta or {}).get("mask_mode") for client in subset_clients}
        if "q_task_label_allocation" not in modes:
            print(
                f"skip row={row_idx}: no q_task_label_allocation mode "
                f"(modes={sorted(str(x) for x in modes)})"
            )
            continue

        q_values = {
            float((client.mask_meta or {}).get("q_value"))
            for client in subset_clients
        }
        _assert_or_warn(
            len(q_values) == 1,
            f"subset row={row_idx} has multiple q values: {q_values}",
            strict=strict,
        )
        q_value = sorted(q_values)[0]

        clients_by_base: Dict[int, List[ClientData]] = {}
        for client in subset_clients:
            meta = client.mask_meta or {}
            base_graph_id = meta.get("base_graph_id")
            _assert_or_warn(
                base_graph_id is not None,
                f"client {client.graph_id} has no base_graph_id in mask metadata",
                strict=strict,
            )
            if base_graph_id is None:
                continue
            clients_by_base.setdefault(int(base_graph_id), []).append(client)

        _assert_or_warn(
            bool(clients_by_base),
            f"subset row={row_idx} produced no physical base-graph groups",
            strict=strict,
        )

        audited_subsets += 1
        print("\n" + "#" * 100)
        print(
            f"subset row={row_idx} | q={q_value} | virtual_clients={len(subset_clients)} | "
            f"physical_groups={len(clients_by_base)}"
        )
        print("#" * 100)

        for base_graph_id, group_clients in sorted(clients_by_base.items()):
            if max_groups is not None and audited_physical_groups >= int(max_groups):
                stop = True
                break

            audited_physical_groups += 1
            assigned_tasks = [
                str((client.mask_meta or {}).get("assigned_task"))
                for client in group_clients
            ]

            print("\n" + "-" * 100)
            print(
                f"physical base_graph_id={base_graph_id} | q={q_value} | "
                f"clients={'|'.join(str(c.graph_id) for c in group_clients)}"
            )
            print("assigned_tasks:", assigned_tasks)
            print("data_dirs:", sorted({str(c.data_dir) for c in group_clients}))
            print("-" * 100)

            _assert_or_warn(
                len(group_clients) == len(TASKS),
                (
                    f"base_graph_id={base_graph_id}: expected {len(TASKS)} virtual "
                    f"clients, got {len(group_clients)}"
                ),
                strict=strict,
            )
            _assert_or_warn(
                sorted(assigned_tasks) == sorted(TASKS),
                (
                    f"base_graph_id={base_graph_id}: assigned tasks should contain "
                    f"each task exactly once; got {assigned_tasks}"
                ),
                strict=strict,
            )
            _assert_or_warn(
                len({str(c.data_dir) for c in group_clients}) == 1,
                (
                    f"base_graph_id={base_graph_id}: its virtual clients point to "
                    "different physical data directories"
                ),
                strict=strict,
            )

            for client in group_clients:
                meta = client.mask_meta or {}
                for split_name in ("train", "val", "test"):
                    split_g = getattr(client, f"{split_name}_g")
                    if split_name in mask_splits:
                        counts = _mask_debug_counts(split_g)
                        hidden_neg_total = int(sum(counts["hidden_neg"]))
                        _assert_or_warn(
                            hidden_neg_total == 0,
                            (
                                f"client {client.graph_id} {split_name} hides negatives, "
                                "but q masking should hide positive labels only. "
                                f"hidden_neg={counts['hidden_neg']}"
                            ),
                            strict=strict,
                        )
                        print(
                            f"{split_name} mask check:",
                            "client:", client.graph_id,
                            "assigned_task:", meta.get("assigned_task"),
                            "visible_pos:", dict(zip(TASKS, counts["visible_pos"])),
                            "hidden_pos:", dict(zip(TASKS, counts["hidden_pos"])),
                            "hidden_neg_total:", hidden_neg_total,
                        )
                    else:
                        _assert_or_warn(
                            _mask_is_all_ones(split_g),
                            f"client {client.graph_id} {split_name} label_mask is not all ones",
                            strict=strict,
                        )

            for split_name in ("train", "val", "test"):
                if split_name not in mask_splits:
                    continue
                audit_q_positive_disjointness_for_split(
                    group_clients,
                    split_name=split_name,
                    q_value=q_value,
                    strict=strict,
                )

            print(
                f"PASS q-mask group: base_graph_id={base_graph_id}, "
                f"clients={len(group_clients)}, q={q_value}"
            )

        if stop:
            break

    if audited_physical_groups == 0:
        print("No q_task_label_allocation physical groups were audited.")
    else:
        print("\n" + "=" * 100)
        print(
            "ALL LOADED Q LABEL MASK AUDITS PASSED | "
            f"audited_subsets={audited_subsets} | "
            f"audited_physical_groups={audited_physical_groups}"
        )
        print("=" * 100)


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


class ProgressPrinter:
    def __init__(self, total_runs: int):
        self.total_runs = total_runs
        self.done = 0

    def step(self, label: str, run_start_time: float | None = None):
        self.done += 1
        print(label)
        if run_start_time is not None:
            run_elapsed = time.perf_counter() - run_start_time
            print(f"run time: {format_seconds(run_elapsed)}")
        print("=======================================================")
        print(
            f"==================== {self.done} / {self.total_runs} ===================="
        )
        print("=======================================================")
