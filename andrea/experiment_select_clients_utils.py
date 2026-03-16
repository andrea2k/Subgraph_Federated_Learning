from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from utils.train_utils import load_datasets
from utils.hetero import make_bidirected_hetero


@dataclass
class ClientData:
    client_id: str
    data_dir: str
    train_g: object
    val_g: object
    test_g: object
    train_h: object
    val_h: object
    test_h: object


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


def zscore_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-12
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        out[c + "_z"] = (out[c] - mu) / (sd + eps)
    return out


def pick_controlled_low_mid_high(
    df: pd.DataFrame,
    target_metric: str,
    control_metrics: List[str],
) -> pd.DataFrame:
    """
    Pick low/mid/high for target_metric while penalizing drift in control_metrics.

    Strategy:
    - z-score metrics
    - define target anchors at 10%, 50%, 90% quantiles of target z-score
    - choose row closest to each anchor
    """
    work = zscore_df(df, [target_metric] + control_metrics).copy()

    tz = target_metric + "_z"
    control_z_cols = [m + "_z" for m in control_metrics]

    q_low = work[tz].quantile(0.10)
    q_mid = work[tz].quantile(0.50)
    q_high = work[tz].quantile(0.90)

    targets = {
        "low": q_low,
        "mid": q_mid,
        "high": q_high,
    }

    chosen_rows = []
    used_idx = set()

    for level, anchor in targets.items():
        cand = work.copy()

        # target closeness
        cand["_score"] = (cand[tz] - anchor).abs()
        for cz in control_z_cols:
            cand["_score"] += cand[cz].abs()

        # avoid reusing the same subset
        if used_idx:
            cand = cand.loc[~cand.index.isin(used_idx)].copy()

        best_idx = cand["_score"].idxmin()
        used_idx.add(best_idx)

        row = work.loc[best_idx].copy()
        row["level"] = level
        row["target_metric"] = target_metric
        row["control_metrics"] = control_metrics
        chosen_rows.append(row)

    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def choose_subsets(
    df_heterogeneity: pd.DataFrame,
    target_metrics: Sequence[str],
    preview_cols: Sequence[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print(df_heterogeneity.columns.tolist())

    chosen_all = []

    for target_metric in target_metrics:
        control_metrics = [m for m in target_metrics if m != target_metric]
        chosen = pick_controlled_low_mid_high(
            df=df_heterogeneity,
            target_metric=target_metric,
            control_metrics=control_metrics,
        )
        chosen_all.append(chosen)

    chosen_df = pd.concat(chosen_all, axis=0).reset_index(drop=True)

    if verbose:
        if preview_cols is None:
            preview_cols = [
                "subset_size",
                "target_metric",
                "level",
                "subset_id",
                "subset_clients",
            ]
        print("\nChosen subsets:")
        print(chosen_df[list(preview_cols)])

    return chosen_df


def collect_needed_client_ids(chosen_df: pd.DataFrame) -> List[int]:
    needed_client_ids = set()
    for s in chosen_df["subset_clients"]:
        needed_client_ids.update(parse_subset_clients(s))
    return sorted(needed_client_ids)


def load_needed_clients(
    chosen_df: pd.DataFrame,
    csv_path: str,
    verbose: bool = True,
) -> Dict[int, ClientData]:
    needed_client_ids = collect_needed_client_ids(chosen_df)
    df_data = pd.read_csv(csv_path).copy()
    df_needed = df_data[df_data["graph_id"].isin(needed_client_ids)].copy()

    if verbose:
        print(f"\nNeed to load {len(df_needed)} unique clients.")

    id_to_client: Dict[int, ClientData] = {}

    for _, row in df_needed.iterrows():
        cid = int(row["graph_id"])
        data_dir = row["data_dir"]

        tr, va, te = load_client_from_dir(data_dir)
        train_h = make_bidirected_hetero(tr)
        val_h = make_bidirected_hetero(va)
        test_h = make_bidirected_hetero(te)

        id_to_client[cid] = ClientData(
            client_id=str(cid),
            data_dir=data_dir,
            train_g=tr,
            val_g=va,
            test_g=te,
            train_h=train_h,
            val_h=val_h,
            test_h=test_h,
        )

    return id_to_client


def choose_subsets_and_load_clients(
    df_heterogeneity: pd.DataFrame,
    target_metrics: Sequence[str],
    csv_path: str,
    preview_cols: Sequence[str] | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, ClientData]]:
    chosen_df = choose_subsets(
        df_heterogeneity=df_heterogeneity,
        target_metrics=target_metrics,
        preview_cols=preview_cols,
        verbose=verbose,
    )
    id_to_client = load_needed_clients(
        chosen_df=chosen_df,
        csv_path=csv_path,
        verbose=verbose,
    )
    return chosen_df, id_to_client
