#!/usr/bin/env python3
import os
import csv
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

def compute_label_percentages(
    input_csv = "./data/y_sums.csv",
    output_csv = "./results/metrics/label_percentages.csv",
    add_mean = True
):
    '''
    Read label totals from `input_csv`compute per-split 
    percentages for each task, optionally append
    a 'mean_over_splits' row, and save to CSV.
    '''
    df = pd.read_csv(input_csv)
    if "total" not in df.columns:
        raise ValueError("Input CSV must include a 'total' column.")

    task_cols = [c for c in df.columns if c != "total"]
    if not task_cols:
        raise ValueError("No task columns found (columns other than 'total').")

    # per-split percentages
    pct = (df[task_cols].div(df["total"], axis=0) * 100.0).round(2)
    pct.index = [f"split_{i+1}" for i in range(len(pct))]

    if add_mean:
        pct.loc["mean_over_splits"] = pct.mean(axis=0).round(2)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pct.to_csv(output_csv, index=True)

    return pct


def _fmt_runtime_hhmmss(seconds: float) -> str:
    # Robust, clamps negatives to 0
    seconds = max(0.0, float(seconds))
    td = timedelta(seconds=round(seconds))
    # Format as HH:MM:SS (e.g., "01:23:45")
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def append_f1_score_to_csv(
    out_csv: str,
    tasks: list[str],
    mean_f1,
    std_f1,
    macro_mean_percent: float,
    seeds: list[int],
    model_name: str = "PNA baseline",
    runtime_seconds: float | None = None,   # <— NEW (optional)
):
    """
    Append a single row with mean/std per task (in %), macro mean (in %), runtime, and metadata.
    Creates the CSV (with header) if it doesn't exist.
    If the CSV exists without a 'runtime' column, this will append that column for new rows.
    """
    # Ensure directory exists (if any)
    dir_ = os.path.dirname(out_csv)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    # Build the default header for this run
    mean_cols = [f"{t}_mean_pct" for t in tasks]
    std_cols  = [f"{t}_std_pct"  for t in tasks]
    default_header = (
        ["timestamp_iso", "model", "n_runs", "seeds", "macro_mean_pct"]
        + mean_cols + std_cols + ["runtime"]  # <— include runtime by default
    )

    # See if file already exists and if it has a header; keep compatibility
    file_exists = os.path.exists(out_csv)
    if file_exists:
        existing_header = None
        try:
            with open(out_csv, "r", newline="") as f_in:
                r = csv.reader(f_in)
                existing_header = next(r, None)
        except Exception:
            existing_header = None

        if existing_header:
            # Ensure 'runtime' is present at least at the end
            header = list(existing_header)
            if "runtime" not in header:
                header = header + ["runtime"]
        else:
            header = default_header
    else:
        header = default_header

    # Prepare row values
    mean_pct = (mean_f1 * 100.0).tolist()
    std_pct  = (std_f1  * 100.0).tolist()

    row = {
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "n_runs": len(seeds),
        "seeds": ",".join(map(str, seeds)),
        "macro_mean_pct": round(macro_mean_percent, 2),
        **{c: round(v, 2) for c, v in zip(mean_cols, mean_pct)},
        **{c: round(v, 2) for c, v in zip(std_cols,  std_pct)},
    }

    # Add runtime field (string "HH:MM:SS"); if missing, leave empty
    row["runtime"] = _fmt_runtime_hhmmss(runtime_seconds) if runtime_seconds is not None else ""

    # Write (create header if file doesn't exist)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        # If the existing file lacked 'runtime', DictWriter will write an extra column at the end
        # (older rows won't have it, which is OK for CSV readers).
        w.writerow(row)


def start_epoch_csv(model_name: str,
                    seed: int,
                    tasks: list,
                    out_dir: str = "./results/metrics/epoch_logs") -> str:
    """
    Creates a timestamped CSV for per-epoch metrics and writes the header.
    Returns the full path to the CSV file.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{model_name}_seed{seed}.csv"
    path = os.path.join(out_dir, fname)

    header = ["epoch", "train_loss", "val_loss", "val_macro_minF1"] + [f"val_{t}_minF1" for t in tasks]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# model_name={model_name}"])
        w.writerow([f"# seed={seed}"])
        w.writerow([f"# started_at={ts}"])
        w.writerow(header)

    return path


def append_epoch_csv(csv_path: str,
                     epoch: int,
                     train_loss: float,
                     val_loss: float,
                     val_f1_tensor) -> None:
    """
    Appends a single epoch row to the CSV. val_f1_tensor is shape [num_tasks].
    """
    if hasattr(val_f1_tensor, "detach"):
        vals = val_f1_tensor.detach().cpu().tolist()
    else:
        vals = list(val_f1_tensor)
    macro = float(sum(vals) / len(vals))

    row = [int(epoch), float(train_loss), float(val_loss), float(macro)] + [float(v) for v in vals]

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)