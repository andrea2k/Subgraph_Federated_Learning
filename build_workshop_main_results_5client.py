#!/usr/bin/env python3
"""Build the workshop paper's five-client main-results figure.

This script reads the frozen thesis numeric ledger. It does not modify the
ledger, experiment results, or existing thesis figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "andrea/final_prauc_analysis/outputs/"
    "final_thesis_numeric_ledger/01_main_vo_macro_results.csv"
)
DEFAULT_OUTPUT = Path("figures/main_results_5client.pdf")

METHODS = (
    "centralized",
    "fedavg",
    "fedprox",
    "fedala",
    "apple",
    "support_simplex",
)

DISPLAY = {
    "centralized": "Global-Centralized",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedala": "FedALA",
    "apple": "APPLE",
    "support_simplex": "FedPTHA",
}

Q_VALUES = (0.20, 0.50, 0.80, 0.95)

tab10 = plt.get_cmap("tab10").colors

STYLE = {
    "centralized": {
        "color": "0.38",
        "marker": None,
        "linestyle": "--",
        "linewidth": 1.35,
        "markersize": 0,
        "alpha": 0.90,
        "zorder": 1,
    },
    "fedavg": {
        "color": tab10[0],
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.55,
        "markersize": 5.2,
        "alpha": 0.95,
        "zorder": 3,
    },
    "fedprox": {
        "color": tab10[1],
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.55,
        "markersize": 5.4,
        "alpha": 0.95,
        "zorder": 3,
    },
    "fedala": {
        "color": tab10[4],
        "marker": "v",
        "linestyle": "-",
        "linewidth": 1.55,
        "markersize": 5.4,
        "alpha": 0.95,
        "zorder": 3,
    },
    "apple": {
        "color": tab10[5],
        "marker": "P",
        "linestyle": "-",
        "linewidth": 1.55,
        "markersize": 5.2,
        "alpha": 0.95,
        "zorder": 3,
    },
    "support_simplex": {
        "color": tab10[3],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.15,
        "markersize": 6.2,
        "alpha": 1.0,
        "zorder": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the five-client main-results figure for the workshop paper."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Frozen result CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output PDF.",
    )
    return parser.parse_args()


def load_plot_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"ABORT: missing frozen input CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "setup",
        "method",
        "q",
        "macro_pr_auc_mean_percent",
        "macro_pr_auc_sample_sd_percent",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"ABORT: input CSV is missing columns: {missing}")

    df["q"] = pd.to_numeric(df["q"], errors="raise").round(2)
    plot = df[
        (df["setup"].astype(str) == "five_client")
        & df["method"].isin(METHODS)
    ].copy()

    if len(plot) != len(METHODS) * len(Q_VALUES):
        raise SystemExit(
            "ABORT: expected 24 five-client plotting rows "
            f"(6 methods x 4 q values), found {len(plot)}"
        )

    if plot.duplicated(["method", "q"]).any():
        raise SystemExit("ABORT: duplicate method x q rows in plotting data")

    if set(plot["method"]) != set(METHODS):
        raise SystemExit("ABORT: selected method registry does not match")

    if set(plot["q"]) != set(Q_VALUES):
        raise SystemExit("ABORT: q grid does not match 0.20, 0.50, 0.80, 0.95")

    numeric_columns = (
        "macro_pr_auc_mean_percent",
        "macro_pr_auc_sample_sd_percent",
    )
    for column in numeric_columns:
        plot[column] = pd.to_numeric(plot[column], errors="raise")
        if not np.isfinite(plot[column]).all():
            raise SystemExit(f"ABORT: non-finite values in {column}")

    if not plot["macro_pr_auc_mean_percent"].between(0, 100).all():
        raise SystemExit("ABORT: Macro PR-AUC means are outside [0, 100]")

    if (plot["macro_pr_auc_sample_sd_percent"] < 0).any():
        raise SystemExit("ABORT: negative sample standard deviation detected")

    return plot.sort_values(["method", "q"]).reset_index(drop=True)


def build_figure(plot: pd.DataFrame, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise SystemExit(
            f"ABORT: output already exists: {output}\n"
            "Re-run with --overwrite only if replacement is intentional."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.45))

    for method in METHODS:
        group = plot[plot["method"] == method].sort_values("q")
        style = STYLE[method]
        x = group["q"].to_numpy()
        y = group["macro_pr_auc_mean_percent"].to_numpy()
        yerr = group["macro_pr_auc_sample_sd_percent"].to_numpy()

        if method == "centralized":
            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=DISPLAY[method],
            )
        else:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                capsize=2.5,
                elinewidth=1.0,
                label=DISPLAY[method],
            )

    ax.set_xticks(Q_VALUES)
    ax.set_xticklabels(["0.20", "0.50", "0.80", "0.95"])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("Macro PR-AUC (%)")
    ax.set_ylim(20, 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.99,
        top=0.98,
        bottom=0.25,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot = load_plot_data(args.input)
    build_figure(plot, args.output, args.overwrite)
    print(f"Created: {args.output}")
    print("Setup: five_client")
    print("Methods: Global-Centralized, FedAvg, FedProx, FedALA, APPLE, FedPTHA")
    print("Y-axis: Macro PR-AUC (%)")
    print("Frozen input was read only; no thesis figure was modified.")


if __name__ == "__main__":
    main()
