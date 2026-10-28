#!/usr/bin/env python3

from __future__ import annotations

from fedptha.paths import generated_root, figures_root

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(str(generated_root()))
INPUT = ROOT / "tables" / "ablation_paired_contrasts.csv"
FIG_DIR = ROOT / str(figures_root())

OUT_PDF = FIG_DIR / "figure5_ablations.pdf"
OUT_PNG = FIG_DIR / "figure5_ablations.png"

Q_ORDER = [0.20, 0.50, 0.80, 0.95]

CONTRAST_ORDER = [
    "support_simplex_postala_minus_learned_simplex_postala",
    "support_simplex_postala_minus_support_only_postala",
    "support_simplex_postala_minus_support_simplex",
]

DISPLAY = {
    "support_simplex_postala_minus_learned_simplex_postala": "Positive-label distribution component",
    "support_simplex_postala_minus_support_only_postala": "Learned source-weight component",
    "support_simplex_postala_minus_support_simplex": "PostALA refinement",
}

SETUP_ORDER = ["five_client", "twenty_client"]

SETUP_TITLE = {
    "five_client": "(a) Five-client setting",
    "twenty_client": "(b) Twenty-client setting",
}


def fail(message: str) -> None:
    raise SystemExit(f"\nABORT: {message}\n")


def choose_column(df: pd.DataFrame, candidates, label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    fail(
        f"Could not identify {label} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


if not INPUT.is_file():
    fail(f"Missing input table: {INPUT}")

df = pd.read_csv(INPUT)

setup_col = choose_column(df, ["setup", "benchmark_setup"], "setup")
q_col = choose_column(df, ["q", "q_value", "gamma"], "q")
contrast_col = choose_column(
    df, ["contrast", "contrast_name", "comparison"], "contrast"
)
mean_col = choose_column(
    df,
    [
        "difference_mean_pp",
        "difference_mean",
        "mean_delta",
        "delta_mean",
        "mean_difference",
        "mean_diff",
        "mean",
    ],
    "mean contrast",
)
sd_col = choose_column(
    df,
    [
        "difference_sample_sd_pp",
        "difference_sample_sd",
        "sd_delta",
        "delta_sd",
        "sample_sd",
        "sd_difference",
        "sd_diff",
        "std",
        "sd",
    ],
    "sample-SD contrast",
)

protocol_col = None
for candidate in ["protocol", "eval_protocol"]:
    if candidate in df.columns:
        protocol_col = candidate
        break

if protocol_col is not None:
    df = df[df[protocol_col].astype(str).str.upper() == "VO"].copy()

df[q_col] = pd.to_numeric(df[q_col], errors="raise").round(2)
df[mean_col] = pd.to_numeric(df[mean_col], errors="raise")
df[sd_col] = pd.to_numeric(df[sd_col], errors="raise")

df = df[
    df[setup_col].isin(SETUP_ORDER)
    & df[q_col].isin(Q_ORDER)
    & df[contrast_col].isin(CONTRAST_ORDER)
].copy()

expected_rows = len(SETUP_ORDER) * len(Q_ORDER) * len(CONTRAST_ORDER)
if len(df) != expected_rows:
    print("Filtered rows:")
    print(
        df[[setup_col, q_col, contrast_col, mean_col, sd_col]]
        .sort_values([setup_col, q_col, contrast_col])
        .to_string(index=False)
    )
    fail(f"Expected {expected_rows} Figure-5 rows after filtering, found {len(df)}.")

dupes = df.duplicated([setup_col, q_col, contrast_col], keep=False)
if dupes.any():
    print(df.loc[dupes].to_string(index=False))
    fail("Duplicate setup x q x contrast rows found.")

# Convert to percentage points if contrasts are stored as fractions.
max_abs = float(df[mean_col].abs().max())
if max_abs <= 1.0 + 1e-12:
    df["_plot_mean_pp"] = 100.0 * df[mean_col]
    df["_plot_sd_pp"] = 100.0 * df[sd_col]
    scale_note = "Converted stored fractional differences to percentage points."
else:
    df["_plot_mean_pp"] = df[mean_col]
    df["_plot_sd_pp"] = df[sd_col]
    scale_note = "Input differences already appear to be in percentage points."

lower = float((df["_plot_mean_pp"] - df["_plot_sd_pp"]).min())
upper = float((df["_plot_mean_pp"] + df["_plot_sd_pp"]).max())
span = max(upper - lower, 1.0)
pad = 0.10 * span

ymin = min(0.0, lower - pad)
ymax = max(0.0, upper + pad)

step = 5.0
ymin = step * math.floor(ymin / step)
ymax = step * math.ceil(ymax / step)

FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.6),
    sharex=True,
    sharey=True,
)

markers = ["o", "s", "^"]
linestyles = ["-", "--", "-."]

for ax, setup in zip(axes, SETUP_ORDER):
    part = df[df[setup_col] == setup].copy()

    for contrast, marker, linestyle in zip(CONTRAST_ORDER, markers, linestyles):
        g = part[part[contrast_col] == contrast].set_index(q_col).reindex(Q_ORDER)

        if g["_plot_mean_pp"].isna().any() or g["_plot_sd_pp"].isna().any():
            fail(f"Incomplete q coverage for {setup} / {contrast}")

        ax.errorbar(
            Q_ORDER,
            g["_plot_mean_pp"].to_numpy(),
            yerr=g["_plot_sd_pp"].to_numpy(),
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=5.5,
            capsize=3.0,
            label=DISPLAY[contrast],
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle=":",
        color="0.35",
        zorder=0,
    )

    ax.set_title(SETUP_TITLE[setup])
    ax.set_xlabel(r"$q$")
    ax.set_xticks(Q_ORDER)
    ax.set_xticklabels(["0.20", "0.50", "0.80", "0.95"])
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.18)

axes[0].set_ylabel("AC Macro PR-AUC difference (percentage points)")
axes[1].tick_params(
    axis="y",
    labelleft=True,
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
)

fig.subplots_adjust(
    left=0.09,
    right=0.99,
    top=0.90,
    bottom=0.22,
    wspace=0.08,
)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("=" * 90)
print("FIGURE 5 - COMPONENT CONTRAST PLOT")
print("=" * 90)
print(f"Input : {INPUT}")
print(f"Rows  : {len(df)} / {expected_rows}")
print(scale_note)
print(f"Y-lim : [{ymin:.1f}, {ymax:.1f}] pp")
print(f"PDF   : {OUT_PDF}")
print(f"PNG   : {OUT_PNG}")
print()
print("Plotted contrasts:")
for contrast in CONTRAST_ORDER:
    print(f"  - {DISPLAY[contrast]}")
print()
print("No numeric point labels or highlight annotations were added.")
