from fedptha.paths import generated_root, figures_root

#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(str(generated_root()))
INPUT = ROOT / "tables/communication_frontier.csv"
OUT_DIR = ROOT / str(figures_root())
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUT = OUT_DIR / "appendix_communication.pdf"
PNG_OUT = OUT_DIR / "appendix_communication.png"
PLOTDATA_OUT = ROOT / "tables/appendix_communication_plotdata.csv"


METHODS = [
    "FedAvg",
    "FedProx",
    "GCFL+",
    "FedALA",
    "APPLE",
    "cwFedAvg-Task",
    "Support-Simplex",
    "Support-Simplex + PostALA",
]

# Keep the same tab10 positions used by the main comparison:
# Fully Local and Global-Centralized occupy the first two tab10 positions
# in Figure 3 and are omitted here.
# Reader-facing names only.
# Historical method strings remain unchanged because they are the
# identities stored in the frozen communication/dominance tables.
DISPLAY = {
    "Support-Simplex": "FedPTHA",
    "Support-Simplex + PostALA": "FedPTHA + PostALA",
}

tab10 = plt.get_cmap("tab10").colors

STYLES = {
    "FedAvg": {
        "color": tab10[0],
        "marker": "s",
    },
    "FedProx": {
        "color": tab10[1],
        "marker": "^",
    },
    "GCFL+": {
        "color": tab10[2],
        "marker": "D",
    },
    "FedALA": {
        "color": tab10[4],
        "marker": "v",
    },
    "APPLE": {
        "color": tab10[5],
        "marker": "P",
    },
    "cwFedAvg-Task": {
        "color": tab10[6],
        "marker": "X",
    },
    "Support-Simplex": {
        "color": tab10[3],
        "marker": "o",
    },
    "Support-Simplex + PostALA": {
        "color": tab10[9],
        "marker": "*",
    },
}

Q_VALUES = [0.20, 0.50, 0.80, 0.95]
SETUPS = ["five_client", "twenty_client"]

EXPECTED_ND = {
    ("five_client", 0.20): {"FedProx", "Support-Simplex"},
    ("five_client", 0.50): {"FedProx", "Support-Simplex"},
    ("five_client", 0.80): {"FedProx", "Support-Simplex + PostALA"},
    ("five_client", 0.95): {"FedProx", "Support-Simplex + PostALA"},
    ("twenty_client", 0.20): {"FedAvg"},
    ("twenty_client", 0.50): {"FedAvg"},
    ("twenty_client", 0.80): {"FedAvg", "Support-Simplex"},
    ("twenty_client", 0.95): {"FedAvg", "Support-Simplex + PostALA"},
}


def as_bool(series):
    if series.dtype == bool:
        return series
    return (
        series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )


df = pd.read_csv(INPUT)

required = {
    "setup",
    "q",
    "method",
    "total_algorithmic_GiB",
    "vo_macro_pr_auc_mean_percent",
    "vo_macro_pr_auc_sample_sd_percent",
    "pareto_nondominated",
}

missing = sorted(required - set(df.columns))
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

df = df[(df["setup"].isin(SETUPS)) & (df["method"].isin(METHODS))].copy()

if len(df) != 64:
    raise RuntimeError(
        f"Expected 64 plotting rows " f"(2 setups x 4 q x 8 methods), found {len(df)}"
    )

df["q"] = df["q"].astype(float).round(2)
df["pareto_nondominated"] = as_bool(df["pareto_nondominated"])

if df["pareto_nondominated"].isna().any():
    raise RuntimeError("Could not parse pareto_nondominated as boolean.")

for setup in SETUPS:
    for q in Q_VALUES:
        g = df[(df["setup"] == setup) & (df["q"] == round(q, 2))]

        if len(g) != 8:
            raise RuntimeError(f"{setup}, q={q}: expected 8 methods, found {len(g)}")

        methods_here = set(g["method"])
        if methods_here != set(METHODS):
            raise RuntimeError(
                f"{setup}, q={q}: method registry mismatch: " f"{sorted(methods_here)}"
            )

        actual_nd = set(g.loc[g["pareto_nondominated"], "method"])

        expected_nd = EXPECTED_ND[(setup, round(q, 2))]

        if actual_nd != expected_nd:
            raise RuntimeError(
                f"{setup}, q={q}: nondominated set mismatch.\n"
                f"expected={sorted(expected_nd)}\n"
                f"actual={sorted(actual_nd)}"
            )

if (
    df[
        [
            "total_algorithmic_GiB",
            "vo_macro_pr_auc_mean_percent",
            "vo_macro_pr_auc_sample_sd_percent",
        ]
    ]
    .isna()
    .any()
    .any()
):
    raise RuntimeError("NaN found in plotting quantities.")

# Freeze exact plotting table.
plotdata = df.sort_values(
    ["setup", "q", "total_algorithmic_GiB", "method"]
).reset_index(drop=True)

plotdata.to_csv(PLOTDATA_OUT, index=False)


# ============================================================
# Figure
# ============================================================

plt.rcParams.update(
    {
        "font.size": 10.0,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12.8, 6.1),
    sharey=True,
)

for row_idx, setup in enumerate(SETUPS):
    for col_idx, q in enumerate(Q_VALUES):
        ax = axes[row_idx, col_idx]

        g = plotdata[
            (plotdata["setup"] == setup) & (plotdata["q"] == round(q, 2))
        ].copy()

        ax.set_xscale("log")
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.5,
        )

        for method in METHODS:
            r = g[g["method"] == method].iloc[0]

            x = float(r["total_algorithmic_GiB"])
            y = float(r["vo_macro_pr_auc_mean_percent"])
            yerr = float(r["vo_macro_pr_auc_sample_sd_percent"])
            nd = bool(r["pareto_nondominated"])

            style = STYLES[method]

            # Dominated methods provide context without competing visually
            # with the mean-based nondominated points.
            if nd:
                point_color = style["color"]
                face_color = style["color"]
                edge_color = style["color"]
                error_color = style["color"]
                alpha = 1.0
                markersize = 9.5
                edgewidth = 1.0
                elinewidth = 1.15
                zorder = 4
            else:
                point_color = "#aaaaaa"
                face_color = "#b8b8b8"
                edge_color = "#aaaaaa"
                error_color = "#b0b0b0"
                alpha = 0.50
                markersize = 7.2
                edgewidth = 0.75
                elinewidth = 0.85
                zorder = 2

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=style["marker"],
                markersize=markersize,
                color=point_color,
                markerfacecolor=face_color,
                markeredgecolor=edge_color,
                markeredgewidth=edgewidth,
                ecolor=error_color,
                elinewidth=elinewidth,
                capsize=2.5,
                alpha=alpha,
                zorder=zorder,
            )

        ax.set_title(rf"$q={q:.2f}$")
        ax.set_ylim(25, 95)

        if row_idx == 0:
            ax.set_xlim(3.7, 10.8)
            ax.set_xticks([4, 6, 10])
            ax.set_xticklabels(["4", "6", "10"])
        else:
            ax.set_xlim(13.5, 180)
            ax.set_xticks([16, 32, 64, 128])
            ax.set_xticklabels(["16", "32", "64", "128"])

        if col_idx == 0:
            ax.set_ylabel("AC Macro PR-AUC (%)")


# Centered row headers keep the q-titles and plotting area unobstructed.
fig.text(
    0.53,
    0.985,
    "Five-client setting",
    ha="center",
    va="top",
    fontsize=10.5,
    fontweight="bold",
)

fig.text(
    0.53,
    0.565,
    "Twenty-client setting",
    ha="center",
    va="center",
    fontsize=10.5,
    fontweight="bold",
)

fig.supxlabel(
    "Total model communication over 80 rounds (GiB; log scale)",
    x=0.53,
    y=0.105,
    fontsize=10.5,
)


method_handles = []

for method in METHODS:
    style = STYLES[method]

    method_handles.append(
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="None",
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            color=style["color"],
            markersize=7.5,
            label=DISPLAY.get(method, method),
        )
    )


fig.legend(
    handles=method_handles,
    loc="lower center",
    bbox_to_anchor=(0.53, 0.005),
    ncol=4,
    frameon=False,
    columnspacing=1.15,
    handletextpad=0.45,
)

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    top=0.925,
    bottom=0.205,
    wspace=0.18,
    hspace=0.30,
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("=" * 88)
print("FIGURE 6 BUILT")
print("=" * 88)
print("Population : 8 main federated collaborative methods")
print("Performance: frozen Figure-3 AC Macro PR-AUC mean +/- sample SD")
print("Communication: frozen total_algorithmic_GiB")
print("Highlighting: non-emphasized methods shown in grey")
print("Reference methods excluded: Fully Local, Global-Centralized")
print("Controlled/oracle diagnostics excluded from main comparison")
print()

for setup in SETUPS:
    print(setup)
    for q in Q_VALUES:
        g = plotdata[
            (plotdata["setup"] == setup)
            & (plotdata["q"] == round(q, 2))
            & (plotdata["pareto_nondominated"])
        ]

        labels = [
            (
                f"{r.method}: "
                f"{r.total_algorithmic_GiB:.6f} GiB, "
                f"{r.vo_macro_pr_auc_mean_percent:.3f} "
                f"+/- {r.vo_macro_pr_auc_sample_sd_percent:.3f}%"
            )
            for r in g.itertuples()
        ]

        print(f"  q={q:.2f}: " + " ; ".join(labels))

print()
print("Wrote:")
print(" ", PDF_OUT)
print(" ", PNG_OUT)
print(" ", PLOTDATA_OUT)
