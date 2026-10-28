from fedptha.paths import generated_root, figures_root

#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(str(generated_root()))
LEDGER = generated_root() / "tables"
OUT_DIR = ROOT / str(figures_root())
AUDIT_DIR = generated_root() / "plot_data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Public display names
# Historical IDs remain unchanged in all frozen source files.
# ============================================================

from fedptha.reporting.names import METHOD_NAMES as DISPLAY

# ============================================================
# Shared visual identity
# Keep method identity consistent across Figures 3, 4, and 6.
# ============================================================

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


def fail(msg):
    raise RuntimeError(msg)


def check_unique_grid(df, group_cols, expected_n, label):
    if len(df) != expected_n:
        fail(f"{label}: expected {expected_n} rows, found {len(df)}")

    if df.duplicated(group_cols).any():
        bad = df.loc[df.duplicated(group_cols, keep=False), group_cols]
        fail(f"{label}: duplicate grid rows:\n{bad.to_string(index=False)}")


# ============================================================
# FIGURE 3
#
# Selected methods:
# Global-Centralized, FedAvg, FedProx, FedALA, APPLE, FedPTHA.
# Both settings, all four q.
# ============================================================

fig3_source = LEDGER / "main_ac_summary.csv"
fig3 = pd.read_csv(fig3_source)
fig3["q"] = pd.to_numeric(fig3["q"], errors="raise").round(2)

fig3_methods = (
    "centralized",
    "fedavg",
    "fedprox",
    "fedala",
    "apple",
    "support_simplex",
)

fig3_plot = fig3[fig3["method"].isin(fig3_methods)].copy()

check_unique_grid(
    fig3_plot,
    ["setup", "method", "q"],
    2 * 6 * 4,
    "Figure 3 selected data",
)

if set(fig3_plot["method"]) != set(fig3_methods):
    fail("Figure 3 selected method registry mismatch.")

if set(fig3_plot["setup"]) != {"five_client", "twenty_client"}:
    fail("Figure 3 setup registry mismatch.")

if set(fig3_plot["q"]) != {0.20, 0.50, 0.80, 0.95}:
    fail("Figure 3 q registry mismatch.")

fig3_plot = fig3_plot.sort_values(["setup", "method", "q"]).reset_index(drop=True)

fig3_plot.to_csv(
    AUDIT_DIR / "figure3_main_plotdata.csv",
    index=False,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.25),
    sharey=True,
)

for ax, setup, title in zip(
    axes,
    ("five_client", "twenty_client"),
    ("(a) Five-client setting", "(b) Twenty-client setting"),
):
    subset = fig3_plot[fig3_plot["setup"] == setup]

    for method in fig3_methods:
        g = subset[subset["method"] == method].sort_values("q")
        style = STYLE[method]

        x = g["q"].to_numpy()
        y = g["macro_pr_auc_mean_percent"].to_numpy()
        yerr = g["macro_pr_auc_sample_sd_percent"].to_numpy()

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

    ax.set_title(title)
    ax.set_xticks([0.20, 0.50, 0.80, 0.95])
    ax.set_xticklabels(["0.20", "0.50", "0.80", "0.95"])
    ax.set_xlabel("$q$")
    ax.set_ylim(20, 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)

axes[0].set_ylabel("AC Macro PR-AUC (%)")

handles, labels = axes[0].get_legend_handles_labels()
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
    left=0.08,
    right=0.99,
    top=0.92,
    bottom=0.24,
    wspace=0.12,
)

fig.savefig(
    OUT_DIR / "figure3_main.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "figure3_main.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# FIGURE 4
#
# FedAvg, FedProx, APPLE, FedPTHA.
# VV / VO / OO, q=.95, both settings.
# ============================================================

fig4_source = LEDGER / "protocol_comparison.csv"
fig4 = pd.read_csv(fig4_source)
fig4["q"] = pd.to_numeric(fig4["q"], errors="raise").round(2)

fig4_methods = (
    "fedavg",
    "fedprox",
    "apple",
    "support_simplex",
)

fig4_plot = fig4[(fig4["method"].isin(fig4_methods)) & (fig4["q"] == 0.95)].copy()

check_unique_grid(
    fig4_plot,
    ["setup", "method", "q"],
    2 * 4,
    "Figure 4 selected data",
)

if set(fig4_plot["method"]) != set(fig4_methods):
    fail("Figure 4 selected method registry mismatch.")

fig4_plot = fig4_plot.sort_values(["setup", "method"]).reset_index(drop=True)

fig4_plot.to_csv(
    AUDIT_DIR / "figure4_protocols_plotdata.csv",
    index=False,
)

protocols = ("AA", "AC", "CC")
protocol_x = [0, 1, 2]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.7, 4.25),
    sharey=True,
)

for ax, setup, title in zip(
    axes,
    ("five_client", "twenty_client"),
    ("(a) Five-client setting", "(b) Twenty-client setting"),
):
    subset = fig4_plot[fig4_plot["setup"] == setup]

    for method in fig4_methods:
        r = subset[subset["method"] == method].iloc[0]
        style = STYLE[method]

        y = [
            100.0 * float(r["VV_mean"]),
            100.0 * float(r["VO_mean"]),
            100.0 * float(r["OO_mean"]),
        ]

        yerr = [
            100.0 * float(r["VV_sample_sd"]),
            100.0 * float(r["VO_sample_sd"]),
            100.0 * float(r["OO_sample_sd"]),
        ]

        ax.errorbar(
            protocol_x,
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

    ax.set_title(title)
    ax.set_xticks(protocol_x)
    ax.set_xticklabels(("AA", "AC", "CC"))
    ax.set_xlabel("Protocol ($q=0.95$)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)

axes[0].set_ylabel("Macro PR-AUC (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.005),
    ncol=4,
    frameon=False,
    columnspacing=1.4,
    handletextpad=0.55,
)

fig.subplots_adjust(
    left=0.08,
    right=0.99,
    top=0.92,
    bottom=0.23,
    wspace=0.12,
)

fig.savefig(
    OUT_DIR / "figure4_protocols.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "figure4_protocols.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# FIGURE 6
#
# Selected communication-performance comparison.
# FedAvg, FedProx, APPLE, FedPTHA.
# q=.20 and .95 only.
#
# IMPORTANT:
# No nondominance classification is used in this figure.
# ============================================================

fig6_source = LEDGER / "communication_performance.csv"
fig6 = pd.read_csv(fig6_source)
fig6["q"] = pd.to_numeric(fig6["q"], errors="raise").round(2)

fig6_methods = (
    "fedavg",
    "fedprox",
    "apple",
    "support_simplex",
)

fig6_q = (0.20, 0.95)

fig6_emphasis = {
    ("five_client", 0.20): {"fedprox", "support_simplex"},
    ("five_client", 0.95): {"fedprox", "support_simplex"},
    ("twenty_client", 0.20): {"fedavg"},
    ("twenty_client", 0.95): {"fedavg", "support_simplex"},
}

fig6_plot = fig6[
    (fig6["method_id"].isin(fig6_methods)) & (fig6["q"].isin(fig6_q))
].copy()

check_unique_grid(
    fig6_plot,
    ["setup", "method_id", "q"],
    2 * 4 * 2,
    "Figure 6 selected data",
)

if set(fig6_plot["method_id"]) != set(fig6_methods):
    fail("Figure 6 selected method registry mismatch.")

if set(fig6_plot["q"]) != {0.20, 0.95}:
    fail("Figure 6 selected q registry mismatch.")

fig6_plot = fig6_plot.sort_values(["setup", "q", "method_id"]).reset_index(drop=True)

fig6_plot.to_csv(
    AUDIT_DIR / "figure6_communication_plotdata.csv",
    index=False,
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.9, 6.25),
    sharey=True,
)

setups = ("five_client", "twenty_client")

for row_idx, setup in enumerate(setups):
    for col_idx, q in enumerate(fig6_q):
        ax = axes[row_idx, col_idx]

        g = fig6_plot[(fig6_plot["setup"] == setup) & (fig6_plot["q"] == q)]

        ax.set_xscale("log")
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.5,
        )

        for method in fig6_methods:
            r = g[g["method_id"] == method].iloc[0]
            style = STYLE[method]
            emphasized = method in fig6_emphasis[(setup, q)]

            if emphasized:
                point_color = style["color"]
                face_color = style["color"]
                edge_color = style["color"]
                error_color = style["color"]
                alpha = 1.0
                markersize = 8.0 if method == "support_simplex" else 7.0
                edgewidth = 1.0
                elinewidth = 1.05
                zorder = 4
            else:
                point_color = "#aaaaaa"
                face_color = "#b8b8b8"
                edge_color = "#aaaaaa"
                error_color = "#b0b0b0"
                alpha = 0.50
                markersize = 7.0
                edgewidth = 0.75
                elinewidth = 0.85
                zorder = 2

            ax.errorbar(
                float(r["total_algorithmic_GiB"]),
                float(r["vo_macro_pr_auc_mean_percent"]),
                yerr=float(r["vo_macro_pr_auc_sample_sd_percent"]),
                fmt=style["marker"],
                markersize=markersize,
                color=point_color,
                markerfacecolor=face_color,
                markeredgecolor=edge_color,
                markeredgewidth=edgewidth,
                ecolor=error_color,
                elinewidth=elinewidth,
                capsize=3.0,
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

fig.text(
    0.54,
    0.985,
    "Five-client setting",
    ha="center",
    va="top",
    fontsize=10.8,
    fontweight="bold",
)

fig.text(
    0.54,
    0.555,
    "Twenty-client setting",
    ha="center",
    va="center",
    fontsize=10.8,
    fontweight="bold",
)

fig.supxlabel(
    "Total model communication over 80 rounds (GiB; log scale)",
    x=0.54,
    y=0.105,
    fontsize=10.5,
)

legend_handles = []

for method in fig6_methods:
    style = STYLE[method]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="None",
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            color=style["color"],
            markersize=7.5,
            label=DISPLAY[method],
        )
    )

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.54, 0.005),
    ncol=4,
    frameon=False,
    columnspacing=1.4,
    handletextpad=0.5,
)

fig.subplots_adjust(
    left=0.09,
    right=0.99,
    top=0.92,
    bottom=0.19,
    wspace=0.15,
    hspace=0.33,
)

fig.savefig(
    OUT_DIR / "figure6_communication.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "figure6_communication.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# Summary
# ============================================================

print("=" * 78)
print("SUPERVISOR-SELECTED MAIN FIGURES CREATED")
print("=" * 78)

print()
print("Figure 3:")
print("  methods : Global-Centralized, FedAvg, FedProx, FedALA, APPLE, FedPTHA")
print("  q       : 0.20, 0.50, 0.80, 0.95")
print("  setups  : five-client, twenty-client")

print()
print("Figure 4:")
print("  methods : FedAvg, FedProx, APPLE, FedPTHA")
print("  protocol: VV, VO, OO at q=0.95")
print("  setups  : five-client, twenty-client")

print()
print("Figure 6:")
print("  methods : FedAvg, FedProx, APPLE, FedPTHA")
print("  q       : 0.20, 0.95")
print("  setups  : five-client, twenty-client")
print("  dominance highlighting: panel-wise; dominated methods shown in grey")

print()
print("Frozen source files were read only.")
print("No experiment or numerical result was modified.")

print()
print("Created:")
for p in (
    OUT_DIR / "figure3_main.pdf",
    OUT_DIR / "figure3_main.png",
    OUT_DIR / "figure4_protocols.pdf",
    OUT_DIR / "figure4_protocols.png",
    OUT_DIR / "figure6_communication.pdf",
    OUT_DIR / "figure6_communication.png",
):
    print(" ", p)

print()
print("Audit plot-data tables:")
for p in (
    AUDIT_DIR / "figure3_main_plotdata.csv",
    AUDIT_DIR / "figure4_protocols_plotdata.csv",
    AUDIT_DIR / "figure6_communication_plotdata.csv",
):
    print(" ", p)

print("=" * 78)
