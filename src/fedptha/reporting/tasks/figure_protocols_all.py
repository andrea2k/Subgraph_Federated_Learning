from __future__ import annotations

from fedptha.paths import generated_root, figures_root

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(str(generated_root()))
INPUT = ROOT / "tables" / "protocol_comparison.csv"
OUT_DIR = ROOT / str(figures_root())
PDF_OUT = OUT_DIR / "appendix_protocols.pdf"
PNG_OUT = OUT_DIR / "appendix_protocols.png"

Q_TARGET = 0.95
PROTOCOLS = ("VV", "VO", "OO")
SETUPS = ("five_client", "twenty_client")

METHODS = (
    "fully_local",
    "centralized",
    "fedavg",
    "fedprox",
    "gcflplus",
    "fedala",
    "apple",
    "cwfedavg_task",
    "support_simplex",
    "support_simplex_postala",
)

from fedptha.reporting.names import METHOD_NAMES as DISPLAY

PANEL_TITLE = {
    "five_client": "(a) Five-client setting",
    "twenty_client": "(b) Twenty-client setting",
}

tab10 = plt.get_cmap("tab10").colors

# Keep the same visual identity used in Figure 3.
STYLE = {
    "fully_local": {
        "color": "0.62",
        "marker": "o",
        "linestyle": ":",
        "linewidth": 1.25,
        "markersize": 4.8,
        "alpha": 0.90,
        "zorder": 2,
    },
    "centralized": {
        "color": "0.45",
        "marker": None,
        "linestyle": "--",
        "linewidth": 1.35,
        "markersize": 0.0,
        "alpha": 0.95,
        "zorder": 1,
    },
    "fedavg": {
        "color": tab10[0],
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.50,
        "markersize": 5.0,
        "alpha": 0.95,
        "zorder": 3,
    },
    "fedprox": {
        "color": tab10[1],
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.50,
        "markersize": 5.2,
        "alpha": 0.95,
        "zorder": 3,
    },
    "gcflplus": {
        "color": tab10[2],
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.50,
        "markersize": 4.8,
        "alpha": 0.95,
        "zorder": 3,
    },
    "fedala": {
        "color": tab10[4],
        "marker": "v",
        "linestyle": "-",
        "linewidth": 1.50,
        "markersize": 5.2,
        "alpha": 0.95,
        "zorder": 3,
    },
    "apple": {
        "color": tab10[5],
        "marker": "P",
        "linestyle": "-",
        "linewidth": 1.50,
        "markersize": 5.0,
        "alpha": 0.95,
        "zorder": 3,
    },
    "cwfedavg_task": {
        "color": tab10[6],
        "marker": "X",
        "linestyle": "-",
        "linewidth": 1.50,
        "markersize": 5.2,
        "alpha": 0.95,
        "zorder": 3,
    },
    "support_simplex": {
        "color": tab10[3],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.00,
        "markersize": 5.8,
        "alpha": 1.00,
        "zorder": 5,
    },
    "support_simplex_postala": {
        "color": tab10[9],
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.00,
        "markersize": 8.0,
        "alpha": 1.00,
        "zorder": 5,
    },
}


def abort(message: str) -> None:
    raise SystemExit(f"\nABORT: {message}\n")


print("=" * 88)
print("FIGURE 4 - VV / VO / OO AT q=0.95")
print("=" * 88)
print()
print("Input:")
print(f"  {INPUT}")

if not INPUT.is_file():
    abort(f"missing Figure-4 analysis table: {INPUT}")

df = pd.read_csv(INPUT)
df["q"] = pd.to_numeric(df["q"], errors="raise").round(2)

required = {
    "setup",
    "method",
    "method_display",
    "q",
}
for protocol in PROTOCOLS:
    required |= {
        f"{protocol}_mean",
        f"{protocol}_sample_sd",
    }

missing = sorted(required - set(df.columns))
if missing:
    abort("missing required columns: " + ", ".join(missing))

if len(df) != 80:
    abort(f"expected 80 analysis rows, found {len(df)}")

if set(df["setup"]) != set(SETUPS):
    abort(f"setup registry mismatch: {sorted(df['setup'].unique())}")

if set(df["method"]) != set(METHODS):
    abort(f"method registry mismatch: {sorted(df['method'].unique())}")

if set(df["q"]) != {0.20, 0.50, 0.80, 0.95}:
    abort(f"q grid mismatch: {sorted(df['q'].unique())}")

if "task_weighted_fedavg" in set(df["method"]):
    abort("forbidden task_weighted_fedavg detected")

if df.duplicated(["setup", "method", "q"]).any():
    abort("duplicate setup/method/q rows detected")

for protocol in PROTOCOLS:
    means = pd.to_numeric(df[f"{protocol}_mean"], errors="coerce")
    sds = pd.to_numeric(df[f"{protocol}_sample_sd"], errors="coerce")

    if not np.isfinite(means).all():
        abort(f"{protocol} means contain non-finite values")
    if not np.isfinite(sds).all():
        abort(f"{protocol} sample SDs contain non-finite values")
    if ((means < 0) | (means > 1)).any():
        abort(f"{protocol} means outside [0,1]")
    if (sds < 0).any():
        abort(f"{protocol} sample SDs contain negative values")

q95 = df[df["q"] == Q_TARGET].copy()

if len(q95) != 20:
    abort(f"expected 20 q=.95 rows, found {len(q95)}")

for setup in SETUPS:
    s = q95[q95["setup"] == setup]
    if len(s) != 10:
        abort(f"{setup}: expected 10 q=.95 methods, found {len(s)}")
    if set(s["method"]) != set(METHODS):
        abort(f"{setup}: q=.95 method registry mismatch")

# Centralized is a q-independent oracle/reference. Its three protocol
# summaries should coincide to plotting precision. We do not alter them.
for setup in SETUPS:
    row = q95[(q95["setup"] == setup) & (q95["method"] == "centralized")].iloc[0]
    values = np.array([float(row[f"{p}_mean"]) for p in PROTOCOLS])
    if not np.allclose(values, values[0], rtol=0, atol=1e-6):
        abort(
            f"{setup}: Global-Centralized differs materially across VV/VO/OO: "
            f"{values.tolist()}"
        )

print()
print("INPUT CHECK: PASS")
print(f"Rows in analysis table : {len(df)} / 80")
print(f"q=.95 plotting rows    : {len(q95)} / 20")
print("Methods                : 10 / 10")
print("Panels                 : 2")
print("Protocols              : VV, VO, OO")
print("Metric                 : Macro true trapezoidal PR-AUC")
print("Statistics             : mean +/- sample SD")
print("Task-Weighted FedAvg   : ABSENT")
print("Ablations              : ABSENT")
print("Diagnostics            : ABSENT")

# Convert fractions to percentage points only for plotting.
for protocol in PROTOCOLS:
    q95[f"{protocol}_mean_percent"] = 100.0 * q95[f"{protocol}_mean"]
    q95[f"{protocol}_sd_percent"] = 100.0 * q95[f"{protocol}_sample_sd"]

x = np.arange(len(PROTOCOLS), dtype=float)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.2, 5.6),
    sharey=True,
)

handles_by_method = {}

for ax, setup in zip(axes, SETUPS):
    subset = q95[q95["setup"] == setup].set_index("method")

    for method in METHODS:
        row = subset.loc[method]
        style = STYLE[method]
        means = np.array(
            [row[f"{p}_mean_percent"] for p in PROTOCOLS],
            dtype=float,
        )
        sds = np.array(
            [row[f"{p}_sd_percent"] for p in PROTOCOLS],
            dtype=float,
        )

        artist = ax.errorbar(
            x,
            means,
            yerr=sds,
            label=DISPLAY[method],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            capsize=2.4 if method != "centralized" else 0.0,
            capthick=0.9,
            elinewidth=0.9,
        )

        if method not in handles_by_method:
            handles_by_method[method] = artist

    ax.set_title(
        PANEL_TITLE[setup],
        fontsize=12.5,
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(("AA", "AC", "CC"))
    ax.set_xlim(-0.16, 2.16)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlabel(
        r"Protocol ($q=0.95$)",
        fontsize=11,
    )
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.40,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        labelsize=9.5,
    )

    # Keep the numeric 0-100 scale visible on both panels.
    ax.tick_params(
        axis="y",
        labelleft=True,
    )

axes[0].set_ylabel(
    "Macro PR-AUC (%)",
    fontsize=11,
)

legend_handles = [handles_by_method[method] for method in METHODS]
legend_labels = [DISPLAY[method] for method in METHODS]

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=5,
    frameon=False,
    fontsize=9.0,
    columnspacing=1.6,
    handlelength=2.4,
    handletextpad=0.6,
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.91,
    bottom=0.245,
    wspace=0.10,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

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

print()
print("=" * 88)
print("PASS - FIGURE 4 FIRST DRAFT CREATED")
print("=" * 88)
print()
print("Plotting source:")
print(f"  {INPUT}")
print()
print("q                         : 0.95 only")
print("Panels                    : 2")
print("Protocols                 : VV, VO, OO")
print("Methods plotted           : 10")
print("Statistics                : mean +/- sample SD")
print("Shared y-axis             : 0 to 100 %")
print("Y tick labels             : shown on both panels")
print("Task-Weighted FedAvg      : ABSENT")
print("Ablations                 : ABSENT")
print("Collaboration diagnostics : ABSENT")
print()
print("Created:")
print(f"  {PDF_OUT}")
print(f"  {PNG_OUT}")
print()
print("No source result file was modified.")
print("=" * 88)
