from fedptha.paths import generated_root, figures_root

#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Paths
# ============================================================

ROOT = Path(".")

INPUT = ROOT / str(generated_root() / "tables") / "main_ac_summary.csv"

OUT_DIR = ROOT / str(figures_root())

PDF_OUT = OUT_DIR / "appendix_main_comparison.pdf"
PNG_OUT = OUT_DIR / "appendix_main_comparison.png"


# ============================================================
# Final Figure-3 method registry
# ============================================================

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

SETUPS = (
    "five_client",
    "twenty_client",
)

PANEL_TITLE = {
    "five_client": "(a) Five-client setting",
    "twenty_client": "(b) Twenty-client setting",
}

Q_VALUES = (
    0.20,
    0.50,
    0.80,
    0.95,
)


# ============================================================
# Visual style
#
# All methods remain visible.
# Support-Simplex variants receive only modest emphasis.
# Global-Centralized and Fully Local are visually references.
# ============================================================

tab10 = plt.get_cmap("tab10").colors

STYLE = {
    "fully_local": {
        "color": "0.55",
        "marker": "o",
        "linestyle": ":",
        "linewidth": 1.35,
        "markersize": 4.5,
        "alpha": 0.85,
        "zorder": 2,
    },
    "centralized": {
        "color": "0.38",
        "marker": None,
        "linestyle": "--",
        "linewidth": 1.25,
        "markersize": 0,
        "alpha": 0.85,
        "zorder": 1,
    },
    "fedavg": {
        "color": tab10[0],
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 4.8,
        "alpha": 0.90,
        "zorder": 3,
    },
    "fedprox": {
        "color": tab10[1],
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 5.0,
        "alpha": 0.90,
        "zorder": 3,
    },
    "gcflplus": {
        "color": tab10[2],
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 4.6,
        "alpha": 0.90,
        "zorder": 3,
    },
    "fedala": {
        "color": tab10[4],
        "marker": "v",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 5.0,
        "alpha": 0.90,
        "zorder": 3,
    },
    "apple": {
        "color": tab10[5],
        "marker": "P",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 5.0,
        "alpha": 0.90,
        "zorder": 3,
    },
    "cwfedavg_task": {
        "color": tab10[6],
        "marker": "X",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 5.2,
        "alpha": 0.90,
        "zorder": 3,
    },
    "support_simplex": {
        "color": tab10[3],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 6.0,
        "alpha": 1.0,
        "zorder": 6,
    },
    "support_simplex_postala": {
        "color": tab10[9],
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 8.0,
        "alpha": 1.0,
        "zorder": 6,
    },
}


# ============================================================
# Read and audit input
# ============================================================

print("=" * 72)
print("FIGURE 3 - VO MAIN PERFORMANCE DRAFT")
print("=" * 72)

if not INPUT.is_file():
    raise SystemExit(f"ABORT: missing plotting input:\n  {INPUT}")

df = pd.read_csv(INPUT)

print()
print("Input:")
print(" ", INPUT)
print()
print("Rows found:", len(df))

if len(df) != 80:
    raise SystemExit(f"ABORT: expected 80 VO summary rows, found {len(df)}")

required_columns = {
    "setup",
    "method",
    "q",
    "macro_pr_auc_mean",
    "macro_pr_auc_sample_sd",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise SystemExit(
        "ABORT: missing required input columns: " + ", ".join(sorted(missing_columns))
    )

# Normalize q.
df["q"] = df["q"].astype(float).round(2)

# Explicit safety: removed method must never return.
if df["method"].astype(str).eq("task_weighted_fedavg").any():
    raise SystemExit("ABORT: task_weighted_fedavg found in Figure-3 input.")

found_methods = set(df["method"].astype(str))
expected_methods = set(METHODS)

if found_methods != expected_methods:
    raise SystemExit(
        "ABORT: Figure-3 method registry mismatch.\n"
        f"Found:    {sorted(found_methods)}\n"
        f"Expected: {sorted(expected_methods)}"
    )

found_setups = set(df["setup"].astype(str))

if found_setups != set(SETUPS):
    raise SystemExit(
        "ABORT: setup registry mismatch.\n" f"Found: {sorted(found_setups)}"
    )

# Exact expected logical grid:
# 2 setups x 10 methods x 4 q = 80.
expected_keys = {
    (
        setup,
        method,
        round(q, 2),
    )
    for setup in SETUPS
    for method in METHODS
    for q in Q_VALUES
}

actual_keys = {
    (
        str(row.setup),
        str(row.method),
        round(float(row.q), 2),
    )
    for row in df.itertuples()
}

if len(actual_keys) != 80:
    raise SystemExit(f"ABORT: unique plotting keys={len(actual_keys)}, expected 80")

if actual_keys != expected_keys:
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    raise SystemExit(
        "ABORT: Figure-3 logical grid mismatch.\n"
        f"Missing: {sorted(missing)[:20]}\n"
        f"Extra:   {sorted(extra)[:20]}"
    )

# Convert decimal PR-AUC to percentage.
df["mean_pct"] = df["macro_pr_auc_mean"].astype(float) * 100.0

df["sd_pct"] = df["macro_pr_auc_sample_sd"].astype(float) * 100.0

# Ensure all values are within a sensible PR-AUC range.
if not df["mean_pct"].between(0, 100).all():
    raise SystemExit("ABORT: Macro PR-AUC mean outside [0,100].")

if (df["sd_pct"] < 0).any():
    raise SystemExit("ABORT: negative sample SD detected.")

print("Unique methods:", len(found_methods), "/ 10")
print("Setups        :", len(found_setups), "/ 2")
print("q points      : 4 per method/setup")
print("Protocol      : VO")
print("Metric        : Macro true trapezoidal PR-AUC")
print()


# ============================================================
# Plot
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.2, 5.6),
    sharey=True,
)

for ax, setup in zip(
    axes,
    SETUPS,
):
    subset = df[df["setup"] == setup].copy()

    for method in METHODS:
        method_df = subset[subset["method"] == method].sort_values("q")

        x = method_df["q"].to_numpy()
        y = method_df["mean_pct"].to_numpy()
        yerr = method_df["sd_pct"].to_numpy()

        style = STYLE[method]

        # Global-Centralized is a q-independent paper-facing
        # oracle/reference, so visually treat it as a reference line.
        if method == "centralized":
            ax.plot(
                x,
                y,
                label=DISPLAY[method],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                zorder=style["zorder"],
            )

            # Still show the frozen sample-SD uncertainty.
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=style["color"],
                elinewidth=0.8,
                capsize=2.0,
                alpha=0.55,
                zorder=style["zorder"],
            )

        else:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                label=DISPLAY[method],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                alpha=style["alpha"],
                capsize=2.5,
                capthick=0.9,
                elinewidth=0.9,
                markeredgewidth=0.8,
                zorder=style["zorder"],
            )

    ax.set_title(
        PANEL_TITLE[setup],
        fontsize=12,
        pad=10,
    )

    ax.set_xlabel(
        r"$q$",
        fontsize=10.5,
    )

    ax.set_xticks(Q_VALUES)

    ax.set_xticklabels(
        [
            "0.20",
            "0.50",
            "0.80",
            "0.95",
        ]
    )

    ax.set_xlim(
        0.15,
        1.00,
    )

    # Frozen first-draft common scale.
    ax.set_ylim(
        20,
        100,
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.45,
    )

    ax.tick_params(
        axis="both",
        labelsize=9.5,
    )

    # Show the shared 20-100 y-axis tick labels on both panels.
    ax.tick_params(
        axis="y",
        labelleft=True,
    )

axes[0].set_ylabel(
    "AC Macro PR-AUC (%)",
    fontsize=10.5,
)

# ============================================================
# Shared legend below both panels
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

# Preserve the intended method ordering.
handle_by_label = {label: handle for handle, label in zip(handles, labels)}

ordered_labels = [DISPLAY[method] for method in METHODS]

ordered_handles = [handle_by_label[label] for label in ordered_labels]

fig.legend(
    ordered_handles,
    ordered_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=5,
    frameon=False,
    fontsize=9,
    columnspacing=1.5,
    handlelength=2.3,
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.91,
    bottom=0.23,
    wspace=0.10,
)

# ============================================================
# Save
# ============================================================

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


# ============================================================
# Report
# ============================================================

print("=" * 72)
print("PASS - FIGURE 3 FIRST DRAFT CREATED")
print("=" * 72)
print()
print("Plotting source:")
print(" ", INPUT)
print()
print("Protocol                 : VO")
print("Methods plotted          : 10")
print("q points / method/setup  : 4")
print("Panels                    : 2")
print("Statistics                : mean +/- sample SD")
print("Common y-axis             : 20 to 100 %")
print("Task-Weighted FedAvg      : ABSENT")
print("Ablations                 : ABSENT")
print("Collaboration diagnostics : ABSENT")
print()
print("Created:")
print(" ", PDF_OUT)
print(" ", PNG_OUT)
print()
print("No numerical result file was modified.")
print("=" * 72)
