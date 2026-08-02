from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
OUT = ROOT / "analysis_outputs/twenty_client_main_results"

OLD_TARGET = "best_realistic_visible_macro_pr_auc_test"

METHODS = {
    "Fully Local": "runs_twenty_client_fully_local_q{q}",
    "FedAvg": "runs_twenty_client_fedavg_q{q}",
    "FedProx": "runs_twenty_client_fedprox_q{q}",
    "GCFL+": "runs_twenty_client_gcflplus_q{q}",
    "FedALA": "runs_twenty_client_fedala_q{q}",
    "APPLE": "runs_twenty_client_apple_q{q}",
    "APPLE-ALA": "runs_twenty_client_apple_ala_q{q}",
    "APPLE-PostALA": "runs_twenty_client_apple_post_ala_q{q}",
}

Q_INFO = {
    "02": (0.2, 4000000),
    "05": (0.5, 4000020),
    "08": (0.8, 4000040),
}

METRICS = [
    "eval_loss",
    "macro_f1",
    "macro_minority_f1",
    "macro_pos_f1",
    "macro_pr_auc",
    "micro_f1",
    "micro_pr_auc",
    "pair_acc",
    "subset_acc",
]


def main_csvs(root):
    return sorted(
        path
        for path in root.rglob("*.csv")
        if not path.name.endswith("_dr.csv")
    )


def select_primary_rows(frame):
    old_columns = [
        column
        for column in frame.columns
        if frame[column].astype(str).eq(OLD_TARGET).any()
    ]

    if len(old_columns) == 1:
        column = old_columns[0]
        return frame[
            frame[column].astype(str).eq(OLD_TARGET)
        ].copy(), "legacy_target"

    if "phase" not in frame.columns:
        return frame.iloc[0:0].copy(), "no_supported_schema"

    mask = frame["phase"].astype(str).eq(
        "best_realistic_visible_test"
    )

    if "selector_metric" in frame.columns:
        mask &= frame["selector_metric"].astype(str).eq(
            "macro_pr_auc"
        )

    return frame[mask].copy(), "phase_selector"


all_rows = []
inventory_rows = []

for method, directory_pattern in METHODS.items():
    for q_code, (q_value, first_id) in Q_INFO.items():
        root = ROOT / directory_pattern.format(q=q_code)

        if not root.is_dir():
            raise FileNotFoundError(root)

        expected_ids = set(range(first_id, first_id + 20))
        collected = []

        files = main_csvs(root)

        for path in files:
            frame = pd.read_csv(
                path,
                dtype=str,
                low_memory=False,
            )

            selected, schema = select_primary_rows(frame)

            if selected.empty:
                continue

            if "graph_id" not in selected.columns:
                continue

            selected["graph_id"] = pd.to_numeric(
                selected["graph_id"],
                errors="coerce",
            )

            selected = selected.dropna(
                subset=["graph_id"]
            ).copy()

            selected["graph_id"] = (
                selected["graph_id"].astype(int)
            )

            selected = selected[
                selected["graph_id"].isin(expected_ids)
            ].copy()

            if selected.empty:
                continue

            if "seed" not in selected.columns:
                raise RuntimeError(
                    f"No seed column in {path}"
                )

            selected["seed"] = pd.to_numeric(
                selected["seed"],
                errors="coerce",
            )

            selected = selected[
                selected["seed"].isin([0, 1, 2])
            ].copy()

            if selected.empty:
                continue

            for metric in METRICS:
                if metric in selected.columns:
                    selected[metric] = pd.to_numeric(
                        selected[metric],
                        errors="coerce",
                    )

            for _, row in selected.iterrows():
                output = {
                    "method": method,
                    "q": q_value,
                    "seed": int(row["seed"]),
                    "graph_id": int(row["graph_id"]),
                    "schema": schema,
                    "source_csv": str(path),
                }

                for metric in METRICS:
                    if metric in selected.columns:
                        output[metric] = row[metric]

                collected.append(output)

        result = pd.DataFrame(collected)

        if result.empty:
            raise RuntimeError(
                f"No selected rows for {method}, q={q_value}"
            )

        duplicates = result.duplicated(
            subset=["seed", "graph_id"],
            keep=False,
        )

        if duplicates.any():
            print()
            print("DUPLICATES FOUND:")
            print(
                result.loc[
                    duplicates,
                    [
                        "seed",
                        "graph_id",
                        "schema",
                        "source_csv",
                    ],
                ]
                .sort_values(["seed", "graph_id"])
                .to_string(index=False)
            )

            raise RuntimeError(
                f"Duplicate rows for {method}, q={q_value}"
            )

        expected_pairs = {
            (seed, graph_id)
            for seed in [0, 1, 2]
            for graph_id in expected_ids
        }

        actual_pairs = set(
            zip(result["seed"], result["graph_id"])
        )

        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)

        print(
            f"{method:20s} q={q_value:.1f} "
            f"rows={len(result):3d} "
            f"seeds={sorted(result['seed'].unique())}"
        )

        if missing:
            print("Missing pairs:", missing)

        if extra:
            print("Extra pairs:", extra)

        if actual_pairs != expected_pairs:
            raise RuntimeError(
                f"Incomplete results for {method}, q={q_value}"
            )

        if "macro_pr_auc" not in result.columns:
            raise RuntimeError(
                f"macro_pr_auc missing for {method}, q={q_value}"
            )

        if result["macro_pr_auc"].isna().any():
            raise RuntimeError(
                f"NaN macro_pr_auc for {method}, q={q_value}"
            )

        all_rows.append(result)

        inventory_rows.append(
            {
                "method": method,
                "q": q_value,
                "main_csv_files": len(files),
                "selected_client_rows": len(result),
                "schemas": "|".join(
                    sorted(result["schema"].unique())
                ),
            }
        )


clients = pd.concat(
    all_rows,
    ignore_index=True,
)

expected_total = 8 * 3 * 3 * 20

print()
print("Total selected client rows:", len(clients))
print("Expected selected client rows:", expected_total)

assert len(clients) == expected_total


available_metrics = [
    metric
    for metric in METRICS
    if metric in clients.columns
    and clients[metric].notna().all()
]

seed_summary = (
    clients
    .groupby(
        ["method", "q", "seed"],
        as_index=False,
    )[available_metrics]
    .mean()
)

summary_rows = []

for (method, q_value), group in seed_summary.groupby(
    ["method", "q"]
):
    row = {
        "method": method,
        "q": q_value,
    }

    for metric in available_metrics:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)

    summary_rows.append(row)

method_summary = pd.DataFrame(summary_rows)

method_summary["rank_all"] = (
    method_summary
    .groupby("q")["macro_pr_auc_mean"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

federated_methods = {
    "FedAvg",
    "FedProx",
    "GCFL+",
    "FedALA",
    "APPLE",
    "APPLE-ALA",
    "APPLE-PostALA",
}

federated = method_summary[
    method_summary["method"].isin(federated_methods)
].copy()

federated["rank_federated"] = (
    federated
    .groupby("q")["macro_pr_auc_mean"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)


OUT.mkdir(parents=True, exist_ok=True)

pd.DataFrame(inventory_rows).to_csv(
    OUT / "inventory.csv",
    index=False,
)

clients.to_csv(
    OUT / "selected_per_client.csv",
    index=False,
)

seed_summary.to_csv(
    OUT / "seed_summary.csv",
    index=False,
)

method_summary.to_csv(
    OUT / "method_summary.csv",
    index=False,
)

federated.to_csv(
    OUT / "federated_summary.csv",
    index=False,
)


print()
print("=" * 86)
print("ALL METHODS — REALISTIC-VISIBLE MACRO PR-AUC")
print("Mean ± standard deviation over seeds")
print("=" * 86)

for q_value in [0.2, 0.5, 0.8]:
    part = (
        method_summary[
            method_summary["q"].eq(q_value)
        ]
        .sort_values(["rank_all", "method"])
        .copy()
    )

    part["result"] = part.apply(
        lambda row:
            f"{row['macro_pr_auc_mean']:.4f} "
            f"± {row['macro_pr_auc_std']:.4f}",
        axis=1,
    )

    print()
    print(f"q = {q_value:.1f}")
    print(
        part[
            ["rank_all", "method", "result"]
        ]
        .rename(columns={"rank_all": "rank"})
        .to_string(index=False)
    )


print()
print("=" * 86)
print("FEDERATED METHODS ONLY")
print("=" * 86)

for q_value in [0.2, 0.5, 0.8]:
    part = (
        federated[
            federated["q"].eq(q_value)
        ]
        .sort_values(["rank_federated", "method"])
        .copy()
    )

    part["result"] = part.apply(
        lambda row:
            f"{row['macro_pr_auc_mean']:.4f} "
            f"± {row['macro_pr_auc_std']:.4f}",
        axis=1,
    )

    print()
    print(f"q = {q_value:.1f}")
    print(
        part[
            ["rank_federated", "method", "result"]
        ]
        .rename(
            columns={"rank_federated": "rank"}
        )
        .to_string(index=False)
    )


print()
print("=" * 86)
print("SEED-LEVEL MACRO PR-AUC")
print("=" * 86)

pivot = seed_summary.pivot_table(
    index=["method", "q"],
    columns="seed",
    values="macro_pr_auc",
)

pivot.columns = [
    f"seed_{int(seed)}"
    for seed in pivot.columns
]

print(pivot.reset_index().to_string(index=False))

print()
print("Saved results to:", OUT)
