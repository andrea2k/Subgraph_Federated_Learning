import pandas as pd

reg = pd.read_csv(
    "./andrea/clustering_task_specialized/cluster_generation_parameters.csv"
)
sel = pd.read_csv("./andrea/clustering_task_specialized/selected_subset.csv")

print("registry shape:", reg.shape)
print("selected shape:", sel.shape)

print(
    reg[
        [
            "graph_id",
            "dataset_id",
            "target_family",
            "assigned_task",
            "mask_task",
            "mask_fraction",
            "mask_mode",
            "controlled_benchmark",
        ]
    ].head(15)
)

print(
    sel[
        [
            "family",
            "subset_id",
            "mask_fraction",
            "specialization_fraction",
            "global_visible_support_fraction_ideal",
            "task_profile_jsd_mean",
            "controlled_benchmark",
            "mask_mode",
        ]
    ]
)
