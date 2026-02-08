import os
import csv

import pandas as pd
import torch

from utils.seed import set_seed, derive_seed

from andrea.multigraph_generation import (
    DATA_ROOT, 
    GRAPH_PARAM_CSV, 
    TASK_FUNCS, 
    TASKS, 
    set_y_and_get_motifs, 
)

from andrea.graph_partition import (
    dataset_id_from_row,
    load_global_data,
    sanity_check,
)

from andrea.components_split_utils import (
    components_original_split,
    components_label_imbalance_split,
)

BASE_SEED = 0
PARTITION_PARAM_CSV = "./andrea/graph_partition_parameters.csv"
SUBGRAPH_ROOT = "./andrea/subgraph_data"

sort_cols = ["dataset_id", "split", "method", "n_clients", "split_id"]
graph_partition_df =  pd.read_csv(PARTITION_PARAM_CSV).sort_values(sort_cols).reset_index(drop=True)

for _, row in graph_partition_df.iterrows()[0:10]:
    print(row)
