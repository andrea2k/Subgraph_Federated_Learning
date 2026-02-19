import pandas as pd
import os
from utils.train_utils import load_datasets, ensure_node_features
from utils.graph_helpers import max_port_cols, check_and_strip_self_loops
import json
import torch

BASE_SEED = 0
GRAPH_PARAM_CSV = "./andrea/test_generation_parameters.csv"
DATA_ROOT = "./andrea/test_data"


CONFIG_PATH = "./configs/pna_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["reverse_mp_with_port_and_ego"]

MODEL_NAME = CONFIG["model_name"]
BEST_MODEL_PATH = CONFIG["best_model_path"]

BASE_SEED = CONFIG["base_seed"]

USE_EGO_IDS = CONFIG["use_ego_ids"]
USE_PORT_IDS = CONFIG["use_port_ids"]
USE_MINI_BATCH = CONFIG["use_mini_batch"]
BATCH_SIZE = CONFIG["batch_size"]
PORT_EMB_DIM = CONFIG["port_emb_dim"]
NUM_EPOCHS = CONFIG["num_epochs"]

DEFAULT_HPARAMS = CONFIG["default_hparams"]

graph_data_df =  pd.read_csv(GRAPH_PARAM_CSV).sort_values(["n", "d", "r", "type"]).reset_index(drop=True)

def base_params():

    return dict(
        num_layers=DEFAULT_HPARAMS["num_layers"],
        neighbors_per_hop=DEFAULT_HPARAMS["neighbors_per_hop"],
        minority_class_weight=DEFAULT_HPARAMS["minority_class_weight"],  
        use_ego_ids=USE_EGO_IDS,
        use_mini_batch=USE_MINI_BATCH,
        batch_size=BATCH_SIZE,
        use_port_ids=USE_PORT_IDS,
        port_emb_dim=PORT_EMB_DIM,
        num_epochs=NUM_EPOCHS,
        hidden_dim=DEFAULT_HPARAMS["hidden_dim"],
        dropout=DEFAULT_HPARAMS["dropout"],
        lr=DEFAULT_HPARAMS["lr"],
        weight_decay=DEFAULT_HPARAMS["weight_decay"],
    )

def dataset_id_from_row(row):
    return f"data_{int(row['n'])}_{int(row['d'])}_{row['r']}_{row['type']}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_hparams = base_params()
    seeds = [BASE_SEED, BASE_SEED+1, BASE_SEED+2]
    

    for _, grow in graph_data_df.head(1).iterrows():
        dataset_id = dataset_id_from_row(grow)
        data_path = os.path.join(DATA_ROOT, dataset_id)
        train_data, val_data, test_data = load_datasets(data_path)
    print("43", train_data)
    train_data = check_and_strip_self_loops(train_data, "train")
    val_data   = check_and_strip_self_loops(val_data, "val")
    test_data  = check_and_strip_self_loops(test_data, "test")

    if use_port_ids:
        tr_in_max, tr_out_max = max_port_cols(train_data)
        va_in_max, va_out_max = max_port_cols(val_data)
        te_in_max, te_out_max = max_port_cols(test_data)
        in_port_vocab_size  = max(tr_in_max,  va_in_max,  te_in_max)  + 1
        out_port_vocab_size = max(tr_out_max, va_out_max, te_out_max) + 1
    else:
        in_port_vocab_size  = 0
        out_port_vocab_size = 0
if __name__ == "__main__":
    main()