#!/bin/bash
#SBATCH --job-name=fedala_head_ms
#SBATCH --account=ewi-st-dis
#SBATCH --partition=ewi-st
#SBATCH --qos=medium
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --output=/home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm/%j.out
#SBATCH --error=/home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm/%j.err

set -euo pipefail

export PROJ=/tudelft.net/staff-umbrella/CSE3000GLTD/ali
export SIF=$PROJ/pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif
export APPTAINER_CACHEDIR=$PROJ/cache/apptainer

cd ~/Subgraph_Federated_Learning
export PYTHONPATH=$PWD
mkdir -p logs/slurm

# Default: do not write .pt checkpoint files. CSV files contain selected-checkpoint results.
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}

srun apptainer exec --nv "$SIF" python -u -m andrea.run_multihead.ablation_studies.run_fedala_head_only_multi_select
