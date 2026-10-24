#!/bin/bash
#SBATCH --job-name=oracle_hard_postala
#SBATCH --account=ewi-st-dis
#SBATCH --partition=ewi-st
#SBATCH --qos=medium
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm/%j.out
#SBATCH --error=/home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm/%j.err

set -euo pipefail

export PROJ=/tudelft.net/staff-umbrella/CSE3000GLTD/ali
export SIF=$PROJ/pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif
export APPTAINER_CACHEDIR=$PROJ/cache/apptainer

cd ~/Subgraph_Federated_Learning
export PYTHONPATH=$PWD
mkdir -p logs/slurm

# Final PR-AUC protocol safety.
export SAVE_CHECKPOINTS=0
export RUN_TAG="${PRAUC_RUN_TAG:-prauc_20260824}"
FINAL_RUNS_BASE="${FINAL_RUNS_BASE:-andrea/final_runs_prauc_20260824}"
export RUNS_ROOT="$FINAL_RUNS_BASE/twenty_client/oracle_physical_community"

echo "PR-AUC final diagnostic: BENCHMARK_SETUP=$BENCHMARK_SETUP RUNS_ROOT=$RUNS_ROOT RUN_TAG=$RUN_TAG SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"

export BENCHMARK_SETUP=${BENCHMARK_SETUP:-twenty_client}
if [[ "$BENCHMARK_SETUP" != "twenty_client" ]]; then
  echo "ERROR: oracle-hard-cluster APPLE-PostALA is only valid for BENCHMARK_SETUP=twenty_client" >&2
  exit 2
fi


srun apptainer exec --nv "$SIF" python -u -m andrea.run_multihead.run_oracle_hard_cluster_apple_post_ala_multi_select
