#!/bin/bash
#SBATCH --job-name=support_only_postala
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

if [[ "${BENCHMARK_SETUP:-}" != "five_client" && "${BENCHMARK_SETUP:-}" != "twenty_client" ]]; then
  echo "ERROR: set BENCHMARK_SETUP=five_client or BENCHMARK_SETUP=twenty_client." >&2
  exit 2
fi
export BENCHMARK_SETUP

# Final PR-AUC protocol safety.
export SAVE_CHECKPOINTS=0
export RUN_TAG="${PRAUC_RUN_TAG:-support_only_postala_prauc_20260825}"
FINAL_RUNS_BASE="${FINAL_RUNS_BASE:-andrea/final_runs_prauc_20260824}"
export RUNS_ROOT="$FINAL_RUNS_BASE/$BENCHMARK_SETUP/support_only_postala"

# This ablation has no learned routing logits.
export APPLE_DR_LR=0.0

echo "BENCHMARK_SETUP=$BENCHMARK_SETUP"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "RUN_TAG=$RUN_TAG"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
echo "METHOD=fixed visible-support task simplex -> PostALA; no learned task routing"

srun apptainer exec --nv "$SIF" \
  python -u -m andrea.run_multihead.run_apple_support_only_post_ala_multi_select
