#!/bin/bash
#SBATCH --job-name=spec_train_bcast
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

if [[ "${BENCHMARK_SETUP:-twenty_client}" != "twenty_client" ]]; then
  echo "ERROR: SpecSelectTrainBcast is twenty_client-only." >&2
  exit 2
fi

export BENCHMARK_SETUP=twenty_client

export PROJ=/tudelft.net/staff-umbrella/CSE3000GLTD/ali
export SIF=$PROJ/pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif
export APPTAINER_CACHEDIR=$PROJ/cache/apptainer

cd /home/nfs/ali7/Subgraph_Federated_Learning
export PYTHONPATH=$PWD

mkdir -p logs/slurm

# Final PR-AUC protocol safety.
export SAVE_CHECKPOINTS=0
export RUN_TAG="${PRAUC_RUN_TAG:-prauc_20260824}"
FINAL_RUNS_BASE="${FINAL_RUNS_BASE:-andrea/final_runs_prauc_20260824}"
export RUNS_ROOT="$FINAL_RUNS_BASE/twenty_client/specialist_rep_transfer"

echo "PR-AUC final diagnostic: BENCHMARK_SETUP=$BENCHMARK_SETUP RUNS_ROOT=$RUNS_ROOT RUN_TAG=$RUN_TAG SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"


srun apptainer exec --nv "$SIF"   python -u -m andrea.run_multihead.run_spec_select_train_bcast_multi_select
