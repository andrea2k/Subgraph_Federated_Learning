#!/bin/bash
#SBATCH --job-name=oracle_hier_taskavg
#SBATCH --account=ewi-st-dis
#SBATCH --partition=ewi-st
#SBATCH --qos=medium
#SBATCH --time=20:00:00
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

export BENCHMARK_SETUP=${BENCHMARK_SETUP:-twenty_client}
if [[ "$BENCHMARK_SETUP" != "twenty_client" ]]; then
  echo "ERROR: oracle hierarchical diagnostic requires BENCHMARK_SETUP=twenty_client" >&2
  exit 2
fi

# CSV-only protocol. Never persist model checkpoints.
export SAVE_CHECKPOINTS=0
export RUN_TAG="${PRAUC_RUN_TAG:-oracle_hier_taskavg_prauc_20260825}"
FINAL_RUNS_BASE="${FINAL_RUNS_BASE:-andrea/final_runs_prauc_20260824}"
export RUNS_ROOT="$FINAL_RUNS_BASE/twenty_client/oracle_hierarchical_taskfedavg_ala"

echo "BENCHMARK_SETUP=$BENCHMARK_SETUP"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "RUN_TAG=$RUN_TAG"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
echo "METHOD=intra Support-Simplex+PostALA -> one-shot inter task-weighted FedAvg -> HeadOnly ALA"

srun apptainer exec --nv "$SIF" \
  python -u -m andrea.run_multihead.run_oracle_hierarchical_taskfedavg_multi_select
