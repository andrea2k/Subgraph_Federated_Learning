#!/bin/sh
#SBATCH --job-name=fed-thesis-26-jan
#SBATCH --account=ewi-st-dis
#SBATCH --qos=short
#SBATCH --partition=general      # Request partition
#SBATCH --time=4:00:00           # Wall-clock time
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Increase file descriptor limit (based on Cagri's version)
ulimit -n 65536

# Put Apptainer cache on /tmp so HOME quota doesn't explode
export APPTAINER_CACHEDIR=/tmp/$USER

# Paths 
APPTAINER_IMAGE="/tudelft.net/staff-umbrella/CSE3000GLTD/sceydeli/pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif"
CODE_DIR="/home/nfs/sceydeli/thesis/Subgraph_Federated_Learning"
CKPT_DIR="/tudelft.net/staff-umbrella/CSE3000GLTD/sceydeli/checkpoints"

# Make sure logs exist
mkdir -p logs/slurm

# Optional module load
# module load cuda/12.4

# Run this script inside the container
srun apptainer exec \
  --nv \
  --containall \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${CKPT_DIR}:/workspace/checkpoints" \
  "${APPTAINER_IMAGE}" \
  bash -c "cd /workspace && export PYTHONPATH=/workspace:\$PYTHONPATH && python -m scripts.training.train_federated_pna"
