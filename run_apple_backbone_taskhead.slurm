#!/bin/sh
#SBATCH --job-name=apple_backbone_apple_taskhead
#SBATCH --account=ewi-st-dis
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=3G
#SBATCH --output=/home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm/%j.out
#SBATCH --error=/home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm/%j.err

mkdir -p /home/nfs/ali7/Subgraph_Federated_Learning/logs/slurm
mkdir -p /home/nfs/ali7/Subgraph_Federated_Learning/andrea

export PROJ=/tudelft.net/staff-umbrella/CSE3000GLTD/ali
export SIF=$PROJ/pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif
export APPTAINER_CACHEDIR=$PROJ/cache/apptainer

cd /home/nfs/ali7/Subgraph_Federated_Learning
export PYTHONPATH=$PWD

echo "Running on $(hostname)"
nvidia-smi
ls -l "$SIF"

srun apptainer exec --nv "$SIF" python -u -m andrea.run_multihead.run_apple_backbone_task_head