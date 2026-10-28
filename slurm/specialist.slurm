#!/bin/bash
#SBATCH --job-name=specialist
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

set -euo pipefail
PROJECT_ROOT="${FEDPTHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$PROJECT_ROOT/slurm/run.sh" specialist "$@"
