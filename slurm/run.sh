#!/bin/bash
# Shared container invocation. Submit the method .slurm files from the project root.
set -euo pipefail
METHOD="${1:?method is required}"
shift
PROJECT_ROOT="${FEDPTHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
if [[ ! -f pyproject.toml ]]; then
  echo "Set FEDPTHA_PROJECT_ROOT to the repository checkout." >&2
  exit 2
fi
if [[ "${BENCHMARK_SETUP:-}" != "five_client" && "${BENCHMARK_SETUP:-}" != "twenty_client" ]]; then
  echo "Set BENCHMARK_SETUP=five_client or BENCHMARK_SETUP=twenty_client." >&2
  exit 2
fi
CONTAINER_PATH="${FEDPTHA_CONTAINER:?Set FEDPTHA_CONTAINER to your PyTorch/PyG Apptainer image.}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PROJECT_ROOT/results/.cache/apptainer}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p logs/slurm
ARGS=(train --method "$METHOD" --benchmark "$BENCHMARK_SETUP" --output-dir "${FEDPTHA_RUNS_ROOT:-results/runs/${RUN_TAG:-thesis}}")
if [[ -n "${TRAIN_SEED:-}" ]]; then ARGS+=(--seeds "$TRAIN_SEED"); fi
if [[ -n "${TRAIN_Q:-}" ]]; then ARGS+=(--q "$TRAIN_Q"); fi
if [[ -n "${TRAIN_ROUNDS:-}" ]]; then ARGS+=(--rounds "$TRAIN_ROUNDS"); fi
srun apptainer exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT" --pwd "$PROJECT_ROOT" --env "PYTHONPATH=$PROJECT_ROOT/src" "$CONTAINER_PATH" python -u -m fedptha "${ARGS[@]}" "$@"
