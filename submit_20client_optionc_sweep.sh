#!/bin/bash
set -euo pipefail

# Submit independent jobs for the corrected 20-client Option-C benchmark.
# One job = one method x one q value x one seed.
# Result roots and experiment-log tags are 20-client- and q-specific.

BENCHMARK_SETUP=${BENCHMARK_SETUP:-twenty_client}
Q_VALUES=${Q_VALUES:-"0.2 0.5 0.8"}
SEEDS=${SEEDS:-"0 1 2"}
ROUNDS=${ROUNDS:-80}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
RUN_TAG_BASE=${RUN_TAG_BASE:-optionc20_main}
DRY_RUN=${DRY_RUN:-0}
METHOD_FILTER=${METHOD_FILTER:-""}

# DAIC scheduling defaults. These command-line options override #SBATCH values
# inside the individual Slurm files.
SBATCH_PARTITION=${SBATCH_PARTITION:-ewi-st}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-ewi-st-dis}
SBATCH_QOS=${SBATCH_QOS:-medium}
SBATCH_EXCLUDE=${SBATCH_EXCLUDE:-cor1,influ2,gpu15}

# Optional wall-time override for every selected method, e.g.
# TIME_OVERRIDE=02:00:00 for smoke tests.
TIME_OVERRIDE=${TIME_OVERRIDE:-""}

if [[ "$BENCHMARK_SETUP" != "twenty_client" ]]; then
  echo "ERROR: this submitter is only for BENCHMARK_SETUP=twenty_client" >&2
  exit 1
fi

# Format: slurm_file|short_method_name|memory|wall_time
# APPLE-Full and APPLE-PostALA receive 64G. Full runs receive 24 hours.
METHODS=(
  "run_fedavg_multi_select.slurm|fedavg|12G|24:00:00"
  "run_fedprox_multi_select.slurm|fedprox|12G|24:00:00"
  "run_fully_local_multi_select.slurm|fully_local|12G|24:00:00"
  "run_gcflplus_multi_select.slurm|gcflplus|12G|24:00:00"
  "run_fedala_fedavg_multi_select.slurm|fedala|24G|24:00:00"
  "run_apple_backbone_task_head_multi_select.slurm|apple|64G|24:00:00"
  "run_apple_ala_multi_select.slurm|apple_ala|32G|24:00:00"
  "run_apple_post_ala_multi_select.slurm|apple_post_ala|64G|24:00:00"
  "run_local_centralized_multi_select.slurm|global_centralized|12G|24:00:00"
)

submitted=0

for entry in "${METHODS[@]}"; do
  IFS='|' read -r script method mem wall_time <<< "$entry"

  if [[ ! -f "$script" ]]; then
    echo "ERROR: missing Slurm file: $script" >&2
    exit 1
  fi

  if [[ -n "$METHOD_FILTER" ]] && [[ ! "$method" =~ $METHOD_FILTER ]] && [[ ! "$script" =~ $METHOD_FILTER ]]; then
    continue
  fi

  effective_time="$wall_time"
  if [[ -n "$TIME_OVERRIDE" ]]; then
    effective_time="$TIME_OVERRIDE"
  fi

  for q in $Q_VALUES; do
    q_tag=${q//./}
    run_tag="${RUN_TAG_BASE}_q${q_tag}"
    runs_root="andrea/runs_${RUN_TAG_BASE}_${method}_q${q_tag}"

    for seed in $SEEDS; do
      job_name="${method}_q${q_tag}_s${seed}"
      export_values="ALL,BENCHMARK_SETUP=${BENCHMARK_SETUP},ONLY_Q=${q},ONLY_SEED=${seed},MAX_SUBSETS=1,ROUNDS=${ROUNDS},LOCAL_EPOCHS=${LOCAL_EPOCHS},SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS},RUN_TAG=${run_tag},RUNS_ROOT=${runs_root}"

      sbatch_args=(
        --partition="$SBATCH_PARTITION"
        --account="$SBATCH_ACCOUNT"
        --qos="$SBATCH_QOS"
        --mem="$mem"
        --time="$effective_time"
        --job-name="$job_name"
        --export="$export_values"
      )

      if [[ -n "$SBATCH_EXCLUDE" ]]; then
        sbatch_args+=(--exclude="$SBATCH_EXCLUDE")
      fi

      if [[ "$DRY_RUN" == "1" ]]; then
        printf 'sbatch'
        printf ' %q' "${sbatch_args[@]}"
        printf ' %q\n' "$script"
      else
        sbatch "${sbatch_args[@]}" "$script"
      fi

      submitted=$((submitted + 1))
    done
  done
done

echo "Prepared/submitted ${submitted} jobs."
echo "Benchmark : ${BENCHMARK_SETUP}"
echo "q values  : ${Q_VALUES}"
echo "seeds     : ${SEEDS}"
echo "rounds    : ${ROUNDS}"
echo "run tag   : ${RUN_TAG_BASE}"
echo "partition : ${SBATCH_PARTITION}"
echo "account   : ${SBATCH_ACCOUNT}"
echo "qos       : ${SBATCH_QOS}"
echo "exclude   : ${SBATCH_EXCLUDE}"
echo "time ovrd : ${TIME_OVERRIDE:-<per-method>}"
