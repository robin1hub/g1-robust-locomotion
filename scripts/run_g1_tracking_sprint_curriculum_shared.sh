#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${ALLOW_SHARED_GPUS:-0}" != "1" ]]; then
  echo "Refusing shared GPUs: set ALLOW_SHARED_GPUS=1 explicitly." >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"
task_id="${TASK_ID:-Unitree-G1-Tracking}"
gpu_ids="${GPU_IDS:-0,1,2,3,4,5}"
num_envs="${NUM_ENVS:-384}"
max_iterations="${MAX_ITERATIONS:-40}"
save_interval="${SAVE_INTERVAL:-10}"
run_name="${RUN_NAME:-tracking_sprint430_flat_strict_probe40}"
motion_file="${MOTION_FILE:-src/assets/motions/g1/motiondecode_BG_Sprint_run_00687_cycle_periodic_v430.npz}"
load_run="${LOAD_RUN:-2026-08-25_10-44-39_tracking_sprint400_antislip_4gpu_8192_each_20}"
load_checkpoint="${LOAD_CHECKPOINT:-model_19.pt}"
init_mode="${INIT_MODE:-resume}"
termination_preset="${TERMINATION_PRESET:-strict}"
sampling_mode="${SAMPLING_MODE:-adaptive}"
learning_rate="${LEARNING_RATE:-0.0003}"
min_free_mib="${MIN_FREE_MIB:-8192}"

case "$termination_preset" in
  strict)
    anchor_pos_threshold="0.25"
    anchor_ori_threshold="0.8"
    ee_pos_threshold="0.25"
    ;;
  medium)
    anchor_pos_threshold="0.40"
    anchor_ori_threshold="1.0"
    ee_pos_threshold="0.40"
    ;;
  lenient)
    anchor_pos_threshold="0.60"
    anchor_ori_threshold="1.2"
    ee_pos_threshold="0.60"
    ;;
  *)
    echo "Unknown TERMINATION_PRESET: $termination_preset" >&2
    exit 1
    ;;
esac

case "$init_mode" in
  fresh)
    resume_args=()
    ;;
  resume)
    resume_args=(
      --agent.resume True
      --agent.load-run "$load_run"
      --agent.load-checkpoint "$load_checkpoint"
      --weights-only-resume True
    )
    ;;
  *)
    echo "Unknown INIT_MODE: $init_mode" >&2
    exit 1
    ;;
esac

IFS=',' read -ra requested_gpus <<< "$gpu_ids"
mapfile -t gpu_free < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
for gpu in "${requested_gpus[@]}"; do
  free_memory="${gpu_free[$gpu]:-0}"
  if (( free_memory < min_free_mib )); then
    echo "Refusing to start: GPU $gpu has only ${free_memory} MiB free." >&2
    exit 1
  fi
done

echo "Sharing GPUs $gpu_ids with $num_envs envs per worker."
echo "Initialization: $init_mode; termination: $termination_preset; sampling: $sampling_mode; learning rate: $learning_rate."

exec env CUDA_VISIBLE_DEVICES="$gpu_ids" "$python_bin" \
  scripts/train.py "$task_id" \
  --motion-file "$motion_file" \
  --env.scene.num-envs "$num_envs" \
  --env.scene.terrain.num-envs "$num_envs" \
  --env.commands.motion.sampling-mode "$sampling_mode" \
  --env.terminations.anchor-pos.params.threshold "$anchor_pos_threshold" \
  --env.terminations.anchor-ori.params.threshold "$anchor_ori_threshold" \
  --env.terminations.ee-body-pos.params.threshold "$ee_pos_threshold" \
  --agent.max-iterations "$max_iterations" \
  --agent.save-interval "$save_interval" \
  --agent.algorithm.learning-rate "$learning_rate" \
  --agent.algorithm.clip-param 0.2 \
  --agent.algorithm.desired-kl 0.01 \
  --agent.algorithm.num-mini-batches 4 \
  --agent.run-name "$run_name" \
  --agent.logger tensorboard \
  "${resume_args[@]}" \
  --gpu-ids all \
  --video False
