#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${ALLOW_SHARED_GPUS:-0}" != "1" ]]; then
  echo "Refusing shared GPUs: set ALLOW_SHARED_GPUS=1 explicitly." >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"
gpu_ids="${GPU_IDS:-0,1,2,3,4,5}"
num_envs="${NUM_ENVS:-512}"
max_iterations="${MAX_ITERATIONS:-5}"
run_name="${RUN_NAME:-tracking_sprint430_periodic_shared_6gpu_512_each_probe5}"
motion_file="${MOTION_FILE:-src/assets/motions/g1/motiondecode_BG_Sprint_run_00687_cycle_periodic_v430.npz}"
load_run="${LOAD_RUN:-2026-08-25_10-44-39_tracking_sprint400_antislip_4gpu_8192_each_20}"
load_checkpoint="${LOAD_CHECKPOINT:-model_10.pt}"
min_free_mib="${MIN_FREE_MIB:-8192}"

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

echo "WARNING: intentionally sharing GPUs $gpu_ids; ${num_envs} envs per worker."

exec env CUDA_VISIBLE_DEVICES="$gpu_ids" "$python_bin" \
  scripts/train.py Unitree-G1-Tracking-Robust-AntiSlip \
  --motion-file "$motion_file" \
  --env.scene.num-envs "$num_envs" \
  --env.scene.terrain.num-envs "$num_envs" \
  --agent.max-iterations "$max_iterations" \
  --agent.save-interval "$max_iterations" \
  --agent.algorithm.learning-rate 0.00003 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.003 \
  --agent.algorithm.num-mini-batches 4 \
  --agent.run-name "$run_name" \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run "$load_run" \
  --agent.load-checkpoint "$load_checkpoint" \
  --weights-only-resume True \
  --gpu-ids all \
  --video False
