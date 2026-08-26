#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
gpu_ids="${GPU_IDS:-0,1,2,3}"
num_envs="${NUM_ENVS:-8192}"
max_iterations="${MAX_ITERATIONS:-45}"
save_interval="${SAVE_INTERVAL:-5}"
run_name="${RUN_NAME:-tracking_sprint380_4gpu_8192_each_45}"
load_run="${LOAD_RUN:-migrated_sprint340}"
load_checkpoint="${LOAD_CHECKPOINT:-model_499.pt}"

IFS=',' read -ra requested_gpus <<< "$gpu_ids"
mapfile -t gpu_memory < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
for gpu in "${requested_gpus[@]}"; do
  used_memory="${gpu_memory[$gpu]:-999999}"
  if (( used_memory > 128 )); then
    echo "Refusing to start: GPU $gpu already uses ${used_memory} MiB." >&2
    exit 1
  fi
done

exec env CUDA_VISIBLE_DEVICES="$gpu_ids" "$python_bin" \
  scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed380.npz \
  --env.scene.num-envs "$num_envs" \
  --env.scene.terrain.num-envs "$num_envs" \
  --agent.max-iterations "$max_iterations" \
  --agent.save-interval "$save_interval" \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.005 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.run-name "$run_name" \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run "$load_run" \
  --agent.load-checkpoint "$load_checkpoint" \
  --weights-only-resume True \
  --gpu-ids all \
  --video False
