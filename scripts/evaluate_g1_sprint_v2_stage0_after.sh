#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

session_name="g1_sprint_v2_stage0"
run_dir="logs/rsl_rl/g1_velocity/2026-08-20_02-42-10_sprint_v2_stage0_2048_1000"
checkpoint="$run_dir/model_1997.pt"

# Bound the wait to one hour and refuse to evaluate an intermediate checkpoint
# if training exits early.
for _ in $(seq 1 120); do
  # Prefix matching would mistake g1_sprint_v2_stage0_eval for the training
  # session itself.  The leading '=' requests an exact tmux target match.
  if ! tmux has-session -t "=$session_name" 2>/dev/null; then
    break
  fi
  sleep 30
done

if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "Timed out waiting for $session_name" >&2
  exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Training ended without expected checkpoint: $checkpoint" >&2
  exit 1
fi

for speed_spec in "1.0:1p0" "1.5:1p5" "1.8:1p8"; do
  speed="${speed_spec%%:*}"
  label="${speed_spec##*:}"
  env CUDA_VISIBLE_DEVICES=5 \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
    MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
    /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
    scripts/evaluate.py Unitree-G1-Sprint-v2 \
    --checkpoint "$checkpoint" \
    --num-envs 64 \
    --seeds "(42,)" \
    --command-speed-mps "$speed" \
    --device cuda:0 \
    --output-dir "evaluations/Unitree-G1-Sprint-v2/stage0_model1997_speed${label}_seed42"
done
