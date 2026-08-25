#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

root="evaluations/Unitree-G1-Sprint-E2B0-AB/e2b0_ab_seed42/b0b_parallel"
source_checkpoint="logs/rsl_rl/g1_velocity/2026-08-20_10-05-45_sprint_e2a_command_4096_600/model_350.pt"
python_bin="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python"
runtime=(
  CUDA_VISIBLE_DEVICES=4
  MUJOCO_GL=egl
  MUJOCO_EGL_DEVICE_ID=0
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib
)
shopt -s nullglob
run_dirs=(logs/rsl_rl/g1_velocity/*_sprint_e2b0b_reward_fix_4096_100)
if (( ${#run_dirs[@]} == 0 )); then
  echo "Could not locate E2-B0B run directory" >&2
  exit 1
fi
run_dir="${run_dirs[-1]}"
common=(
  Unitree-G1-Sprint-E2B0B-Reward-Fix
  --num-envs 64 --seeds "(42,)" --command-speed-mps 1.5
  --clean False --device cuda:0
)

env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
  --checkpoint "$source_checkpoint" --output-dir "$root/baseline_straight"
for spec in \
  "straight:0.0:0.0" \
  "lat_pos:0.3:0.0" \
  "lat_neg:-0.3:0.0" \
  "yaw_pos_015:0.0:0.15" \
  "yaw_neg_015:0.0:-0.15"; do
  name="${spec%%:*}"
  rest="${spec#*:}"
  lateral="${rest%%:*}"
  yaw="${rest##*:}"
  env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
    --checkpoint "$run_dir/model_*.pt" \
    --command-lateral-speed-mps "$lateral" \
    --command-yaw-rate-radps "$yaw" \
    --output-dir "$root/$name"
done
"$python_bin" scripts/select_sprint_e2b0_checkpoint.py \
  --baseline "$root/baseline_straight/results.json" \
  --straight "$root/straight/results.json" \
  --lat-pos "$root/lat_pos/results.json" \
  --lat-neg "$root/lat_neg/results.json" \
  --yaw-pos-015 "$root/yaw_pos_015/results.json" \
  --yaw-neg-015 "$root/yaw_neg_015/results.json" \
  --output "$root/decision.json"
