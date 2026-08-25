#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

session_name="g1_sprint_e2a"
run_dir="logs/rsl_rl/g1_velocity/2026-08-20_10-05-45_sprint_e2a_command_4096_600"
root="evaluations/Unitree-G1-Sprint-E2A-Command/e2a_command_matrix_seed42"
source_checkpoint="logs/rsl_rl/g1_velocity/2026-08-20_07-52-59_sprint_v3_lane_adapt_4096_resume2400/model_2700.pt"

for _ in $(seq 1 160); do
  if ! tmux has-session -t "=$session_name" 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "Timed out waiting for $session_name" >&2
  exit 1
fi
if [[ ! -f "$run_dir/model_599.pt" ]]; then
  echo "Missing expected final checkpoint: $run_dir/model_599.pt" >&2
  exit 1
fi

runtime=(
  CUDA_VISIBLE_DEVICES=5
  MUJOCO_GL=egl
  MUJOCO_EGL_DEVICE_ID=0
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib
)
python_bin="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python"
common=(
  Unitree-G1-Sprint-E2A-Command
  --num-envs 64 --seeds "(42,)" --command-speed-mps 1.5
  --clean False --device cuda:0
)

env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
  --checkpoint "$source_checkpoint" \
  --output-dir "$root/baseline_straight"

for spec in \
  "straight:0.0:0.0" \
  "lat_pos:0.3:0.0" \
  "lat_neg:-0.3:0.0" \
  "yaw_pos:0.0:0.5" \
  "yaw_neg:0.0:-0.5"; do
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

"$python_bin" scripts/select_sprint_e2a_checkpoint.py \
  --baseline "$root/baseline_straight/results.json" \
  --straight "$root/straight/results.json" \
  --lat-pos "$root/lat_pos/results.json" \
  --lat-neg "$root/lat_neg/results.json" \
  --yaw-pos "$root/yaw_pos/results.json" \
  --yaw-neg "$root/yaw_neg/results.json" \
  --output "$root/decision.json"
