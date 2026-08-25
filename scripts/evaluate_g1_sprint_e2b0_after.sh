#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

session_name="g1_sprint_e2b0"
root="evaluations/Unitree-G1-Sprint-E2B0-Yaw-Probe/e2b0_command_matrix_seed42"
source_checkpoint="logs/rsl_rl/g1_velocity/2026-08-20_10-05-45_sprint_e2a_command_4096_600/model_350.pt"

for _ in $(seq 1 120); do
  if ! tmux has-session -t "=$session_name" 2>/dev/null; then break; fi
  sleep 15
done
if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "Timed out waiting for $session_name" >&2
  exit 1
fi
shopt -s nullglob
run_dirs=(logs/rsl_rl/g1_velocity/*_sprint_e2b0_yaw_probe_4096_200)
if (( ${#run_dirs[@]} == 0 )); then
  echo "Could not locate E2-B0 training directory" >&2
  exit 1
fi
run_dir="${run_dirs[-1]}"
if [[ ! -f "$run_dir/model_199.pt" ]]; then
  echo "Missing expected final checkpoint: $run_dir/model_199.pt" >&2
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
  Unitree-G1-Sprint-E2B0-Yaw-Probe
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
  "yaw_pos_015:0.0:0.15" \
  "yaw_neg_015:0.0:-0.15" \
  "yaw_pos_030:0.0:0.30" \
  "yaw_neg_030:0.0:-0.30"; do
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
  --yaw-pos-030 "$root/yaw_pos_030/results.json" \
  --yaw-neg-030 "$root/yaw_neg_030/results.json" \
  --output "$root/decision.json"
