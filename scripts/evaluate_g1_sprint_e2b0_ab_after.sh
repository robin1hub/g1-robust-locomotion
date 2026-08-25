#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

session_a="g1_sprint_e2b0a"
session_b="g1_sprint_e2b0b"
root="evaluations/Unitree-G1-Sprint-E2B0-AB/e2b0_ab_seed42"
source_checkpoint="logs/rsl_rl/g1_velocity/2026-08-20_10-05-45_sprint_e2a_command_4096_600/model_350.pt"
python_bin="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python"
runtime=(
  CUDA_VISIBLE_DEVICES=5
  MUJOCO_GL=egl
  MUJOCO_EGL_DEVICE_ID=0
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib
)

for _ in $(seq 1 160); do
  if ! tmux has-session -t "=$session_a" 2>/dev/null \
    && ! tmux has-session -t "=$session_b" 2>/dev/null; then
    break
  fi
  sleep 15
done
if tmux has-session -t "=$session_a" 2>/dev/null \
  || tmux has-session -t "=$session_b" 2>/dev/null; then
  echo "Timed out waiting for E2-B0 A/B training" >&2
  exit 1
fi

shopt -s nullglob
a_dirs=(logs/rsl_rl/g1_velocity/*_sprint_e2b0a_task_fix_4096_100)
b_dirs=(logs/rsl_rl/g1_velocity/*_sprint_e2b0b_reward_fix_4096_100)
if (( ${#a_dirs[@]} == 0 || ${#b_dirs[@]} == 0 )); then
  echo "Could not locate both E2-B0 A/B run directories" >&2
  exit 1
fi
a_run="${a_dirs[-1]}"
b_run="${b_dirs[-1]}"
for run_dir in "$a_run" "$b_run"; do
  if [[ ! -f "$run_dir/model_99.pt" ]]; then
    echo "Missing expected final checkpoint: $run_dir/model_99.pt" >&2
    exit 1
  fi
done

evaluate_variant() {
  local label="$1"
  local task="$2"
  local run_dir="$3"
  local variant_root="$root/$label"
  local common=(
    "$task" --num-envs 64 --seeds "(42,)" --command-speed-mps 1.5
    --clean False --device cuda:0
  )

  env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
    --checkpoint "$source_checkpoint" \
    --output-dir "$variant_root/baseline_straight"
  for spec in \
    "straight:0.0:0.0" \
    "lat_pos:0.3:0.0" \
    "lat_neg:-0.3:0.0" \
    "yaw_pos_015:0.0:0.15" \
    "yaw_neg_015:0.0:-0.15"; do
    local name="${spec%%:*}"
    local rest="${spec#*:}"
    local lateral="${rest%%:*}"
    local yaw="${rest##*:}"
    env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
      --checkpoint "$run_dir/model_*.pt" \
      --command-lateral-speed-mps "$lateral" \
      --command-yaw-rate-radps "$yaw" \
      --output-dir "$variant_root/$name"
  done
  "$python_bin" scripts/select_sprint_e2b0_checkpoint.py \
    --baseline "$variant_root/baseline_straight/results.json" \
    --straight "$variant_root/straight/results.json" \
    --lat-pos "$variant_root/lat_pos/results.json" \
    --lat-neg "$variant_root/lat_neg/results.json" \
    --yaw-pos-015 "$variant_root/yaw_pos_015/results.json" \
    --yaw-neg-015 "$variant_root/yaw_neg_015/results.json" \
    --output "$variant_root/decision.json"
}

evaluate_variant \
  "b0a" "Unitree-G1-Sprint-E2B0A-Task-Fix" "$a_run"
evaluate_variant \
  "b0b" "Unitree-G1-Sprint-E2B0B-Reward-Fix" "$b_run"

"$python_bin" scripts/compare_sprint_e2b0_ab.py \
  --a "$root/b0a/decision.json" \
  --b "$root/b0b/decision.json" \
  --output "$root/decision.json"
