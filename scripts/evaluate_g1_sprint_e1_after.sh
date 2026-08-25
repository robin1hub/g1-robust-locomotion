#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

session_name="g1_sprint_e1_phase"
run_dir="logs/rsl_rl/g1_velocity/2026-08-20_09-27-50_sprint_e1_adaptive_phase_4096_300"
screen_dir="evaluations/Unitree-G1-Sprint-v4-AdaptivePhase/e1_checkpoints_dr_speed1p5_seed42"

for _ in $(seq 1 120); do
  if ! tmux has-session -t "=$session_name" 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "Timed out waiting for $session_name" >&2
  exit 1
fi
if [[ ! -f "$run_dir/model_2999.pt" ]]; then
  echo "Missing expected final checkpoint: $run_dir/model_2999.pt" >&2
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

env "${runtime[@]}" "$python_bin" scripts/evaluate.py \
  Unitree-G1-Sprint-v4-AdaptivePhase \
  --checkpoint "$run_dir/model_*.pt" \
  --num-envs 64 --seeds "(42,)" --command-speed-mps 1.5 \
  --clean False --device cuda:0 --output-dir "$screen_dir"

decision="$screen_dir/best_checkpoint.json"
"$python_bin" scripts/select_sprint_v3_checkpoint.py \
  "$screen_dir/results.json" "$decision" || true
best=$("$python_bin" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$decision")
echo "Selected E1 checkpoint: $best"

for mode in clean dr; do
  clean_flag=False
  if [[ "$mode" == clean ]]; then clean_flag=True; fi
  env "${runtime[@]}" "$python_bin" scripts/evaluate.py \
    Unitree-G1-Sprint-v4-AdaptivePhase \
    --checkpoint "$best" \
    --num-envs 128 --seeds "(11,23,42,67,89)" --command-speed-mps 1.5 \
    --clean "$clean_flag" --device cuda:0 \
    --output-dir "evaluations/Unitree-G1-Sprint-v4-AdaptivePhase/e1_best_${mode}_5seed_128"
done
