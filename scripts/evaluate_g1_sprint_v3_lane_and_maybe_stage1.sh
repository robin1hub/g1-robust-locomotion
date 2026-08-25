#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

session_name="g1_sprint_v3_lane_4096"
run_dir="logs/rsl_rl/g1_velocity/2026-08-20_07-52-59_sprint_v3_lane_adapt_4096_resume2400"
output_dir="evaluations/Unitree-G1-Sprint-v3-Lane/lane_adapt_4096_checkpoints_speed1p5_seed42"

for _ in $(seq 1 120); do
  if ! tmux has-session -t "=$session_name" 2>/dev/null; then
    break
  fi
  sleep 30
done
if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "Timed out waiting for $session_name" >&2
  exit 1
fi
if [[ ! -f "$run_dir/model_2799.pt" ]]; then
  echo "Training ended without expected checkpoint: $run_dir/model_2799.pt" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES=5 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/evaluate.py Unitree-G1-Sprint-v3-Lane \
  --checkpoint "$run_dir/model_*.pt" \
  --num-envs 64 \
  --seeds "(42,)" \
  --command-speed-mps 1.5 \
  --device cuda:0 \
  --output-dir "$output_dir"

decision="$output_dir/stage1_decision.json"
if /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/select_sprint_v3_checkpoint.py "$output_dir/results.json" "$decision"; then
  best=$(/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$decision")
  mkdir -p logs/rsl_rl/g1_velocity/sprint_v3_stage1_warmstart
  cp "$best" logs/rsl_rl/g1_velocity/sprint_v3_stage1_warmstart/model_best.pt
  tmux new-session -d -s g1_sprint_v3_stage1 \
    "bash -c 'bash scripts/run_g1_sprint_v3_stage1.sh > logs/benchmarks/g1_sprint_v3_stage1_20260820.log 2>&1'"
  echo "Stage-1 gate passed; launched g1_sprint_v3_stage1 from $best"
else
  echo "Stage-1 gate failed; faster training was not launched. See $decision" >&2
  exit 2
fi
