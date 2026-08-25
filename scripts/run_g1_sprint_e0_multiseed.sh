#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

checkpoint="logs/rsl_rl/g1_velocity/2026-08-20_07-52-59_sprint_v3_lane_adapt_4096_resume2400/model_2700.pt"
common=(
  Unitree-G1-Sprint-v3-Lane
  --checkpoint "$checkpoint"
  --num-envs 128
  --seeds "(11,23,42,67,89)"
  --command-speed-mps 1.5
  --device cuda:0
)
runtime=(
  CUDA_VISIBLE_DEVICES=5
  MUJOCO_GL=egl
  MUJOCO_EGL_DEVICE_ID=0
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib
)
python_bin="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python"

env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
  --clean True \
  --output-dir evaluations/Unitree-G1-Sprint-v3-Lane/e0_model2700_clean_5seed_128

env "${runtime[@]}" "$python_bin" scripts/evaluate.py "${common[@]}" \
  --clean False \
  --output-dir evaluations/Unitree-G1-Sprint-v3-Lane/e0_model2700_dr_5seed_128
