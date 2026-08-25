#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/or/relative/path/to/model_N.pt" >&2
  exit 2
fi

cd /data/users/yanghao/projects/unitree_rl_mjlab

checkpoint="$1"
motion_file="src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz"

runtime_lib_dir="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib"
exec env CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH="${runtime_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/play.py Unitree-G1-Tracking \
  --checkpoint-file "$checkpoint" \
  --motion-file "$motion_file" \
  --num-envs 1 \
  --device cuda:0 \
  --video True \
  --video-length 500 \
  --no-terminations True
