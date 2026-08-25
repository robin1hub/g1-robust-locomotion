#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

exec env \
  CUDA_VISIBLE_DEVICES=5 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/train.py Unitree-G1-Sprint-v2 \
  --env.scene.num-envs 2048 \
  --env.scene.terrain.num-envs 2048 \
  --agent.max-iterations 500 \
  --agent.save-interval 100 \
  --agent.run-name sprint_v2_short_2048_500 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-18_03-34-44_g1_marathon_short_1024 \
  --agent.load-checkpoint model_499.pt \
  --gpu-ids '[0]' \
  --video False
