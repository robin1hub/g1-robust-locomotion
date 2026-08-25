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
  scripts/train.py Unitree-G1-Sprint-v3-Lane \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 400 \
  --agent.save-interval 100 \
  --agent.run-name sprint_v3_lane_adapt_4096_resume2400 \
  --agent.algorithm.learning-rate 0.0003 \
  --agent.algorithm.entropy-coef 0.005 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-20_07-48-17_sprint_v3_lane_adapt_2048_500 \
  --agent.load-checkpoint model_2400.pt \
  --gpu-ids '[0]' \
  --video False
