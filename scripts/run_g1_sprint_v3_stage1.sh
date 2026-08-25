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
  --env.scene.num-envs 2048 \
  --env.scene.terrain.num-envs 2048 \
  --agent.max-iterations 1000 \
  --agent.save-interval 200 \
  --agent.run-name sprint_v3_stage1_2048_1000 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run sprint_v3_stage1_warmstart \
  --agent.load-checkpoint model_best.pt \
  --gpu-ids '[0]' \
  --video False
