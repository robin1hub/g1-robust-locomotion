#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

exec env \
  CUDA_VISIBLE_DEVICES=1 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/train.py Unitree-G1-Sprint-S3-Natural-v1 \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 200 \
  --agent.save-interval 25 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.005 \
  --agent.algorithm.entropy-coef 0.004 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.run-name sprint_s3_natural_v1_4096_200 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-25_03-31-41_sprint_s3_speed340_4096_400 \
  --agent.load-checkpoint model_375.pt \
  --weights-only-resume True \
  --gpu-ids '[0]' \
  --video False
