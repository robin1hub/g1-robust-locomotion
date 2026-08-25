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
  scripts/train.py Unitree-G1-Sprint-E2B0B-Reward-Fix \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 100 \
  --agent.save-interval 25 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.005 \
  --agent.algorithm.entropy-coef 0.004 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.run-name sprint_e2b0b_reward_fix_4096_100 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-20_10-05-45_sprint_e2a_command_4096_600 \
  --agent.load-checkpoint model_350.pt \
  --weights-only-resume True \
  --gpu-ids '[0]' \
  --video False
