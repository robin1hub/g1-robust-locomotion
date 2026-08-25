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
  scripts/train.py Unitree-G1-Sprint-v4-AdaptivePhase \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 300 \
  --agent.save-interval 50 \
  --agent.run-name sprint_e1_adaptive_phase_4096_300 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.005 \
  --agent.algorithm.entropy-coef 0.004 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-20_07-52-59_sprint_v3_lane_adapt_4096_resume2400 \
  --agent.load-checkpoint model_2700.pt \
  --gpu-ids '[0]' \
  --video False
