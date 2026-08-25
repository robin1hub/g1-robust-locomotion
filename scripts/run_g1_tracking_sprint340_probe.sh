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
  scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 500 \
  --agent.save-interval 50 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.005 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.run-name tracking_sprint340_probe_4096_500 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-19_06-28-23_t001_joint_dr_full_2gpu_resume6000 \
  --agent.load-checkpoint model_9999.pt \
  --weights-only-resume True \
  --gpu-ids '[0]' \
  --video False
