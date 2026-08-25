#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

exec env \
  CUDA_VISIBLE_DEVICES=6 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.save-interval 500 \
  --agent.run-name t001_joint_dr_full_4096 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-18_07-22-34_lafan1_run_stage1_4096 \
  --agent.load-checkpoint model_4999.pt \
  --gpu-ids '[0]' \
  --video False
