#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

# Keep the global rollout size at 4096 environments: 2048 per rank × 2 GPUs.
exec env \
  CUDA_VISIBLE_DEVICES=5,7 \
  MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --env.scene.num-envs 2048 \
  --env.scene.terrain.num-envs 2048 \
  --agent.max-iterations 4000 \
  --agent.save-interval 500 \
  --agent.run-name t001_joint_dr_full_2gpu_resume6000 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run 2026-08-19_04-56-15_t001_joint_dr_full_4096 \
  --agent.load-checkpoint model_6000.pt \
  --gpu-ids all \
  --video False
