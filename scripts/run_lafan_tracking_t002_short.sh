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
  scripts/train.py Unitree-G1-Tracking-Robust-History \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --env.scene.num-envs 2048 \
  --env.scene.terrain.num-envs 2048 \
  --agent.max-iterations 200 \
  --agent.save-interval 100 \
  --agent.run-name t002_history4_short_2048_200 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run t002_history4_init_from_t001 \
  --agent.load-checkpoint model_9999.pt \
  --gpu-ids '[0]' \
  --video False
