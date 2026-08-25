#!/usr/bin/env bash
set -euo pipefail

cd /data/users/yanghao/projects/unitree_rl_mjlab

exec /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/train.py Unitree-G1-Tracking \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.run-name lafan1_run_stage1_4096 \
  --agent.logger tensorboard \
  --gpu-ids '[6]'
