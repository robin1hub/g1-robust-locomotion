from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_marathon_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_sprint_v2_env_cfg,
  unitree_g1_sprint_v3_env_cfg,
  unitree_g1_sprint_v3_lane_env_cfg,
  unitree_g1_sprint_v4_adaptive_phase_env_cfg,
  unitree_g1_sprint_e2a_command_env_cfg,
  unitree_g1_sprint_e2b0_yaw_probe_env_cfg,
  unitree_g1_sprint_e2b0a_task_fix_env_cfg,
  unitree_g1_sprint_e2b0b_reward_fix_env_cfg,
  unitree_g1_sprint_e2b0b2_yaw_focus_env_cfg,
  unitree_g1_sprint_e2b0b3_yaw030_env_cfg,
  unitree_g1_sprint_e2b1_yaw050_env_cfg,
  unitree_g1_sprint_s1_speed220_env_cfg,
  unitree_g1_sprint_s2_speed280_env_cfg,
  unitree_g1_sprint_s3_speed340_env_cfg,
)
from .rl_cfg import (
  unitree_g1_ppo_runner_cfg,
  unitree_g1_symmetry_ppo_runner_cfg,
)

# 导入本模块时便将任务写入 MjLab 的全局 registry。train.py 之后只需给出
# "Unitree-G1-Flat"，就能取得这里绑定的 env_cfg、rl_cfg 和 runner_cls。
register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Marathon",
  env_cfg=unitree_g1_marathon_env_cfg(),
  play_env_cfg=unitree_g1_marathon_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-v2",
  env_cfg=unitree_g1_sprint_v2_env_cfg(),
  play_env_cfg=unitree_g1_sprint_v2_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-v3",
  env_cfg=unitree_g1_sprint_v3_env_cfg(),
  play_env_cfg=unitree_g1_sprint_v3_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-v3-Lane",
  env_cfg=unitree_g1_sprint_v3_lane_env_cfg(),
  play_env_cfg=unitree_g1_sprint_v3_lane_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-v4-AdaptivePhase",
  env_cfg=unitree_g1_sprint_v4_adaptive_phase_env_cfg(),
  play_env_cfg=unitree_g1_sprint_v4_adaptive_phase_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2A-Command",
  env_cfg=unitree_g1_sprint_e2a_command_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2a_command_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2B0-Yaw-Probe",
  env_cfg=unitree_g1_sprint_e2b0_yaw_probe_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b0_yaw_probe_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2B0A-Task-Fix",
  env_cfg=unitree_g1_sprint_e2b0a_task_fix_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b0a_task_fix_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2B0B-Reward-Fix",
  env_cfg=unitree_g1_sprint_e2b0b_reward_fix_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b0b_reward_fix_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2B0B2-Yaw-Focus",
  env_cfg=unitree_g1_sprint_e2b0b2_yaw_focus_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b0b2_yaw_focus_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2B0B3-Yaw030",
  env_cfg=unitree_g1_sprint_e2b0b3_yaw030_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b0b3_yaw030_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2B1-Yaw050",
  env_cfg=unitree_g1_sprint_e2b1_yaw050_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b1_yaw050_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-E2C-Symmetry",
  env_cfg=unitree_g1_sprint_e2b1_yaw050_env_cfg(),
  play_env_cfg=unitree_g1_sprint_e2b1_yaw050_env_cfg(play=True),
  rl_cfg=unitree_g1_symmetry_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-S1-Speed220",
  env_cfg=unitree_g1_sprint_s1_speed220_env_cfg(),
  play_env_cfg=unitree_g1_sprint_s1_speed220_env_cfg(play=True),
  rl_cfg=unitree_g1_symmetry_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-S2-Speed280",
  env_cfg=unitree_g1_sprint_s2_speed280_env_cfg(),
  play_env_cfg=unitree_g1_sprint_s2_speed280_env_cfg(play=True),
  rl_cfg=unitree_g1_symmetry_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Sprint-S3-Speed340",
  env_cfg=unitree_g1_sprint_s3_speed340_env_cfg(),
  play_env_cfg=unitree_g1_sprint_s3_speed340_env_cfg(play=True),
  rl_cfg=unitree_g1_symmetry_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  # 我们初步要跑的平地速度任务。它与 Rough 共用 PPO 配置和 runner，主要区别
  # 在地形、地形扫描传感器以及对应的 observation/curriculum。
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
