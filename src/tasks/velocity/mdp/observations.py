from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# 所有返回值的第0维都是并行环境 B；函数只读取仿真状态，不修改环境。


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  # site_pos_w 是世界坐标系位置；最后一维索引2代表 z 高度。
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  # 接触力可能有很大的尖峰。保留符号的 log1p 压缩可降低输入尺度，同时不丢方向。
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    # 用 sin/cos 编码周期相位，避免标量相位从1跳回0产生不连续。策略可利用它
    # 学出左右脚交替的周期步态；静止指令下清零，避免机器人原地踏步。
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def running_gait_period(
  speed: torch.Tensor,
  speed_range: tuple[float, float] = (0.5, 4.0),
  period_range: tuple[float, float] = (0.55, 0.30),
) -> torch.Tensor:
  """Interpolate the gait period used by both observation and reward terms."""
  lo, hi = speed_range
  blend = torch.clamp((speed - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
  slow_period, fast_period = period_range
  return slow_period + blend * (fast_period - slow_period)


def adaptive_running_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_range: tuple[float, float] = (0.5, 4.0),
  period_range: tuple[float, float] = (0.55, 0.30),
) -> torch.Tensor:
  """Sin/cos gait phase using the same speed-dependent period as the reward."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = torch.clamp(command[:, 0], min=0.0)
  period = running_gait_period(speed, speed_range, period_range)
  global_phase = (env.episode_length_buf * env.step_dt) / period
  phase_obs = torch.stack(
    (torch.sin(global_phase * 2.0 * torch.pi), torch.cos(global_phase * 2.0 * torch.pi)),
    dim=-1,
  )
  return torch.where(
    (speed >= speed_range[0]).unsqueeze(1), phase_obs, torch.zeros_like(phase_obs)
  )


def zero_track_state(env: ManagerBasedRlEnv, size: int = 5) -> torch.Tensor:
  """Keep a checkpoint-compatible observation slot without world-track leakage."""
  return torch.zeros(env.num_envs, size, device=env.device)


def straight_track_state(
  env: ManagerBasedRlEnv,
  lane_half_width: float = 0.9,
  speed_scale: float = 4.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Deployment-available state relative to a world +X straight track.

  Returns normalized lateral position, heading cosine/sine, and normalized
  world forward/lateral velocity.  These signals close the observability gap
  between the world-frame lane rewards and the actor's body-frame inputs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  world_velocity = asset.data.root_link_lin_vel_w
  body_forward = torch.zeros_like(world_velocity)
  body_forward[:, 0] = 1.0
  forward_axis_w = quat_apply(asset.data.root_link_quat_w, body_forward)
  lateral_position = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return torch.stack(
    (
      lateral_position / lane_half_width,
      forward_axis_w[:, 0],
      forward_axis_w[:, 1],
      world_velocity[:, 0] / speed_scale,
      world_velocity[:, 1] / speed_scale,
    ),
    dim=-1,
  )
