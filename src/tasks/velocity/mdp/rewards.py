from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from .observations import running_gait_period

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# RewardManager 最终会计算 weight * func(...)。因此这里通常返回非负“得分”或
# 非负“代价”；一个函数究竟奖励还是惩罚，要结合配置中的 weight 正负号判断。


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  # command 和 actual 都在机器人机体坐标系。水平速度跟踪是主要目标，同时额外
  # 抑制不需要的竖直速度。指数核把误差0映射为1，误差增大时平滑趋近0。
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def yaw_rate_tracking_error_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Non-saturating squared body-frame yaw-rate tracking error."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])


def forward_velocity_tracking_error_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Non-saturating squared body-frame forward-velocity tracking error."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return torch.square(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])


def forward_progress(
  env: ManagerBasedRlEnv,
  max_speed: float = 4.5,
  upright_power: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward speed while suppressing fall-and-slide reward exploits."""
  asset: Entity = env.scene[asset_cfg.name]
  forward_speed = torch.clamp(asset.data.root_link_lin_vel_b[:, 0], 0.0, max_speed)
  tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  upright_gate = torch.exp(-upright_power * tilt_sq)
  return (forward_speed / max_speed) * upright_gate


def straight_track_progress(
  env: ManagerBasedRlEnv,
  max_speed: float = 4.5,
  lane_half_width: float = 0.9,
  upright_power: float = 2.0,
  heading_power: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward world-frame +X progress only while upright, aligned, and in lane.

  Unlike body-frame forward speed, this cannot be maximized by continuously
  turning the robot. The lane gate also prevents sideways drift followed by a
  late correction from receiving the same reward as a straight sprint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  world_velocity = asset.data.root_link_lin_vel_w
  world_forward_speed = torch.clamp(world_velocity[:, 0], 0.0, max_speed)

  body_forward = torch.zeros_like(world_velocity)
  body_forward[:, 0] = 1.0
  forward_axis_w = quat_apply(asset.data.root_link_quat_w, body_forward)
  heading_gate = torch.clamp(forward_axis_w[:, 0], 0.0, 1.0).pow(heading_power)

  lateral_position = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  lane_gate = torch.exp(-torch.square(lateral_position / lane_half_width))
  tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  upright_gate = torch.exp(-upright_power * tilt_sq)

  env.extras["log"]["Metrics/world_forward_speed_mps"] = torch.mean(
    world_velocity[:, 0]
  )
  env.extras["log"]["Metrics/lateral_position_abs_m"] = torch.mean(
    torch.abs(lateral_position)
  )
  env.extras["log"]["Metrics/heading_alignment"] = torch.mean(forward_axis_w[:, 0])
  return (world_forward_speed / max_speed) * heading_gate * lane_gate * upright_gate


def straight_track_lateral_position_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared distance from the world-frame straight track centerline."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral_position = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return torch.square(lateral_position)


def straight_track_lane_barrier_l4(
  env: ManagerBasedRlEnv,
  lane_half_width: float = 0.9,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Barrier-like lane cost that grows rapidly near the track boundary."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral_position = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return torch.pow(torch.abs(lateral_position) / lane_half_width, 4)


def outward_lateral_velocity(
  env: ManagerBasedRlEnv,
  center_deadzone: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize velocity that increases distance from the track centerline."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral_position = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  outward_speed = torch.sign(lateral_position) * asset.data.root_link_lin_vel_w[:, 1]
  active = torch.abs(lateral_position) > center_deadzone
  return torch.where(active, torch.relu(outward_speed), torch.zeros_like(outward_speed))


def world_lateral_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize world-frame lateral velocity during a straight sprint."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 1])


def straight_track_heading_error_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation of the robot's forward axis from world +X."""
  asset: Entity = env.scene[asset_cfg.name]
  body_forward = torch.zeros_like(asset.data.root_link_lin_vel_w)
  body_forward[:, 0] = 1.0
  forward_axis_w = quat_apply(asset.data.root_link_quat_w, body_forward)
  return torch.square(1.0 - forward_axis_w[:, 0]) + torch.square(forward_axis_w[:, 1])


def world_yaw_rate_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize world-frame yaw rate to close the Marathon-v1 spin loophole."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_w[:, 2])


def normalized_mechanical_power(
  env: ManagerBasedRlEnv,
  power_scale: float = 1000.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mechanical power proxy, normalized to keep reward weights interpretable."""
  asset: Entity = env.scene[asset_cfg.name]
  actuator_force = asset.data.actuator_force[:, asset_cfg.joint_ids]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  power = torch.sum(torch.abs(actuator_force * joint_vel), dim=1)
  env.extras["log"]["Metrics/mechanical_power_mean_w"] = torch.mean(power)
  return power / power_scale


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  # 直立时重力在机体系主要指向 z，xy 投影接近0。配置给它负权重，因此倾斜越多
  # 总奖励扣得越多。
  return xy_squared


def speed_dependent_torso_lean_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_range: tuple[float, float] = (1.5, 2.2),
  lean_range_deg: tuple[float, float] = (2.0, 8.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize torso roll and error from a speed-dependent forward lean target."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
  else:
    projected_gravity_b = asset.data.projected_gravity_b

  speed_lo, speed_hi = speed_range
  blend = torch.clamp(
    (command[:, 0] - speed_lo) / max(speed_hi - speed_lo, 1.0e-6), 0.0, 1.0
  )
  lean_lo, lean_hi = lean_range_deg
  target_lean_rad = torch.deg2rad(lean_lo + blend * (lean_hi - lean_lo))
  target_gravity_x = torch.sin(target_lean_rad)
  pitch_error = projected_gravity_b[:, 0] - target_gravity_x
  roll_error = projected_gravity_b[:, 1]
  env.extras["log"]["Metrics/target_torso_lean_deg"] = torch.mean(
    torch.rad2deg(target_lean_rad)
  )
  env.extras["log"]["Metrics/actual_torso_lean_deg"] = torch.mean(
    torch.rad2deg(torch.asin(torch.clamp(projected_gravity_b[:, 0], -1.0, 1.0)))
  )
  return torch.square(pitch_error) + torch.square(roll_error)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    # 返回超过阈值的子步数量，而不是仅返回是否碰撞，让持续碰撞受到更大惩罚。
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  penalize_yaw: bool = True,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  if not penalize_yaw:
    angmom = angmom[..., :2]
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    # offset 给每只脚分配期望相位；双足常用相差0.5，使左右脚交替支撑。
    # 实际接触状态与期望 stance 一致的比例就是步态得分。
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def adaptive_running_gait(
  env: ManagerBasedRlEnv,
  offset: list[float],
  command_name: str,
  sensor_name: str,
  speed_range: tuple[float, float] = (0.5, 4.0),
  period_range: tuple[float, float] = (0.55, 0.30),
  stance_range: tuple[float, float] = (0.55, 0.38),
) -> torch.Tensor:
  """Match a speed-adaptive alternating gait with a flight phase at high speed."""
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  speed = torch.clamp(command[:, 0], min=0.0)
  lo, hi = speed_range
  blend = torch.clamp((speed - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
  slow_stance, fast_stance = stance_range
  period = running_gait_period(speed, speed_range, period_range)
  stance_ratio = slow_stance + blend * (fast_stance - slow_stance)

  global_phase = (env.episode_length_buf * env.step_dt) / period
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype)
  leg_phase = (global_phase.unsqueeze(1) + offsets.unsqueeze(0)) % 1.0
  expected_contact = leg_phase < stance_ratio.unsqueeze(1)
  actual_contact = sensor.data.current_contact_time > 0
  reward = (expected_contact == actual_contact).float().mean(dim=1)
  return reward * (speed >= lo).float()


class phase_motion_joint_style:
  """Reward a phase-aligned running pose from a retargeted G1 motion cycle.

  The reference supplies style only. Forward speed, balance, contacts, and
  robustness remain controlled by the velocity task's existing rewards.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    motion_file = Path(cfg.params["motion_file"]).expanduser()
    if not motion_file.is_absolute():
      motion_file = Path.cwd() / motion_file
    with np.load(motion_file) as motion:
      joint_pos = motion["joint_pos"]
    frame_start = int(cfg.params["frame_start"])
    frame_end = int(cfg.params["frame_end"])
    reference = joint_pos[frame_start:frame_end]
    if reference.ndim != 2 or len(reference) < 2:
      raise ValueError("Motion style reference must contain at least two frames")

    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_count = asset.data.joint_pos[:, cfg.params["asset_cfg"].joint_ids].shape[1]
    if reference.shape[1] != joint_count:
      raise ValueError(
        f"Motion has {reference.shape[1]} joints but environment has {joint_count}"
      )
    self.reference = torch.as_tensor(reference, device=env.device, dtype=torch.float32)

    # Legs define foot placement and knee flexion; arms retain a softer style
    # target so PPO can still use them for balance and turning.
    joint_weights = torch.ones(joint_count, device=env.device)
    joint_weights[12:15] = 0.5
    joint_weights[15:] = 0.7
    self.joint_weights = joint_weights

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    motion_file: str,
    frame_start: int,
    frame_end: int,
    command_name: str,
    std: float,
    speed_range: tuple[float, float],
    period_range: tuple[float, float],
    minimum_speed: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del motion_file, frame_start, frame_end
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    speed = torch.clamp(command[:, 0], min=0.0)
    period = running_gait_period(speed, speed_range, period_range)
    phase = torch.remainder((env.episode_length_buf * env.step_dt) / period, 1.0)

    frame = phase * self.reference.shape[0]
    index0 = torch.floor(frame).long() % self.reference.shape[0]
    index1 = (index0 + 1) % self.reference.shape[0]
    alpha = (frame - torch.floor(frame)).unsqueeze(1)
    target = self.reference[index0] * (1.0 - alpha) + self.reference[index1] * alpha

    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    squared_error = torch.square(current - target) * self.joint_weights
    mse = torch.mean(squared_error, dim=1)
    env.extras["log"]["Metrics/motion_style_joint_rmse"] = torch.sqrt(
      torch.mean(mse)
    )
    return torch.exp(-mse / std**2) * (speed >= minimum_speed).float()


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    # 根据指令而非实际速度选择容忍度，防止策略通过“故意不动”进入更宽松区间。
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    # 对各关节误差按允许标准差归一化，再用指数变成(0,1]的姿态得分。
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
