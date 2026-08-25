"""Reproducible robustness benchmark for motion-tracking policies.

Each episode covers exactly one reference clip and terminates before the motion
command wraps. This avoids counting the command's clip-boundary state reset as
policy recovery.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro

import mjlab
from mjlab.actuator import DelayedActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
  checkpoint: str
  motion_file: str
  num_envs: int = 64
  seeds: tuple[int, ...] = (42, 43, 44)
  scenarios: tuple[str, ...] = (
    "clean",
    "friction_0p2",
    "friction_0p4",
    "friction_0p6",
    "friction_0p8",
    "push_0p25",
    "push_0p5",
    "push_0p75",
    "push_1p0",
  )
  push_time_s: float = 1.5
  patch_start_m: float = 2.25
  patch_length_m: float = 1.5
  patch_half_width_m: float = 1.0
  device: str | None = None
  output_dir: str | None = None


@dataclass(frozen=True)
class Scenario:
  name: str
  foot_friction: float
  lateral_velocity_push_mps: float = 0.0
  local_patch_friction: float | None = None
  mass_scale: float = 1.0
  torso_payload_kg: float = 0.0
  torso_com_offset_y_m: float = 0.0
  motor_strength_scale: float = 1.0
  action_delay_ms: float = 0.0


EPISODE_FIELDS = (
  "scenario",
  "seed",
  "env_id",
  "success",
  "termination_reason",
  "return",
  "length_steps",
  "duration_s",
  "root_position_rmse_m",
  "root_orientation_rmse_rad",
  "root_linear_velocity_rmse_mps",
  "root_angular_velocity_rmse_radps",
  "body_mpkpe_rmse_m",
  "body_orientation_rmse_rad",
  "joint_position_rmse_rad",
  "joint_velocity_rmse_radps",
  "foot_slip_mean_mps",
  "patch_exposure_s",
  "patch_foot_slip_mean_mps",
  "reached_post_patch",
  "post_patch_root_position_rmse_m",
  "action_delta_rms",
  "root_displacement_m",
)


def _parse_number(value: str) -> float:
  return float(value.replace("p", "."))


def _parse_scenario(name: str) -> Scenario:
  if name == "clean":
    return Scenario(name=name, foot_friction=1.0)
  combination_presets = {
    # E007 uses individually near-safe boundary values to expose interaction
    # effects without making every combined episode trivially impossible.
    "combo_patch_motor": Scenario(
      name=name, foot_friction=1.0, local_patch_friction=0.20, motor_strength_scale=0.58
    ),
    "combo_patch_delay": Scenario(
      name=name, foot_friction=1.0, local_patch_friction=0.20, action_delay_ms=35.0
    ),
    "combo_motor_delay": Scenario(
      name=name, foot_friction=1.0, motor_strength_scale=0.58, action_delay_ms=35.0
    ),
    "combo_actuation_payload": Scenario(
      name=name,
      foot_friction=1.0,
      torso_payload_kg=6.0,
      motor_strength_scale=0.58,
      action_delay_ms=35.0,
    ),
    "combo_all": Scenario(
      name=name,
      foot_friction=1.0,
      lateral_velocity_push_mps=0.25,
      local_patch_friction=0.20,
      torso_payload_kg=6.0,
      motor_strength_scale=0.58,
      action_delay_ms=35.0,
    ),
  }
  if name in combination_presets:
    return combination_presets[name]
  if name.startswith("friction_"):
    value = _parse_number(name.removeprefix("friction_"))
    if value <= 0:
      raise ValueError(f"Friction must be positive: {name}")
    return Scenario(name=name, foot_friction=value)
  if name.startswith("push_"):
    value = _parse_number(name.removeprefix("push_"))
    if value < 0:
      raise ValueError(f"Push magnitude must be non-negative: {name}")
    return Scenario(name=name, foot_friction=1.0, lateral_velocity_push_mps=value)
  if name.startswith("local_friction_"):
    value = _parse_number(name.removeprefix("local_friction_"))
    if value <= 0:
      raise ValueError(f"Local patch friction must be positive: {name}")
    return Scenario(name=name, foot_friction=1.0, local_patch_friction=value)
  if name.startswith("mass_scale_"):
    value = _parse_number(name.removeprefix("mass_scale_"))
    if value <= 0:
      raise ValueError(f"Mass scale must be positive: {name}")
    return Scenario(name=name, foot_friction=1.0, mass_scale=value)
  if name.startswith("payload_") and name.endswith("kg"):
    value = _parse_number(name.removeprefix("payload_").removesuffix("kg"))
    if value < 0:
      raise ValueError(f"Payload must be non-negative: {name}")
    return Scenario(name=name, foot_friction=1.0, torso_payload_kg=value)
  if name.startswith("com_y_pos_"):
    value = _parse_number(name.removeprefix("com_y_pos_"))
    if value <= 0:
      raise ValueError(f"Positive COM offset must be positive: {name}")
    return Scenario(name=name, foot_friction=1.0, torso_com_offset_y_m=value)
  if name.startswith("com_y_neg_"):
    value = _parse_number(name.removeprefix("com_y_neg_"))
    if value <= 0:
      raise ValueError(f"Negative COM offset magnitude must be positive: {name}")
    return Scenario(name=name, foot_friction=1.0, torso_com_offset_y_m=-value)
  if name.startswith("motor_scale_"):
    value = _parse_number(name.removeprefix("motor_scale_"))
    if value <= 0:
      raise ValueError(f"Motor strength scale must be positive: {name}")
    return Scenario(name=name, foot_friction=1.0, motor_strength_scale=value)
  if name.startswith("delay_") and name.endswith("ms"):
    value = _parse_number(name.removeprefix("delay_").removesuffix("ms"))
    if value < 0:
      raise ValueError(f"Action delay must be non-negative: {name}")
    return Scenario(name=name, foot_friction=1.0, action_delay_ms=value)
  raise ValueError(
    f"Unknown scenario {name!r}; use clean, friction_<value>, push_<value>, "
    "local_friction_<value>, mass_scale_<value>, payload_<kg>kg, or "
    "com_y_(pos|neg)_<meters>, motor_scale_<value>, delay_<milliseconds>ms, "
    "or a documented combo_* preset."
  )


def _mean(values: list[float]) -> float:
  return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
  return statistics.stdev(values) if len(values) > 1 else 0.0


def _termination_reason(env: ManagerBasedRlEnv, env_id: int) -> str:
  names = [
    name
    for name in env.termination_manager.active_terms
    if bool(env.termination_manager.get_term(name)[env_id].item())
  ]
  return "+".join(names) if names else "unknown"


def _add_foot_contact_sensor(env_cfg) -> None:
  sensor = ContactSensorCfg(
    name="benchmark_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (sensor,)


def _configure_scenario(
  task_id: str,
  cfg: EvalConfig,
  scenario: Scenario,
  seed: int,
) -> tuple[object, int]:
  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = cfg.num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = cfg.num_envs
  env_cfg.curriculum = {}

  # A robust-training task may wrap the base actuators for randomized delays.
  # The benchmark always starts from zero delay and then applies the requested
  # deterministic scenario, so unwrap any training-only delay configuration.
  robot_cfg = env_cfg.scene.entities["robot"]
  if robot_cfg.articulation is None:
    raise RuntimeError("Robot has no articulation configuration")
  robot_cfg.articulation = replace(
    robot_cfg.articulation,
    actuators=tuple(
      actuator_cfg.base_cfg
      if isinstance(actuator_cfg, DelayedActuatorCfg)
      else actuator_cfg
      for actuator_cfg in robot_cfg.articulation.actuators
    ),
  )

  motion_file = Path(cfg.motion_file).expanduser().resolve()
  with np.load(motion_file) as motion:
    motion_frames = int(motion["joint_pos"].shape[0])
  if motion_frames < 3:
    raise ValueError(f"Motion clip is too short: {motion_frames} frames")

  motion_cfg = cast(MotionCommandCfg, env_cfg.commands["motion"])
  motion_cfg.motion_file = str(motion_file)
  motion_cfg.sampling_mode = "start"

  # Stop before MotionCommand reaches its clip-boundary state reset. One frame
  # is consumed below to initialize its root-relative body targets.
  step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  episode_steps = motion_frames - 2
  env_cfg.episode_length_s = episode_steps * step_dt

  # Isolate the tested factor. Initial pose/velocity/joint noise and observation
  # corruption stay enabled to provide a distribution across envs and seeds.
  env_cfg.events.pop("base_com", None)
  env_cfg.events.pop("encoder_bias", None)
  env_cfg.events.pop("motor_strength", None)
  env_cfg.events.pop("action_delay", None)
  env_cfg.events.pop("torso_payload", None)
  friction_event = env_cfg.events["foot_friction"]
  friction_event.params["operation"] = "abs"
  friction_event.params["ranges"] = (
    scenario.foot_friction,
    scenario.foot_friction,
  )

  if scenario.mass_scale != 1.0:
    # pseudo_inertia scales mass and inertia together by exp(2 * alpha).
    alpha = 0.5 * math.log(scenario.mass_scale)
    env_cfg.events["mass_scale"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "alpha_range": (alpha, alpha),
      },
    )
  if scenario.torso_payload_kg > 0:
    # A point payload at the torso COM changes mass without adding rotational inertia.
    env_cfg.events["torso_payload"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": (scenario.torso_payload_kg, scenario.torso_payload_kg),
      },
    )
  if scenario.torso_com_offset_y_m != 0:
    offset = scenario.torso_com_offset_y_m
    env_cfg.events["torso_com_offset"] = EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (0.0, 0.0), 1: (offset, offset), 2: (0.0, 0.0)},
      },
    )
  if scenario.motor_strength_scale != 1.0:
    if scenario.action_delay_ms > 0:
      # dr.effort_limits intentionally rejects DelayedActuator. For combined
      # motor+delay scenarios, apply the same deterministic scale to each base
      # actuator config before wrapping it below.
      robot_cfg = env_cfg.scene.entities["robot"]
      if robot_cfg.articulation is None:
        raise RuntimeError("Robot has no articulation configuration")
      scaled_actuators = []
      for actuator_cfg in robot_cfg.articulation.actuators:
        effort_limit = getattr(actuator_cfg, "effort_limit", None)
        if effort_limit is None:
          raise TypeError(
            f"Cannot statically scale effort limit for {type(actuator_cfg).__name__}"
          )
        scaled_actuators.append(
          replace(
            actuator_cfg,
            effort_limit=effort_limit * scenario.motor_strength_scale,
          )
        )
      robot_cfg.articulation = replace(
        robot_cfg.articulation, actuators=tuple(scaled_actuators)
      )
    else:
      env_cfg.events["motor_strength"] = EventTermCfg(
        mode="startup",
        func=dr.effort_limits,
        params={
          # The DR helper indexes actuator groups, not the 29 individual controls.
          "asset_cfg": SceneEntityCfg("robot"),
          "operation": "scale",
          "effort_limit_range": (
            scenario.motor_strength_scale,
            scenario.motor_strength_scale,
          ),
        },
      )
  if scenario.action_delay_ms > 0:
    physics_dt = env_cfg.sim.mujoco.timestep
    delay_steps_float = scenario.action_delay_ms / (1000.0 * physics_dt)
    delay_steps = round(delay_steps_float)
    if not math.isclose(delay_steps_float, delay_steps, abs_tol=1e-6):
      raise ValueError(
        f"Delay {scenario.action_delay_ms} ms is not divisible by the "
        f"{physics_dt * 1000:g} ms physics step"
      )
    robot_cfg = env_cfg.scene.entities["robot"]
    if robot_cfg.articulation is None:
      raise RuntimeError("Robot has no articulation configuration")
    delayed_actuators = tuple(
      DelayedActuatorCfg(
        base_cfg=actuator_cfg,
        delay_target="position",
        delay_min_lag=delay_steps,
        delay_max_lag=delay_steps,
      )
      for actuator_cfg in robot_cfg.articulation.actuators
    )
    robot_cfg.articulation = replace(
      robot_cfg.articulation, actuators=delayed_actuators
    )

  if scenario.lateral_velocity_push_mps > 0:
    push = env_cfg.events["push_robot"]
    push.interval_range_s = (cfg.push_time_s, cfg.push_time_s)
    # Global-time interval events pass env_ids=None, which the built-in
    # push_by_setting_velocity does not handle correctly for vectorized envs.
    # A fixed per-env interval still triggers every environment at the same step.
    push.is_global_time = False
    push.params["velocity_range"] = {
      "x": (
        scenario.lateral_velocity_push_mps,
        scenario.lateral_velocity_push_mps,
      )
    }
  else:
    env_cfg.events.pop("push_robot", None)

  _add_foot_contact_sensor(env_cfg)
  return env_cfg, episode_steps


def _set_spatial_foot_friction(
  base_env: ManagerBasedRlEnv,
  robot,
  foot_site_pos_w: torch.Tensor,
  initial_root_pos_w: torch.Tensor,
  forward_direction_xy: torch.Tensor,
  cfg: EvalConfig,
  scenario: Scenario,
) -> torch.Tensor:
  """Apply patch friction independently to each foot and return its mask."""
  if scenario.local_patch_friction is None:
    return torch.zeros(
      (cfg.num_envs, 2), dtype=torch.bool, device=base_env.device
    )

  relative_xy = foot_site_pos_w[..., :2] - initial_root_pos_w[:, None, :2]
  forward_progress = torch.sum(relative_xy * forward_direction_xy, dim=-1)
  lateral_direction_xy = torch.stack(
    (-forward_direction_xy[1], forward_direction_xy[0])
  )
  lateral_offset = torch.sum(relative_xy * lateral_direction_xy, dim=-1)
  patch_end_m = cfg.patch_start_m + cfg.patch_length_m
  inside = (
    (forward_progress >= cfg.patch_start_m)
    & (forward_progress <= patch_end_m)
    & (torch.abs(lateral_offset) <= cfg.patch_half_width_m)
  )

  left_local_ids, _ = robot.find_geoms(r"^left_foot[1-7]_collision$")
  right_local_ids, _ = robot.find_geoms(r"^right_foot[1-7]_collision$")
  if not left_local_ids or not right_local_ids:
    raise RuntimeError("Could not resolve left/right foot collision geoms")
  left_geom_ids = robot.indexing.geom_ids[left_local_ids]
  right_geom_ids = robot.indexing.geom_ids[right_local_ids]
  all_geom_ids = torch.cat((left_geom_ids, right_geom_ids))
  env_grid = torch.arange(cfg.num_envs, device=base_env.device)[:, None]
  geom_grid = all_geom_ids[None, :]

  friction = torch.full(
    (cfg.num_envs, len(all_geom_ids)),
    scenario.foot_friction,
    dtype=base_env.sim.model.geom_friction.dtype,
    device=base_env.device,
  )
  friction[inside[:, 0], : len(left_geom_ids)] = scenario.local_patch_friction
  friction[inside[:, 1], len(left_geom_ids) :] = scenario.local_patch_friction
  base_env.sim.model.geom_friction[env_grid, geom_grid, 0] = friction
  return inside


@torch.inference_mode()
def _evaluate_seed(
  task_id: str,
  checkpoint: Path,
  cfg: EvalConfig,
  scenario: Scenario,
  seed: int,
) -> list[dict]:
  env_cfg, episode_steps = _configure_scenario(task_id, cfg, scenario, seed)
  agent_cfg = load_rl_cfg(task_id)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, base_env.command_manager.get_term("motion"))
  # MotionCommand's relative-body targets are zero until its first update. That
  # would make ee_body_pos terminate every initial episode at step 1. Advance
  # once before collecting metrics; episode_steps already excludes this frame.
  command._update_command()
  robot = base_env.scene["robot"]
  contact_sensor = base_env.scene["benchmark_feet_ground_contact"]
  foot_site_ids, _ = robot.find_sites(("left_foot", "right_foot"))

  count = cfg.num_envs
  active = torch.ones(count, dtype=torch.bool, device=base_env.device)
  returns = torch.zeros(count, device=base_env.device)
  lengths = torch.zeros(count, dtype=torch.long, device=base_env.device)
  accumulators = {
    name: torch.zeros(count, device=base_env.device)
    for name in (
      "root_position",
      "root_orientation",
      "root_linear_velocity",
      "root_angular_velocity",
      "body_position",
      "body_orientation",
      "joint_position",
      "joint_velocity",
      "action_delta",
    )
  }
  slip_sum = torch.zeros(count, device=base_env.device)
  slip_count = torch.zeros(count, device=base_env.device)
  patch_exposure_steps = torch.zeros(count, device=base_env.device)
  patch_slip_sum = torch.zeros(count, device=base_env.device)
  patch_slip_count = torch.zeros(count, device=base_env.device)
  post_patch_root_error_sum = torch.zeros(count, device=base_env.device)
  post_patch_steps = torch.zeros(count, device=base_env.device)
  previous_action: torch.Tensor | None = None
  initial_root_pos = command.robot_anchor_pos_w.clone()
  reference_displacement_xy = (
    command.motion.body_pos_w[-1, command.motion_anchor_body_index, :2]
    - command.motion.body_pos_w[0, command.motion_anchor_body_index, :2]
  )
  reference_distance = torch.linalg.vector_norm(reference_displacement_xy)
  if float(reference_distance.item()) < 1e-6:
    raise RuntimeError("Reference motion has no horizontal displacement")
  forward_direction_xy = reference_displacement_xy / reference_distance
  last_root_pos = initial_root_pos.clone()
  rows: list[dict] = []

  obs = env.get_observations()
  for _ in range(episode_steps + 1):
    action = policy(obs)
    mask = active.float()
    last_root_pos[:] = command.robot_anchor_pos_w

    accumulators["root_position"] += (
      torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=1)
      * mask
    )
    accumulators["root_orientation"] += (
      torch.square(
        quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
      )
      * mask
    )
    accumulators["root_linear_velocity"] += (
      torch.sum(
        torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w),
        dim=1,
      )
      * mask
    )
    accumulators["root_angular_velocity"] += (
      torch.sum(
        torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w),
        dim=1,
      )
      * mask
    )
    body_pos_error = torch.linalg.vector_norm(
      command.body_pos_relative_w - command.robot_body_pos_w, dim=-1
    ).mean(dim=1)
    body_ori_error = quat_error_magnitude(
      command.body_quat_relative_w, command.robot_body_quat_w
    ).mean(dim=1)
    accumulators["body_position"] += torch.square(body_pos_error) * mask
    accumulators["body_orientation"] += torch.square(body_ori_error) * mask
    accumulators["joint_position"] += (
      torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=1)
      * mask
    )
    accumulators["joint_velocity"] += (
      torch.mean(torch.square(command.joint_vel - command.robot_joint_vel), dim=1)
      * mask
    )

    if previous_action is not None:
      accumulators["action_delta"] += (
        torch.mean(torch.square(action - previous_action), dim=1) * mask
      )
    previous_action = action.clone()

    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0
    slip_speed = torch.linalg.vector_norm(
      robot.data.site_lin_vel_w[:, foot_site_ids, :2], dim=-1
    )
    inside_patch = _set_spatial_foot_friction(
      base_env,
      robot,
      robot.data.site_pos_w[:, foot_site_ids],
      initial_root_pos,
      forward_direction_xy,
      cfg,
      scenario,
    )
    slip_sum += torch.sum(slip_speed * in_contact, dim=1) * mask
    slip_count += torch.sum(in_contact, dim=1) * mask
    patch_exposure_steps += torch.any(inside_patch, dim=1) * mask
    patch_contact = in_contact & inside_patch
    patch_slip_sum += torch.sum(slip_speed * patch_contact, dim=1) * mask
    patch_slip_count += torch.sum(patch_contact, dim=1) * mask
    root_relative_xy = command.robot_anchor_pos_w[:, :2] - initial_root_pos[:, :2]
    root_forward_progress = torch.sum(
      root_relative_xy * forward_direction_xy, dim=-1
    )
    after_patch = root_forward_progress > cfg.patch_start_m + cfg.patch_length_m
    root_position_error_sq = torch.sum(
      torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=1
    )
    post_patch_root_error_sum += root_position_error_sq * after_patch * mask
    post_patch_steps += after_patch * mask

    obs, reward, dones, _ = env.step(action)
    returns += reward * mask
    lengths += active.long()

    finished = active & dones.bool()
    if torch.any(finished):
      for env_id in torch.nonzero(finished, as_tuple=False).squeeze(-1).tolist():
        steps = int(lengths[env_id].item())
        timed_out = bool(base_env.reset_time_outs[env_id].item())
        terminated = bool(base_env.reset_terminated[env_id].item())
        reached_post_patch = bool(post_patch_steps[env_id].item() > 0)
        rows.append(
          {
            "scenario": scenario.name,
            "seed": seed,
            "env_id": env_id,
            "success": timed_out and not terminated,
            "termination_reason": _termination_reason(base_env, env_id),
            "return": float(returns[env_id].item()),
            "length_steps": steps,
            "duration_s": steps * base_env.step_dt,
            "root_position_rmse_m": math.sqrt(
              float(accumulators["root_position"][env_id].item()) / steps
            ),
            "root_orientation_rmse_rad": math.sqrt(
              float(accumulators["root_orientation"][env_id].item()) / steps
            ),
            "root_linear_velocity_rmse_mps": math.sqrt(
              float(accumulators["root_linear_velocity"][env_id].item()) / steps
            ),
            "root_angular_velocity_rmse_radps": math.sqrt(
              float(accumulators["root_angular_velocity"][env_id].item()) / steps
            ),
            "body_mpkpe_rmse_m": math.sqrt(
              float(accumulators["body_position"][env_id].item()) / steps
            ),
            "body_orientation_rmse_rad": math.sqrt(
              float(accumulators["body_orientation"][env_id].item()) / steps
            ),
            "joint_position_rmse_rad": math.sqrt(
              float(accumulators["joint_position"][env_id].item()) / steps
            ),
            "joint_velocity_rmse_radps": math.sqrt(
              float(accumulators["joint_velocity"][env_id].item()) / steps
            ),
            "foot_slip_mean_mps": float(slip_sum[env_id].item())
            / max(float(slip_count[env_id].item()), 1.0),
            "patch_exposure_s": float(patch_exposure_steps[env_id].item())
            * base_env.step_dt,
            "patch_foot_slip_mean_mps": float(patch_slip_sum[env_id].item())
            / max(float(patch_slip_count[env_id].item()), 1.0),
            "reached_post_patch": reached_post_patch,
            "post_patch_root_position_rmse_m": (
              math.sqrt(
                float(post_patch_root_error_sum[env_id].item())
                / float(post_patch_steps[env_id].item())
              )
              if reached_post_patch
              else math.nan
            ),
            "action_delta_rms": math.sqrt(
              float(accumulators["action_delta"][env_id].item())
              / max(steps - 1, 1)
            ),
            "root_displacement_m": float(
              torch.linalg.vector_norm(
                last_root_pos[env_id] - initial_root_pos[env_id]
              ).item()
            ),
          }
        )
      active[finished] = False

    if not torch.any(active):
      break

  env.close()
  if len(rows) != count:
    raise RuntimeError(
      f"Only {len(rows)}/{count} episodes completed for {scenario.name}, seed={seed}."
    )
  return rows


def _summarize(scenario: Scenario, rows: list[dict]) -> dict:
  summary: dict[str, object] = {
    "scenario": scenario.name,
    "foot_friction": scenario.foot_friction,
    "lateral_velocity_push_mps": scenario.lateral_velocity_push_mps,
    "local_patch_friction": scenario.local_patch_friction,
    "mass_scale": scenario.mass_scale,
    "torso_payload_kg": scenario.torso_payload_kg,
    "torso_com_offset_y_m": scenario.torso_com_offset_y_m,
    "motor_strength_scale": scenario.motor_strength_scale,
    "action_delay_ms": scenario.action_delay_ms,
    "episodes": len(rows),
    "success_rate": _mean([float(row["success"]) for row in rows]),
    "post_patch_reach_rate": _mean(
      [float(row["reached_post_patch"]) for row in rows]
    ),
  }
  for field in EPISODE_FIELDS:
    if field in {
      "scenario",
      "seed",
      "env_id",
      "success",
      "termination_reason",
      "reached_post_patch",
    }:
      continue
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    summary[f"{field}_mean"] = _mean(values)
    summary[f"{field}_std"] = _std(values)
  return summary


def run_evaluation(task_id: str, cfg: EvalConfig) -> Path:
  configure_torch_backends()
  checkpoint = Path(cfg.checkpoint).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  motion_file = Path(cfg.motion_file).expanduser().resolve()
  if not motion_file.is_file():
    raise FileNotFoundError(motion_file)
  if cfg.num_envs <= 0 or not cfg.seeds or not cfg.scenarios:
    raise ValueError("num_envs, seeds, and scenarios must be non-empty")
  if cfg.patch_start_m < 0 or cfg.patch_length_m <= 0 or cfg.patch_half_width_m <= 0:
    raise ValueError("Local patch start/length/half-width must define a valid region")

  scenarios = [_parse_scenario(name) for name in cfg.scenarios]
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
  output_dir = Path(
    cfg.output_dir or f"evaluations/{task_id}/robustness_{timestamp}"
  ).resolve()
  output_dir.mkdir(parents=True, exist_ok=False)

  all_rows: list[dict] = []
  summaries: list[dict] = []
  for scenario in scenarios:
    scenario_rows: list[dict] = []
    for seed in cfg.seeds:
      print(
        f"[INFO] scenario={scenario.name}, seed={seed}, envs={cfg.num_envs}",
        flush=True,
      )
      scenario_rows.extend(
        _evaluate_seed(task_id, checkpoint, cfg, scenario, seed)
      )
    all_rows.extend(scenario_rows)
    summary = _summarize(scenario, scenario_rows)
    summaries.append(summary)
    print(
      f"[RESULT] {scenario.name}: success={summary['success_rate']:.3f}, "
      f"MPKPE={summary['body_mpkpe_rmse_m_mean']:.3f} m, "
      f"slip={summary['foot_slip_mean_mps_mean']:.3f} m/s",
      flush=True,
    )

  with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
    writer.writeheader()
    writer.writerows(all_rows)
  with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)

  metadata = {
    "task_id": task_id,
    "checkpoint": str(checkpoint),
    "motion_file": str(motion_file),
    "num_envs_per_seed": cfg.num_envs,
    "seeds": list(cfg.seeds),
    "scenarios": [asdict(scenario) for scenario in scenarios],
    "push_time_s": cfg.push_time_s,
    "local_patch": {
      "start_m": cfg.patch_start_m,
      "length_m": cfg.patch_length_m,
      "half_width_m": cfg.patch_half_width_m,
      "applied_per_foot": True,
    },
    "clip_boundary_reset_excluded": True,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "summaries": summaries,
  }
  (output_dir / "results.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  return output_dir


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    EvalConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  output_dir = run_evaluation(chosen_task, cfg)
  print(f"[INFO] Results written to: {output_dir}")


if __name__ == "__main__":
  main()
