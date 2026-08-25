"""Headless, reproducible evaluation for trained velocity policies.

The play script is intended for visual inspection and uses play-time overrides.
This script instead keeps the training-time randomization and finite episode
length, disables curriculum updates, and reports one complete episode per
parallel environment.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
  """Configuration for checkpoint evaluation."""

  checkpoint: str
  """Checkpoint path or glob, for example ``.../model_*.pt``."""

  num_envs: int = 64
  """Parallel environments, and therefore episodes, evaluated per seed."""

  seeds: tuple[int, ...] = (42,)
  command_speed_mps: float | None = None
  """If set, evaluate a deterministic straight-line forward speed command."""
  command_lateral_speed_mps: float = 0.0
  """Body-frame lateral command used with ``command_speed_mps``."""
  command_yaw_rate_radps: float = 0.0
  """Body-frame yaw-rate command used with ``command_speed_mps``."""
  yaw_filter_tau_s: float = 0.3
  """Time constant of the low-pass yaw-rate metric used for gait filtering."""
  clean: bool = False
  """Disable observation/dynamics randomization for a nominal clean evaluation."""
  device: str | None = None
  output_dir: str | None = None


EPISODE_FIELDS = (
  "checkpoint",
  "seed",
  "env_id",
  "return",
  "length_steps",
  "duration_s",
  "fell_over",
  "illegal_contact",
  "outside_lane",
  "running_backwards",
  "timed_out",
  "velocity_xy_rmse",
  "velocity_x_rmse",
  "velocity_y_rmse",
  "velocity_yaw_rmse",
  "yaw_filtered_rmse",
  "lateral_direction_correct_fraction",
  "yaw_direction_correct_fraction",
  "yaw_filtered_direction_correct_fraction",
  "foot_slip_mean_mps",
  "action_delta_rms",
  "base_tilt_rms",
  "world_forward_speed_mean_mps",
  "body_forward_speed_mean_mps",
  "body_lateral_speed_mean_mps",
  "body_yaw_rate_mean_radps",
  "yaw_filtered_mean_radps",
  "heading_yaw_rate_mean_radps",
  "yaw_steady_state_gain",
  "yaw_response_reached",
  "yaw_response_time_s",
  "yaw_overshoot_ratio",
  "lateral_position_rmse_m",
  "heading_alignment_mean",
  "heading_error_rms_deg",
  "heading_error_abs_mean_deg",
  "world_yaw_rate_rmse_radps",
  "forward_displacement_m",
  "terminal_lateral_position_m",
  "terminal_lateral_velocity_mps",
  "terminal_heading_error_deg",
  "lateral_drift_slope_dy_dx",
)


def _resolve_checkpoints(pattern: str) -> list[Path]:
  matches = [Path(path).resolve() for path in glob.glob(pattern)]
  if not matches and Path(pattern).is_file():
    matches = [Path(pattern).resolve()]
  matches = sorted(set(matches), key=lambda path: path.name)
  if not matches:
    raise FileNotFoundError(f"No checkpoint matched: {pattern}")
  return matches


def _mean(values: list[float]) -> float:
  return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
  return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize(checkpoint: Path, rows: list[dict]) -> dict:
  numeric_fields = (
    "return",
    "length_steps",
    "duration_s",
    "velocity_xy_rmse",
    "velocity_x_rmse",
    "velocity_y_rmse",
    "velocity_yaw_rmse",
    "yaw_filtered_rmse",
    "lateral_direction_correct_fraction",
    "yaw_direction_correct_fraction",
    "yaw_filtered_direction_correct_fraction",
    "foot_slip_mean_mps",
    "action_delta_rms",
    "base_tilt_rms",
    "world_forward_speed_mean_mps",
    "body_forward_speed_mean_mps",
    "body_lateral_speed_mean_mps",
    "body_yaw_rate_mean_radps",
    "yaw_filtered_mean_radps",
    "heading_yaw_rate_mean_radps",
    "yaw_steady_state_gain",
    "yaw_response_reached",
    "yaw_response_time_s",
    "yaw_overshoot_ratio",
    "lateral_position_rmse_m",
    "heading_alignment_mean",
    "heading_error_rms_deg",
    "heading_error_abs_mean_deg",
    "world_yaw_rate_rmse_radps",
    "forward_displacement_m",
    "terminal_lateral_position_m",
    "terminal_lateral_velocity_mps",
    "terminal_heading_error_deg",
    "lateral_drift_slope_dy_dx",
  )
  summary: dict[str, object] = {
    "checkpoint": str(checkpoint),
    "episodes": len(rows),
    "fall_rate": _mean([float(row["fell_over"]) for row in rows]),
    "illegal_contact_rate": _mean(
      [float(row["illegal_contact"]) for row in rows]
    ),
    "outside_lane_rate": _mean([float(row["outside_lane"]) for row in rows]),
    "running_backwards_rate": _mean(
      [float(row["running_backwards"]) for row in rows]
    ),
    "timeout_rate": _mean([float(row["timed_out"]) for row in rows]),
    "outside_positive_y_rate": _mean(
      [
        float(row["outside_lane"] and row["terminal_lateral_position_m"] > 0.0)
        for row in rows
      ]
    ),
    "outside_negative_y_rate": _mean(
      [
        float(row["outside_lane"] and row["terminal_lateral_position_m"] < 0.0)
        for row in rows
      ]
    ),
  }
  for field in numeric_fields:
    values = [float(row[field]) for row in rows]
    summary[f"{field}_mean"] = _mean(values)
    summary[f"{field}_std"] = _std(values)
  return summary


@torch.inference_mode()
def _evaluate_seed(
  task_id: str,
  checkpoint: Path,
  cfg: EvalConfig,
  seed: int,
) -> list[dict]:
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = cfg.num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = cfg.num_envs

  # Evaluation must not change difficulty or command ranges while it is running.
  # Startup/interval events remain enabled so dynamics randomization and pushes
  # are evaluated under the same conditions as training.
  env_cfg.curriculum = {}
  if cfg.clean:
    env_cfg.observations["actor"].enable_corruption = False
    for event_name in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
      env_cfg.events.pop(event_name, None)
  if cfg.command_speed_mps is not None:
    if cfg.command_speed_mps < 0:
      raise ValueError("command_speed_mps must be non-negative")
    command_cfg = env_cfg.commands["twist"]
    if not hasattr(command_cfg, "ranges"):
      raise TypeError("Expected a velocity command config with ranges for 'twist'")
    command_cfg.ranges.lin_vel_x = (cfg.command_speed_mps, cfg.command_speed_mps)
    command_cfg.ranges.lin_vel_y = (
      cfg.command_lateral_speed_mps,
      cfg.command_lateral_speed_mps,
    )
    command_cfg.ranges.ang_vel_z = (
      cfg.command_yaw_rate_radps,
      cfg.command_yaw_rate_radps,
    )
    if hasattr(command_cfg, "rel_combined_envs"):
      # Fixed-command evaluation must bypass categorical masking; otherwise
      # straight/lateral/turn categories would zero requested command axes.
      command_cfg.rel_straight_envs = 0.0
      command_cfg.rel_lateral_envs = 0.0
      command_cfg.rel_turn_envs = 0.0
      command_cfg.rel_combined_envs = 1.0
    elif hasattr(command_cfg, "rel_straight_envs"):
      command_cfg.rel_straight_envs = 0.0
    command_cfg.ranges.heading = None

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  robot = base_env.scene["robot"]
  foot_site_ids, _ = robot.find_sites(("left_foot", "right_foot"))
  contact_sensor = base_env.scene["feet_ground_contact"]
  command_term = base_env.command_manager.get_term("twist")
  if command_term is None:
    raise RuntimeError("Velocity evaluation requires the 'twist' command term.")

  count = cfg.num_envs
  active = torch.ones(count, dtype=torch.bool, device=base_env.device)
  returns = torch.zeros(count, device=base_env.device)
  lengths = torch.zeros(count, dtype=torch.long, device=base_env.device)
  vel_xy_sq = torch.zeros(count, device=base_env.device)
  vel_x_sq = torch.zeros(count, device=base_env.device)
  vel_y_sq = torch.zeros(count, device=base_env.device)
  vel_yaw_sq = torch.zeros(count, device=base_env.device)
  yaw_filtered_sq = torch.zeros(count, device=base_env.device)
  lateral_direction_correct = torch.zeros(count, device=base_env.device)
  lateral_direction_count = torch.zeros(count, device=base_env.device)
  yaw_direction_correct = torch.zeros(count, device=base_env.device)
  yaw_direction_count = torch.zeros(count, device=base_env.device)
  yaw_filtered_direction_correct = torch.zeros(count, device=base_env.device)
  yaw_filtered_direction_count = torch.zeros(count, device=base_env.device)
  action_delta_sq = torch.zeros(count, device=base_env.device)
  tilt_sq = torch.zeros(count, device=base_env.device)
  slip_sum = torch.zeros(count, device=base_env.device)
  slip_count = torch.zeros(count, device=base_env.device)
  world_forward_speed_sum = torch.zeros(count, device=base_env.device)
  body_forward_speed_sum = torch.zeros(count, device=base_env.device)
  body_lateral_speed_sum = torch.zeros(count, device=base_env.device)
  body_yaw_rate_sum = torch.zeros(count, device=base_env.device)
  yaw_filtered_sum = torch.zeros(count, device=base_env.device)
  heading_delta_sum = torch.zeros(count, device=base_env.device)
  yaw_response_step = torch.full(
    (count,), -1, dtype=torch.long, device=base_env.device
  )
  yaw_peak_along_command = torch.zeros(count, device=base_env.device)
  lateral_position_sq = torch.zeros(count, device=base_env.device)
  heading_alignment_sum = torch.zeros(count, device=base_env.device)
  heading_error_sq = torch.zeros(count, device=base_env.device)
  heading_error_abs = torch.zeros(count, device=base_env.device)
  world_yaw_rate_sq = torch.zeros(count, device=base_env.device)
  previous_action: torch.Tensor | None = None
  rows: list[dict] = []

  obs = env.get_observations()
  if cfg.yaw_filter_tau_s <= 0.0:
    raise ValueError("yaw_filter_tau_s must be positive")
  yaw_filter_alpha = base_env.step_dt / (cfg.yaw_filter_tau_s + base_env.step_dt)
  filtered_yaw_rate = robot.data.root_link_ang_vel_b[:, 2].clone()
  initial_body_forward = torch.zeros_like(robot.data.root_link_lin_vel_w)
  initial_body_forward[:, 0] = 1.0
  initial_forward_axis_w = quat_apply(
    robot.data.root_link_quat_w, initial_body_forward
  )
  previous_heading = torch.atan2(
    initial_forward_axis_w[:, 1], initial_forward_axis_w[:, 0]
  )
  # All initial episodes end by this point because the time limit is finite.
  max_steps = env.max_episode_length + 1
  for _ in range(max_steps):
    action = policy(obs)

    command = command_term.command
    lin_vel = robot.data.root_link_lin_vel_b
    ang_vel = robot.data.root_link_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    world_lin_vel = robot.data.root_link_lin_vel_w
    world_ang_vel = robot.data.root_link_ang_vel_w
    relative_root_pos = robot.data.root_link_pos_w - base_env.scene.env_origins
    body_forward = torch.zeros_like(world_lin_vel)
    body_forward[:, 0] = 1.0
    forward_axis_w = quat_apply(robot.data.root_link_quat_w, body_forward)
    heading_error = torch.atan2(forward_axis_w[:, 1], forward_axis_w[:, 0])
    heading_delta = torch.atan2(
      torch.sin(heading_error - previous_heading),
      torch.cos(heading_error - previous_heading),
    )
    previous_heading = heading_error.clone()
    filtered_yaw_rate += yaw_filter_alpha * (ang_vel[:, 2] - filtered_yaw_rate)
    mask = active.float()

    vel_xy_sq += torch.sum(torch.square(command[:, :2] - lin_vel[:, :2]), dim=1) * mask
    vel_x_sq += torch.square(command[:, 0] - lin_vel[:, 0]) * mask
    vel_y_sq += torch.square(command[:, 1] - lin_vel[:, 1]) * mask
    vel_yaw_sq += torch.square(command[:, 2] - ang_vel[:, 2]) * mask
    yaw_filtered_sq += torch.square(command[:, 2] - filtered_yaw_rate) * mask
    lateral_active = (torch.abs(command[:, 1]) > 1.0e-4).float() * mask
    lateral_direction_correct += (
      (command[:, 1] * lin_vel[:, 1] > 0.0).float() * lateral_active
    )
    lateral_direction_count += lateral_active
    yaw_active = (torch.abs(command[:, 2]) > 1.0e-4).float() * mask
    yaw_direction_correct += (
      (command[:, 2] * ang_vel[:, 2] > 0.0).float() * yaw_active
    )
    yaw_direction_count += yaw_active
    yaw_filtered_direction_correct += (
      (command[:, 2] * filtered_yaw_rate > 0.0).float() * yaw_active
    )
    yaw_filtered_direction_count += yaw_active
    command_yaw_abs = torch.abs(command[:, 2])
    command_yaw_sign = torch.sign(command[:, 2])
    filtered_along_command = filtered_yaw_rate * command_yaw_sign
    yaw_peak_along_command = torch.maximum(
      yaw_peak_along_command, filtered_along_command * yaw_active
    )
    response_now = (
      active
      & (command_yaw_abs > 1.0e-4)
      & (yaw_response_step < 0)
      & (filtered_along_command >= 0.8 * command_yaw_abs)
    )
    yaw_response_step[response_now] = lengths[response_now]
    tilt_sq += torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * mask
    world_forward_speed_sum += world_lin_vel[:, 0] * mask
    body_forward_speed_sum += lin_vel[:, 0] * mask
    body_lateral_speed_sum += lin_vel[:, 1] * mask
    body_yaw_rate_sum += ang_vel[:, 2] * mask
    yaw_filtered_sum += filtered_yaw_rate * mask
    heading_delta_sum += heading_delta * mask
    lateral_position_sq += torch.square(relative_root_pos[:, 1]) * mask
    heading_alignment_sum += forward_axis_w[:, 0] * mask
    heading_error_sq += torch.square(heading_error) * mask
    heading_error_abs += torch.abs(heading_error) * mask
    world_yaw_rate_sq += torch.square(world_ang_vel[:, 2]) * mask

    if previous_action is not None:
      action_delta_sq += torch.mean(torch.square(action - previous_action), dim=1) * mask
    previous_action = action.clone()

    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0
    foot_vel_xy = robot.data.site_lin_vel_w[:, foot_site_ids, :2]
    slip_speed = torch.linalg.vector_norm(foot_vel_xy, dim=-1)
    slip_sum += torch.sum(slip_speed * in_contact, dim=1) * mask
    slip_count += torch.sum(in_contact, dim=1) * mask

    obs, reward, dones, _ = env.step(action)
    returns += reward * mask
    lengths += active.long()

    finished = active & dones.bool()
    if torch.any(finished):
      finished_ids = torch.nonzero(finished, as_tuple=False).squeeze(-1)
      for env_id in finished_ids.tolist():
        steps = int(lengths[env_id].item())
        action_steps = max(steps - 1, 1)
        active_termination_terms = set(base_env.termination_manager.active_terms)

        def term_active(name: str) -> bool:
          return name in active_termination_terms and bool(
            base_env.termination_manager.get_term(name)[env_id].item()
          )

        forward_displacement = (
          float(world_forward_speed_sum[env_id].item()) * base_env.step_dt
        )
        terminal_lateral_position = float(relative_root_pos[env_id, 1].item())
        terminal_heading_error = float(heading_error[env_id].item())
        command_yaw = float(command[env_id, 2].item())
        yaw_active_episode = abs(command_yaw) > 1.0e-4
        response_step = int(yaw_response_step[env_id].item())
        response_reached = yaw_active_episode and response_step >= 0
        filtered_yaw_mean = float(yaw_filtered_sum[env_id].item()) / steps
        rows.append(
          {
            "checkpoint": checkpoint.name,
            "seed": seed,
            "env_id": env_id,
            "return": float(returns[env_id].item()),
            "length_steps": steps,
            "duration_s": steps * base_env.step_dt,
            "fell_over": term_active("fell_over"),
            "illegal_contact": term_active("illegal_contact"),
            "outside_lane": term_active("outside_lane"),
            "running_backwards": term_active("running_backwards"),
            "timed_out": bool(base_env.reset_time_outs[env_id].item()),
            "velocity_xy_rmse": math.sqrt(float(vel_xy_sq[env_id].item()) / steps),
            "velocity_x_rmse": math.sqrt(float(vel_x_sq[env_id].item()) / steps),
            "velocity_y_rmse": math.sqrt(float(vel_y_sq[env_id].item()) / steps),
            "velocity_yaw_rmse": math.sqrt(float(vel_yaw_sq[env_id].item()) / steps),
            "yaw_filtered_rmse": math.sqrt(
              float(yaw_filtered_sq[env_id].item()) / steps
            ),
            "lateral_direction_correct_fraction": float(
              lateral_direction_correct[env_id].item()
            )
            / max(float(lateral_direction_count[env_id].item()), 1.0),
            "yaw_direction_correct_fraction": float(
              yaw_direction_correct[env_id].item()
            )
            / max(float(yaw_direction_count[env_id].item()), 1.0),
            "yaw_filtered_direction_correct_fraction": float(
              yaw_filtered_direction_correct[env_id].item()
            )
            / max(float(yaw_filtered_direction_count[env_id].item()), 1.0),
            "foot_slip_mean_mps": float(slip_sum[env_id].item())
            / max(float(slip_count[env_id].item()), 1.0),
            "action_delta_rms": math.sqrt(
              float(action_delta_sq[env_id].item()) / action_steps
            ),
            "base_tilt_rms": math.sqrt(float(tilt_sq[env_id].item()) / steps),
            "world_forward_speed_mean_mps": float(
              world_forward_speed_sum[env_id].item()
            )
            / steps,
            "body_forward_speed_mean_mps": float(
              body_forward_speed_sum[env_id].item()
            )
            / steps,
            "body_lateral_speed_mean_mps": float(
              body_lateral_speed_sum[env_id].item()
            )
            / steps,
            "body_yaw_rate_mean_radps": float(body_yaw_rate_sum[env_id].item())
            / steps,
            "yaw_filtered_mean_radps": filtered_yaw_mean,
            "heading_yaw_rate_mean_radps": float(
              heading_delta_sum[env_id].item()
            )
            / (steps * base_env.step_dt),
            "yaw_steady_state_gain": filtered_yaw_mean / command_yaw
            if yaw_active_episode
            else 0.0,
            "yaw_response_reached": response_reached,
            "yaw_response_time_s": response_step * base_env.step_dt
            if response_reached
            else steps * base_env.step_dt,
            "yaw_overshoot_ratio": float(
              yaw_peak_along_command[env_id].item()
            )
            / abs(command_yaw)
            if yaw_active_episode
            else 0.0,
            "lateral_position_rmse_m": math.sqrt(
              float(lateral_position_sq[env_id].item()) / steps
            ),
            "heading_alignment_mean": float(
              heading_alignment_sum[env_id].item()
            )
            / steps,
            "heading_error_rms_deg": math.degrees(
              math.sqrt(float(heading_error_sq[env_id].item()) / steps)
            ),
            "heading_error_abs_mean_deg": math.degrees(
              float(heading_error_abs[env_id].item()) / steps
            ),
            "world_yaw_rate_rmse_radps": math.sqrt(
              float(world_yaw_rate_sq[env_id].item()) / steps
            ),
            "forward_displacement_m": forward_displacement,
            "terminal_lateral_position_m": terminal_lateral_position,
            "terminal_lateral_velocity_mps": float(world_lin_vel[env_id, 1].item()),
            "terminal_heading_error_deg": math.degrees(terminal_heading_error),
            "lateral_drift_slope_dy_dx": terminal_lateral_position
            / max(abs(forward_displacement), 0.1),
          }
        )
      active[finished] = False

    if not torch.any(active):
      break

  env.close()
  if len(rows) != count:
    raise RuntimeError(f"Only {len(rows)}/{count} evaluation episodes completed.")
  return rows


def run_evaluation(task_id: str, cfg: EvalConfig) -> Path:
  configure_torch_backends()
  checkpoints = _resolve_checkpoints(cfg.checkpoint)
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
  output_dir = Path(cfg.output_dir or f"evaluations/{task_id}/{timestamp}").resolve()
  output_dir.mkdir(parents=True, exist_ok=False)

  all_rows: list[dict] = []
  summaries: list[dict] = []
  for checkpoint in checkpoints:
    checkpoint_rows: list[dict] = []
    for seed in cfg.seeds:
      print(f"[INFO] Evaluating {checkpoint.name}, seed={seed}, envs={cfg.num_envs}")
      checkpoint_rows.extend(_evaluate_seed(task_id, checkpoint, cfg, seed))
    all_rows.extend(checkpoint_rows)
    summary = _summarize(checkpoint, checkpoint_rows)
    summaries.append(summary)
    print(
      "[RESULT] "
      f"{checkpoint.name}: return={summary['return_mean']:.3f}, "
      f"fall_rate={summary['fall_rate']:.3f}, "
      f"vel_xy_rmse={summary['velocity_xy_rmse_mean']:.3f}, "
      f"slip={summary['foot_slip_mean_mps_mean']:.3f} m/s"
    )

  with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
    writer.writeheader()
    writer.writerows(all_rows)

  summary_fields = list(summaries[0])
  with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summaries)

  metadata = {
    "task_id": task_id,
    "checkpoints": [str(path) for path in checkpoints],
    "num_envs_per_seed": cfg.num_envs,
    "seeds": list(cfg.seeds),
    "command_speed_mps": cfg.command_speed_mps,
    "command_lateral_speed_mps": cfg.command_lateral_speed_mps,
    "command_yaw_rate_radps": cfg.command_yaw_rate_radps,
    "yaw_filter_tau_s": cfg.yaw_filter_tau_s,
    "clean": cfg.clean,
    "device": cfg.device,
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

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
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
