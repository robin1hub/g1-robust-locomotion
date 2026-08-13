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
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
  """Configuration for checkpoint evaluation."""

  checkpoint: str
  """Checkpoint path or glob, for example ``.../model_*.pt``."""

  num_envs: int = 64
  """Parallel environments, and therefore episodes, evaluated per seed."""

  seeds: tuple[int, ...] = (42,)
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
  "timed_out",
  "velocity_xy_rmse",
  "velocity_yaw_rmse",
  "foot_slip_mean_mps",
  "action_delta_rms",
  "base_tilt_rms",
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
    "velocity_yaw_rmse",
    "foot_slip_mean_mps",
    "action_delta_rms",
    "base_tilt_rms",
  )
  summary: dict[str, object] = {
    "checkpoint": str(checkpoint),
    "episodes": len(rows),
    "fall_rate": _mean([float(row["fell_over"]) for row in rows]),
    "timeout_rate": _mean([float(row["timed_out"]) for row in rows]),
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
  vel_yaw_sq = torch.zeros(count, device=base_env.device)
  action_delta_sq = torch.zeros(count, device=base_env.device)
  tilt_sq = torch.zeros(count, device=base_env.device)
  slip_sum = torch.zeros(count, device=base_env.device)
  slip_count = torch.zeros(count, device=base_env.device)
  previous_action: torch.Tensor | None = None
  rows: list[dict] = []

  obs = env.get_observations()
  # All initial episodes end by this point because the time limit is finite.
  max_steps = env.max_episode_length + 1
  for _ in range(max_steps):
    action = policy(obs)

    command = command_term.command
    lin_vel = robot.data.root_link_lin_vel_b
    ang_vel = robot.data.root_link_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    mask = active.float()

    vel_xy_sq += torch.sum(torch.square(command[:, :2] - lin_vel[:, :2]), dim=1) * mask
    vel_yaw_sq += torch.square(command[:, 2] - ang_vel[:, 2]) * mask
    tilt_sq += torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * mask

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
        rows.append(
          {
            "checkpoint": checkpoint.name,
            "seed": seed,
            "env_id": env_id,
            "return": float(returns[env_id].item()),
            "length_steps": steps,
            "duration_s": steps * base_env.step_dt,
            "fell_over": bool(base_env.reset_terminated[env_id].item()),
            "timed_out": bool(base_env.reset_time_outs[env_id].item()),
            "velocity_xy_rmse": math.sqrt(float(vel_xy_sq[env_id].item()) / steps),
            "velocity_yaw_rmse": math.sqrt(float(vel_yaw_sq[env_id].item()) / steps),
            "foot_slip_mean_mps": float(slip_sum[env_id].item())
            / max(float(slip_count[env_id].item()), 1.0),
            "action_delta_rms": math.sqrt(
              float(action_delta_sq[env_id].item()) / action_steps
            ),
            "base_tilt_rms": math.sqrt(float(tilt_sq[env_id].item()) / steps),
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
