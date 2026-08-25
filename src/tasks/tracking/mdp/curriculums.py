"""Curricula for robust motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RobustnessStage(TypedDict):
  step: int
  friction: tuple[float, float]
  motor_strength: tuple[float, float]
  delay_lag: tuple[int, int]
  payload_kg: tuple[float, float]
  push_xy_mps: tuple[float, float]
  push_yaw_radps: tuple[float, float]


def joint_randomization_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  stages: list[RobustnessStage],
) -> dict[str, torch.Tensor]:
  """Expand joint dynamics randomization ranges according to training steps."""
  del env_ids
  active = stages[0]
  active_index = 0
  for index, stage in enumerate(stages):
    if env.common_step_counter >= stage["step"]:
      active = stage
      active_index = index

  env.event_manager.get_term_cfg("foot_friction").params["ranges"] = active[
    "friction"
  ]
  env.event_manager.get_term_cfg("motor_strength").params[
    "effort_limit_range"
  ] = active["motor_strength"]
  env.event_manager.get_term_cfg("action_delay").params["lag_range"] = active[
    "delay_lag"
  ]
  env.event_manager.get_term_cfg("torso_payload").params["ranges"] = active[
    "payload_kg"
  ]
  env.event_manager.get_term_cfg("push_robot").params["velocity_range"] = {
    "x": active["push_xy_mps"],
    "y": active["push_xy_mps"],
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": active["push_yaw_radps"],
  }

  return {
    "stage": torch.tensor(float(active_index)),
    "friction_min": torch.tensor(active["friction"][0]),
    "motor_min": torch.tensor(active["motor_strength"][0]),
    "delay_max_ms": torch.tensor(active["delay_lag"][1] * 5.0),
    "payload_max_kg": torch.tensor(active["payload_kg"][1]),
  }
