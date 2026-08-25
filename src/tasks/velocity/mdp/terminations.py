from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def outside_straight_lane(
  env: ManagerBasedRlEnv,
  lane_half_width: float = 0.9,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when the root leaves the world-frame straight running lane."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral_position = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return torch.abs(lateral_position) > lane_half_width


def running_backwards(
  env: ManagerBasedRlEnv,
  max_backward_distance: float = 0.5,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate a policy that moves behind the straight-track start line."""
  asset: Entity = env.scene[asset_cfg.name]
  forward_position = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return forward_position < -max_backward_distance
