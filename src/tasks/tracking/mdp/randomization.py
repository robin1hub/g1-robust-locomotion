"""Tracking-task domain randomization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import (
  BuiltinPositionActuator,
  DelayedActuator,
  IdealPdActuator,
  XmlPositionActuator,
)
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@requires_model_fields("actuator_forcerange")
def delayed_actuator_effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale effort limits for position actuators, including delayed wrappers."""
  asset = env.scene[asset_cfg.name]
  if env_ids is None or isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[index] for index in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]
  if not isinstance(actuators, list):
    actuators = [actuators]

  low, high = effort_limit_range
  default_forcerange = env.sim.get_default_field("actuator_forcerange")
  for actuator in actuators:
    if isinstance(actuator, DelayedActuator):
      actuator = actuator.base_actuator
    ctrl_ids = actuator.global_ctrl_ids
    samples = torch.empty(
      (len(env_ids), len(ctrl_ids)), device=env.device, dtype=torch.float32
    ).uniform_(low, high)
    if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
        default_forcerange[ctrl_ids, 0] * samples
      )
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
        default_forcerange[ctrl_ids, 1] * samples
      )
    elif isinstance(actuator, IdealPdActuator):
      if actuator.default_force_limit is None:
        raise RuntimeError("IdealPdActuator has no default effort limit")
      actuator.set_effort_limit(
        env_ids, actuator.default_force_limit[env_ids] * samples
      )
    else:
      raise TypeError(
        "delayed_actuator_effort_limits only supports position/PD actuators, "
        f"got {type(actuator).__name__}"
      )
