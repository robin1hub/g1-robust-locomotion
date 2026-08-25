"""Left/right reflection used by the G1 velocity-policy PPO augmentation."""

from __future__ import annotations

import torch
from tensordict import TensorDict


# MuJoCo actuator order. Reflection across the sagittal plane swaps paired
# joints. Rotations about x/z change sign; rotations about y keep their sign.
_JOINT_PERM = (
  6, 7, 8, 9, 10, 11,
  0, 1, 2, 3, 4, 5,
  12, 13, 14,
  22, 23, 24, 25, 26, 27, 28,
  15, 16, 17, 18, 19, 20, 21,
)
_JOINT_SIGN = (
  1, -1, -1, 1, 1, -1,
  1, -1, -1, 1, 1, -1,
  -1, -1, 1,
  1, -1, -1, 1, -1, 1, -1,
  1, -1, -1, 1, -1, 1, -1,
)


def _mirror_joint_blocks(value: torch.Tensor) -> torch.Tensor:
  blocks = value.reshape(value.shape[0], -1, 29)
  perm = torch.tensor(_JOINT_PERM, device=value.device)
  sign = torch.tensor(_JOINT_SIGN, device=value.device, dtype=value.dtype)
  return (blocks[:, :, perm] * sign).reshape_as(value)


def _mirror_vector_blocks(
  value: torch.Tensor, signs: tuple[int, int, int]
) -> torch.Tensor:
  blocks = value.reshape(value.shape[0], -1, 3)
  sign = torch.tensor(signs, device=value.device, dtype=value.dtype)
  return (blocks * sign).reshape_as(value)


def _mirror_phase_blocks(value: torch.Tensor) -> torch.Tensor:
  # Swapping left/right legs is a half-cycle shift: sin/cos(phi+pi)=-sin/cos(phi).
  return -value


def mirror_g1_actor_observation(value: torch.Tensor) -> torch.Tensor:
  """Mirror the fixed 397-D E2 actor observation layout."""
  if value.shape[-1] != 397:
    raise ValueError(f"Expected 397 actor features, got {value.shape[-1]}")
  mirrored = value.clone()
  mirrored[:, 0:12] = _mirror_vector_blocks(value[:, 0:12], (-1, 1, -1))
  mirrored[:, 12:24] = _mirror_vector_blocks(value[:, 12:24], (1, -1, 1))
  mirrored[:, 24:36] = _mirror_vector_blocks(value[:, 24:36], (1, -1, -1))
  mirrored[:, 36:44] = _mirror_phase_blocks(value[:, 36:44])
  mirrored[:, 44:160] = _mirror_joint_blocks(value[:, 44:160])
  mirrored[:, 160:276] = _mirror_joint_blocks(value[:, 160:276])
  mirrored[:, 276:392] = _mirror_joint_blocks(value[:, 276:392])
  # 392:397 is the checkpoint-compatible zero_track_state slot.
  mirrored[:, 392:397] = value[:, 392:397]
  return mirrored


def mirror_g1_critic_observation(value: torch.Tensor) -> torch.Tensor:
  """Mirror the fixed 113-D E2 critic observation layout."""
  if value.shape[-1] != 113:
    raise ValueError(f"Expected 113 critic features, got {value.shape[-1]}")
  mirrored = value.clone()
  mirrored[:, 0:3] = _mirror_vector_blocks(value[:, 0:3], (-1, 1, -1))
  mirrored[:, 3:6] = _mirror_vector_blocks(value[:, 3:6], (1, -1, 1))
  mirrored[:, 6:9] = _mirror_vector_blocks(value[:, 6:9], (1, -1, -1))
  mirrored[:, 9:11] = _mirror_phase_blocks(value[:, 9:11])
  mirrored[:, 11:40] = _mirror_joint_blocks(value[:, 11:40])
  mirrored[:, 40:69] = _mirror_joint_blocks(value[:, 40:69])
  mirrored[:, 69:98] = _mirror_joint_blocks(value[:, 69:98])
  mirrored[:, 98:101] = _mirror_vector_blocks(value[:, 98:101], (1, -1, 1))
  mirrored[:, 101:103] = value[:, 101:103].flip(-1)
  mirrored[:, 103:105] = value[:, 103:105].flip(-1)
  mirrored[:, 105:107] = value[:, 105:107].flip(-1)
  forces = value[:, 107:113].reshape(value.shape[0], 2, 3).flip(1)
  force_sign = torch.tensor((1, -1, 1), device=value.device, dtype=value.dtype)
  mirrored[:, 107:113] = (forces * force_sign).reshape(value.shape[0], 6)
  return mirrored


def mirror_g1_actions(value: torch.Tensor) -> torch.Tensor:
  if value.shape[-1] != 29:
    raise ValueError(f"Expected 29 actions, got {value.shape[-1]}")
  return _mirror_joint_blocks(value)


def g1_lateral_symmetry(
  env,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Return original+mirrored batches in the format required by RSL-RL."""
  del env  # Layout is deliberately asserted above instead of inferred silently.
  obs_aug = None
  if obs is not None:
    mirrored_obs = obs.clone()
    if "actor" in mirrored_obs.keys():
      mirrored_obs["actor"] = mirror_g1_actor_observation(obs["actor"])
    if "critic" in mirrored_obs.keys():
      mirrored_obs["critic"] = mirror_g1_critic_observation(obs["critic"])
    unknown = set(mirrored_obs.keys()) - {"actor", "critic"}
    if unknown:
      raise KeyError(f"Unsupported observation sets for G1 symmetry: {unknown}")
    obs_aug = torch.cat((obs, mirrored_obs), dim=0)

  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat((actions, mirror_g1_actions(actions)), dim=0)
  return obs_aug, actions_aug
