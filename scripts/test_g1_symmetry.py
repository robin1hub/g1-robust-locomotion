"""Fast algebraic checks for the G1 sagittal reflection."""

import torch
from tensordict import TensorDict

from src.tasks.velocity.mdp.symmetry import (
  g1_lateral_symmetry,
  mirror_g1_actions,
  mirror_g1_actor_observation,
  mirror_g1_critic_observation,
)


def main() -> None:
  torch.manual_seed(42)
  actor = torch.randn(7, 397)
  actor[:, 392:397] = 0.0
  critic = torch.randn(7, 113)
  actions = torch.randn(7, 29)

  assert torch.equal(
    mirror_g1_actor_observation(mirror_g1_actor_observation(actor)), actor
  )
  assert torch.equal(
    mirror_g1_critic_observation(mirror_g1_critic_observation(critic)), critic
  )
  assert torch.equal(mirror_g1_actions(mirror_g1_actions(actions)), actions)

  obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[7])
  obs_aug, actions_aug = g1_lateral_symmetry(None, obs, actions)
  assert obs_aug is not None and actions_aug is not None
  assert obs_aug.batch_size == torch.Size([14])
  assert actions_aug.shape == (14, 29)
  assert torch.equal(obs_aug["actor"][:7], actor)
  assert torch.equal(obs_aug["actor"][7:], mirror_g1_actor_observation(actor))
  assert torch.equal(actions_aug[:7], actions)
  assert torch.equal(actions_aug[7:], mirror_g1_actions(actions))
  print("G1 symmetry involution and augmentation checks passed.")


if __name__ == "__main__":
  main()
