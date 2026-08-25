"""Append zero-initialized straight-track observations to a Sprint actor."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


OLD_DIM = 392
TRACK_DIM = 5


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  return parser.parse_args()


def append_values(tensor: torch.Tensor, value: float) -> torch.Tensor:
  padding = tensor.new_full((*tensor.shape[:-1], TRACK_DIM), value)
  return torch.cat((tensor, padding), dim=-1)


def main() -> None:
  args = parse_args()
  source = args.source.expanduser().resolve()
  output = args.output.expanduser().resolve()
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  transformed = deepcopy(checkpoint)
  actor = transformed["actor_state_dict"]

  weight = actor["mlp.0.weight"]
  if tuple(weight.shape) != (512, OLD_DIM):
    raise ValueError(f"Expected actor first layer (512, {OLD_DIM}), got {weight.shape}")
  actor["mlp.0.weight"] = append_values(weight, 0.0)
  actor["obs_normalizer._mean"] = append_values(
    actor["obs_normalizer._mean"], 0.0
  )
  actor["obs_normalizer._var"] = append_values(actor["obs_normalizer._var"], 1.0)
  actor["obs_normalizer._std"] = append_values(actor["obs_normalizer._std"], 1.0)

  expanded_optimizer_states = 0
  for state in transformed["optimizer_state_dict"]["state"].values():
    exp_avg = state.get("exp_avg")
    if isinstance(exp_avg, torch.Tensor) and tuple(exp_avg.shape) == (512, OLD_DIM):
      state["exp_avg"] = append_values(exp_avg, 0.0)
      state["exp_avg_sq"] = append_values(state["exp_avg_sq"], 0.0)
      expanded_optimizer_states += 1
  if expanded_optimizer_states != 1:
    raise ValueError(
      f"Expected one 512x{OLD_DIM} optimizer state, found {expanded_optimizer_states}"
    )

  transformed.setdefault("infos", {})["track_state_warm_start"] = {
    "source": str(source),
    "old_actor_obs_dim": OLD_DIM,
    "new_actor_obs_dim": OLD_DIM + TRACK_DIM,
    "track_features": ("lateral_position", "heading_cos", "heading_sin", "vx", "vy"),
    "initial_policy_equivalent_to_source": True,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(transformed, output)
  print(f"Wrote track-state warm-start checkpoint to {output}")
  print(f"Actor input: {OLD_DIM} -> {OLD_DIM + TRACK_DIM}")


if __name__ == "__main__":
  main()
