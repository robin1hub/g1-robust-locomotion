"""Warm-start a short-history actor from a current-frame tracking policy."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


# Actor observation order and dimensions for the 29-DoF G1 tracking task.
# The final value is the number of frames used by T002 for that term.
ACTOR_LAYOUT = (
  ("command", 58, 1),
  ("motion_anchor_pos_b", 3, 1),
  ("motion_anchor_ori_b", 6, 1),
  ("base_lin_vel", 3, 4),
  ("base_ang_vel", 3, 4),
  ("joint_pos", 29, 4),
  ("joint_vel", 29, 4),
  ("actions", 29, 4),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  return parser.parse_args()


def expand_statistics(tensor: torch.Tensor) -> torch.Tensor:
  chunks = []
  cursor = 0
  for _, width, frames in ACTOR_LAYOUT:
    chunk = tensor[..., cursor : cursor + width]
    chunks.extend([chunk] * frames)
    cursor += width
  if cursor != tensor.shape[-1]:
    raise ValueError(f"Layout covers {cursor} features, checkpoint has {tensor.shape[-1]}")
  return torch.cat(chunks, dim=-1)


def expand_current_frame_columns(tensor: torch.Tensor) -> torch.Tensor:
  """Put the old policy weights on each term's newest frame.

  History buffers are ordered oldest to newest. Past-frame columns start at
  zero, making the transformed policy exactly reproduce T001 before T002
  learning begins while leaving useful trainable inputs for adaptation.
  """
  old_width = tensor.shape[-1]
  new_width = sum(width * frames for _, width, frames in ACTOR_LAYOUT)
  output = tensor.new_zeros((*tensor.shape[:-1], new_width))
  old_cursor = 0
  new_cursor = 0
  for _, width, frames in ACTOR_LAYOUT:
    newest_start = new_cursor + (frames - 1) * width
    output[..., newest_start : newest_start + width] = tensor[
      ..., old_cursor : old_cursor + width
    ]
    old_cursor += width
    new_cursor += width * frames
  if old_cursor != old_width:
    raise ValueError(f"Layout covers {old_cursor} features, tensor has {old_width}")
  return output


def main() -> None:
  args = parse_args()
  source = args.source.expanduser().resolve()
  output = args.output.expanduser().resolve()
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  transformed = deepcopy(checkpoint)
  actor = transformed["actor_state_dict"]

  for name in (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  ):
    actor[name] = expand_statistics(actor[name])
  actor["mlp.0.weight"] = expand_current_frame_columns(actor["mlp.0.weight"])

  # Preserve Adam state for every unchanged parameter. The first actor layer's
  # moments are expanded in the same way as its weights; new history columns
  # begin with zero momentum.
  optimizer = transformed["optimizer_state_dict"]
  expanded_optimizer_states = 0
  for state in optimizer["state"].values():
    exp_avg = state.get("exp_avg")
    if isinstance(exp_avg, torch.Tensor) and tuple(exp_avg.shape) == (512, 160):
      state["exp_avg"] = expand_current_frame_columns(exp_avg)
      state["exp_avg_sq"] = expand_current_frame_columns(state["exp_avg_sq"])
      expanded_optimizer_states += 1
  if expanded_optimizer_states != 1:
    raise ValueError(
      f"Expected one 512x160 optimizer state, found {expanded_optimizer_states}"
    )

  transformed.setdefault("infos", {})["history_warm_start"] = {
    "source": str(source),
    "old_actor_obs_dim": 160,
    "new_actor_obs_dim": actor["mlp.0.weight"].shape[1],
    "history_frames": 4,
    "history_terms": [name for name, _, frames in ACTOR_LAYOUT if frames > 1],
    "initial_policy_equivalent_to_source": True,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(transformed, output)
  print(f"Wrote history warm-start checkpoint to {output}")
  print(f"Actor input: 160 -> {actor['mlp.0.weight'].shape[1]}")


if __name__ == "__main__":
  main()
