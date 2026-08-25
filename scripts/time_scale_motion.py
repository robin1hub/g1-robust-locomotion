"""Time-scale an MjLab motion NPZ while preserving its spatial trajectory."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class Args:
  input_file: Path
  output_file: Path
  speed_scale: float


def _linear_sample(values: np.ndarray, source_frame: np.ndarray) -> np.ndarray:
  index0 = np.floor(source_frame).astype(np.int64)
  index1 = np.minimum(index0 + 1, len(values) - 1)
  alpha = (source_frame - index0).reshape((-1,) + (1,) * (values.ndim - 1))
  return values[index0] * (1.0 - alpha) + values[index1] * alpha


def _quaternion_sample(values: np.ndarray, source_frame: np.ndarray) -> np.ndarray:
  # Normalized linear interpolation with shortest-arc sign correction is stable
  # for the small (20 ms) rotations in these retargeted clips.
  index0 = np.floor(source_frame).astype(np.int64)
  index1 = np.minimum(index0 + 1, len(values) - 1)
  q0 = values[index0]
  q1 = values[index1]
  q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0, -q1, q1)
  alpha = (source_frame - index0)[:, None, None]
  result = q0 * (1.0 - alpha) + q1 * alpha
  return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-8)


def main(args: Args) -> None:
  if args.speed_scale <= 0.0:
    raise ValueError("speed_scale must be positive")
  with np.load(args.input_file) as source:
    data = {key: source[key] for key in source.files}

  source_count = data["joint_pos"].shape[0]
  output_count = int(round((source_count - 1) / args.speed_scale)) + 1
  source_frame = np.linspace(0.0, source_count - 1, output_count)

  output = {
    "fps": data["fps"].copy(),
    "joint_pos": _linear_sample(data["joint_pos"], source_frame).astype(np.float32),
    "joint_vel": (
      _linear_sample(data["joint_vel"], source_frame) * args.speed_scale
    ).astype(np.float32),
    "body_pos_w": _linear_sample(data["body_pos_w"], source_frame).astype(np.float32),
    "body_quat_w": _quaternion_sample(data["body_quat_w"], source_frame).astype(
      np.float32
    ),
    "body_lin_vel_w": (
      _linear_sample(data["body_lin_vel_w"], source_frame) * args.speed_scale
    ).astype(np.float32),
    "body_ang_vel_w": (
      _linear_sample(data["body_ang_vel_w"], source_frame) * args.speed_scale
    ).astype(np.float32),
  }
  args.output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.output_file, **output)
  duration = (output_count - 1) / float(output["fps"][0])
  displacement = output["body_pos_w"][-1, 0, :2] - output["body_pos_w"][0, 0, :2]
  print(
    f"Wrote {args.output_file}: {source_count}->{output_count} frames, "
    f"duration={duration:.3f}s, planar_speed={np.linalg.norm(displacement) / duration:.3f}m/s"
  )


if __name__ == "__main__":
  main(tyro.cli(Args))
