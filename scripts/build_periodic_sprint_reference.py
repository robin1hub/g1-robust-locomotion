"""Build a constant-speed periodic G1 reference from one mocap gait cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def periodicize_cycle(cycle: np.ndarray) -> np.ndarray:
  """Remove endpoint drift while preserving motion inside the selected cycle."""
  result = cycle.copy()
  alpha = np.linspace(0.0, 1.0, len(cycle), dtype=np.float64)[:, None]

  joint_delta = np.arctan2(
    np.sin(cycle[-1, 7:] - cycle[0, 7:]),
    np.cos(cycle[-1, 7:] - cycle[0, 7:]),
  )
  result[:, 7:] -= alpha * joint_delta

  rotations = Rotation.from_quat(cycle[:, 3:7])
  final_correction = rotations[0] * rotations[-1].inv()
  corrections = Rotation.from_rotvec(alpha * final_correction.as_rotvec())
  result[:, 3:7] = (corrections * rotations).as_quat()

  result[:, 2] -= alpha[:, 0] * (cycle[-1, 2] - cycle[0, 2])
  return result


def build_periodic_motion(
  cycle: np.ndarray,
  fps: float,
  cycles: int,
  target_speed: float,
) -> np.ndarray:
  cycle = periodicize_cycle(cycle)
  cycle_intervals = len(cycle) - 1
  cycle_duration = cycle_intervals / fps

  displacement = cycle[-1, :2] - cycle[0, :2]
  forward = displacement / np.linalg.norm(displacement)
  alpha = np.linspace(0.0, 1.0, len(cycle))[:, None]
  horizontal_residual = (
    cycle[:, :2] - cycle[0, :2] - alpha * displacement
  )

  frames = []
  for cycle_index in range(cycles):
    stop = len(cycle) if cycle_index == cycles - 1 else len(cycle) - 1
    phase_time = np.arange(stop, dtype=np.float64) / fps
    absolute_time = cycle_index * cycle_duration + phase_time
    repeated = cycle[:stop].copy()
    repeated[:, :2] = (
      forward[None, :] * (target_speed * absolute_time[:, None])
      + horizontal_residual[:stop]
    )
    frames.append(repeated)
  return np.concatenate(frames, axis=0)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-file", type=Path, required=True)
  parser.add_argument("--output-file", type=Path, required=True)
  parser.add_argument("--start-frame", type=int, required=True)
  parser.add_argument("--cycle-frames", type=int, required=True)
  parser.add_argument("--fps", type=float, default=120.0)
  parser.add_argument("--cycles", type=int, default=5)
  parser.add_argument("--target-speed", type=float, default=4.3)
  args = parser.parse_args()

  source = np.loadtxt(args.input_file, delimiter=",", skiprows=1)
  stop_frame = args.start_frame + args.cycle_frames
  cycle_wxyz = source[args.start_frame : stop_frame + 1].copy()
  if len(cycle_wxyz) != args.cycle_frames + 1:
    raise ValueError("Selected cycle extends beyond the input motion")

  # MotionDecode stores wxyz; mjlab's CSV converter expects xyzw.
  cycle = np.concatenate(
    (cycle_wxyz[:, :3], cycle_wxyz[:, 4:7], cycle_wxyz[:, 3:4], cycle_wxyz[:, 7:]),
    axis=1,
  )
  motion = build_periodic_motion(
    cycle,
    fps=args.fps,
    cycles=args.cycles,
    target_speed=args.target_speed,
  )
  args.output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(args.output_file, motion, delimiter=",", fmt="%.8f")

  duration = (len(motion) - 1) / args.fps
  print(
    f"Saved {len(motion)} frames ({duration:.3f}s) at "
    f"{args.target_speed:.3f} m/s to {args.output_file}"
  )


if __name__ == "__main__":
  main()
