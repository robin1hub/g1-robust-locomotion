#!/usr/bin/env python3
"""Retarget a LAFAN1 BVH clip to Unitree G1 without opening a GUI."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive frame index.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive frame index.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")

    frames, human_height = load_bvh_file(str(args.bvh_file), format="lafan1")
    end = len(frames) if args.end_frame is None else min(args.end_frame, len(frames))
    if end <= args.start_frame:
        raise ValueError(f"Empty frame range [{args.start_frame}, {end})")

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot=args.robot,
        actual_human_height=human_height,
        verbose=not args.quiet,
    )
    selected = frames[args.start_frame:end]
    qpos = []
    started = time.perf_counter()
    for frame in tqdm(selected, desc="Retargeting", unit="frame"):
        qpos.append(retargeter.retarget(frame))
    elapsed = time.perf_counter() - started
    qpos_array = np.asarray(qpos, dtype=np.float32)

    motion = {
        "fps": args.fps,
        "root_pos": qpos_array[:, :3],
        # GMR/MuJoCo qpos is wxyz; its saved motion convention is xyzw.
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(motion, file)

    summary = {
        "source": str(args.bvh_file.resolve()),
        "output": str(args.output.resolve()),
        "robot": args.robot,
        "source_frame_count": len(frames),
        "start_frame": args.start_frame,
        "end_frame": end,
        "saved_frame_count": len(selected),
        "fps": args.fps,
        "duration_seconds": len(selected) / args.fps,
        "retarget_seconds": elapsed,
        "qpos_shape": list(qpos_array.shape),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
