"""Select a lane-stable Sprint-v3 checkpoint and enforce the stage-1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("results", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()

  payload = json.loads(args.results.read_text())
  summaries = payload["summaries"]
  eligible = [
    row
    for row in summaries
    if row["fall_rate"] <= 0.05
    and row["world_forward_speed_mean_mps_mean"] >= 1.25
    and row["foot_slip_mean_mps_mean"] <= 0.35
  ]
  pool = eligible or summaries
  best = min(
    pool,
    key=lambda row: (
      row["outside_lane_rate"],
      -row["duration_s_mean"],
      -row["world_forward_speed_mean_mps_mean"],
    ),
  )
  passed = bool(eligible) and best["outside_lane_rate"] <= 0.20
  decision = {
    "passed": passed,
    "checkpoint": best["checkpoint"],
    "criteria": {
      "fall_rate_max": 0.05,
      "outside_lane_rate_max": 0.20,
      "world_forward_speed_min_mps": 1.25,
      "foot_slip_max_mps": 0.35,
    },
    "best_summary": best,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(decision, indent=2) + "\n")
  print(json.dumps(decision, indent=2))
  raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
  main()
