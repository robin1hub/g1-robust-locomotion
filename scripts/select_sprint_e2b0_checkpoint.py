"""Select an E2-B0 yaw-probe checkpoint without lane-based metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summaries(path: Path) -> dict[str, dict]:
  rows = json.loads(path.read_text())["summaries"]
  return {Path(row["checkpoint"]).name: row for row in rows}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--straight", type=Path, required=True)
  parser.add_argument("--lat-pos", type=Path, required=True)
  parser.add_argument("--lat-neg", type=Path, required=True)
  parser.add_argument("--yaw-pos-015", type=Path, required=True)
  parser.add_argument("--yaw-neg-015", type=Path, required=True)
  parser.add_argument("--yaw-pos-030", type=Path)
  parser.add_argument("--yaw-neg-030", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  baseline = json.loads(args.baseline.read_text())["summaries"][0]
  baseline_speed = baseline["body_forward_speed_mean_mps_mean"]
  sets = {
    "straight": summaries(args.straight),
    "lat_pos": summaries(args.lat_pos),
    "lat_neg": summaries(args.lat_neg),
    "yaw_pos_015": summaries(args.yaw_pos_015),
    "yaw_neg_015": summaries(args.yaw_neg_015),
  }
  if (args.yaw_pos_030 is None) != (args.yaw_neg_030 is None):
    parser.error("--yaw-pos-030 and --yaw-neg-030 must be provided together")
  if args.yaw_pos_030 is not None and args.yaw_neg_030 is not None:
    sets["yaw_pos_030"] = summaries(args.yaw_pos_030)
    sets["yaw_neg_030"] = summaries(args.yaw_neg_030)
  names = set.intersection(*(set(rows) for rows in sets.values()))
  candidates = []
  for name in sorted(names):
    cases = [rows[name] for rows in sets.values()]
    s = sets["straight"][name]
    lp, ln = sets["lat_pos"][name], sets["lat_neg"][name]
    yaw_cases = [rows[name] for key, rows in sets.items() if key.startswith("yaw_")]
    row = {
      "checkpoint": s["checkpoint"],
      "failure_rate_max": max(
        case["fall_rate"] + case["illegal_contact_rate"] for case in cases
      ),
      "straight_speed_mps": s["body_forward_speed_mean_mps_mean"],
      "straight_retention": s["body_forward_speed_mean_mps_mean"] / baseline_speed,
      "lateral_direction_correct_min": min(
        lp["lateral_direction_correct_fraction_mean"],
        ln["lateral_direction_correct_fraction_mean"],
      ),
      "lateral_rmse_max_mps": max(
        lp["velocity_y_rmse_mean"], ln["velocity_y_rmse_mean"]
      ),
      "yaw_direction_correct_min": min(
        case["yaw_direction_correct_fraction_mean"] for case in yaw_cases
      ),
      "yaw_rmse_max_radps": max(
        case["velocity_yaw_rmse_mean"] for case in yaw_cases
      ),
    }
    row["passed"] = (
      row["failure_rate_max"] <= 0.02
      and row["straight_retention"] >= 0.95
      and row["lateral_direction_correct_min"] >= 0.90
      and row["lateral_rmse_max_mps"] <= 0.18
      and row["yaw_direction_correct_min"] >= 0.90
      and row["yaw_rmse_max_radps"] <= 0.30
    )
    row["score"] = (
      4.0 * row["failure_rate_max"]
      + max(0.0, 0.95 - row["straight_retention"])
      + max(0.0, 0.90 - row["lateral_direction_correct_min"])
      + max(0.0, row["lateral_rmse_max_mps"] - 0.18) / 0.18
      + max(0.0, 0.90 - row["yaw_direction_correct_min"])
      + max(0.0, row["yaw_rmse_max_radps"] - 0.30) / 0.30
    )
    candidates.append(row)

  best = min(candidates, key=lambda row: (not row["passed"], row["score"]))
  output = {
    "passed": best["passed"],
    "baseline_straight_speed_mps": baseline_speed,
    "thresholds": {
      "failure_rate_max": 0.02,
      "straight_retention_min": 0.95,
      "lateral_direction_correct_min": 0.90,
      "lateral_rmse_max_mps": 0.18,
      "yaw_direction_correct_min": 0.90,
      "yaw_rmse_max_radps": 0.30,
    },
    "best": best,
    "candidates": candidates,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(output, indent=2) + "\n")
  print(json.dumps(output, indent=2))


if __name__ == "__main__":
  main()
