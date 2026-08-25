"""Select an E2-A checkpoint from straight, lateral, and yaw command tests."""

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
  parser.add_argument("--yaw-pos", type=Path, required=True)
  parser.add_argument("--yaw-neg", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  baseline = json.loads(args.baseline.read_text())["summaries"][0]
  baseline_speed = baseline["body_forward_speed_mean_mps_mean"]
  sets = {
    "straight": summaries(args.straight),
    "lat_pos": summaries(args.lat_pos),
    "lat_neg": summaries(args.lat_neg),
    "yaw_pos": summaries(args.yaw_pos),
    "yaw_neg": summaries(args.yaw_neg),
  }
  names = set.intersection(*(set(rows) for rows in sets.values()))
  candidates = []
  for name in sorted(names):
    s = sets["straight"][name]
    lp, ln = sets["lat_pos"][name], sets["lat_neg"][name]
    yp, yn = sets["yaw_pos"][name], sets["yaw_neg"][name]
    row = {
      "checkpoint": s["checkpoint"],
      "fall_rate_max": max(x["fall_rate"] for x in (s, lp, ln, yp, yn)),
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
        yp["yaw_direction_correct_fraction_mean"],
        yn["yaw_direction_correct_fraction_mean"],
      ),
      "yaw_rmse_max_radps": max(
        yp["velocity_yaw_rmse_mean"], yn["velocity_yaw_rmse_mean"]
      ),
    }
    row["passed"] = (
      row["fall_rate_max"] <= 0.02
      and row["straight_retention"] >= 0.95
      and row["lateral_direction_correct_min"] >= 0.95
      and row["lateral_rmse_max_mps"] <= 0.15
      and row["yaw_direction_correct_min"] >= 0.95
      and row["yaw_rmse_max_radps"] <= 0.20
    )
    row["score"] = (
      4.0 * row["fall_rate_max"]
      + max(0.0, 0.95 - row["straight_retention"])
      + max(0.0, 0.95 - row["lateral_direction_correct_min"])
      + max(0.0, row["lateral_rmse_max_mps"] - 0.15) / 0.15
      + max(0.0, 0.95 - row["yaw_direction_correct_min"])
      + max(0.0, row["yaw_rmse_max_radps"] - 0.20) / 0.20
    )
    candidates.append(row)

  best = min(candidates, key=lambda row: (not row["passed"], row["score"]))
  output = {
    "passed": best["passed"],
    "baseline_straight_speed_mps": baseline_speed,
    "best": best,
    "candidates": candidates,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(output, indent=2) + "\n")
  print(json.dumps(output, indent=2))


if __name__ == "__main__":
  main()
