"""Plot summary curves produced by evaluate_tracking_robustness.py."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


def _read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as file:
    return list(csv.DictReader(file))


def _metric(row: dict[str, str], name: str) -> float:
  return float(row[name])


def main(
  summary_file: str,
  output_file: str | None = None,
  extra_summary_file: str | None = None,
) -> None:
  summary_path = Path(summary_file).expanduser().resolve()
  rows = _read_rows(summary_path)
  if extra_summary_file:
    rows.extend(_read_rows(Path(extra_summary_file).expanduser().resolve()))
  rows = list({row["scenario"]: row for row in rows}.values())
  clean = next(row for row in rows if row["scenario"] == "clean")

  friction_rows = [row for row in rows if row["scenario"].startswith("friction_")]
  friction_rows.append(clean)
  friction_rows.sort(key=lambda row: _metric(row, "foot_friction"))

  push_rows = [row for row in rows if row["scenario"].startswith("push_")]
  push_rows.append(clean)
  push_rows.sort(key=lambda row: _metric(row, "lateral_velocity_push_mps"))

  fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
  friction = [_metric(row, "foot_friction") for row in friction_rows]
  push = [_metric(row, "lateral_velocity_push_mps") for row in push_rows]

  axes[0, 0].plot(
    friction,
    [100 * _metric(row, "success_rate") for row in friction_rows],
    marker="o",
  )
  axes[0, 0].set(xlabel="Foot friction coefficient", ylabel="Success rate (%)")
  axes[0, 0].set_ylim(-5, 105)

  axes[0, 1].plot(
    friction,
    [_metric(row, "foot_slip_mean_mps_mean") for row in friction_rows],
    marker="o",
    color="tab:red",
  )
  axes[0, 1].set(xlabel="Foot friction coefficient", ylabel="Contact slip (m/s)")

  axes[1, 0].plot(
    push,
    [_metric(row, "root_position_rmse_m_mean") for row in push_rows],
    marker="o",
    color="tab:orange",
  )
  axes[1, 0].set(xlabel="Lateral velocity impulse (m/s)", ylabel="Root RMSE (m)")

  axes[1, 1].plot(
    push,
    [_metric(row, "root_displacement_m_mean") for row in push_rows],
    marker="o",
    color="tab:green",
  )
  axes[1, 1].set(
    xlabel="Lateral velocity impulse (m/s)", ylabel="Root displacement (m)"
  )

  for axis in axes.flat:
    axis.grid(alpha=0.3)
  fig.suptitle("G1 LAFAN1 Tracking Robustness (3 seeds × 64 episodes)")

  output_path = (
    Path(output_file).expanduser().resolve()
    if output_file
    else summary_path.with_name("robustness_curves.png")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  print(f"Saved plot to {output_path}")


if __name__ == "__main__":
  tyro.cli(main)
