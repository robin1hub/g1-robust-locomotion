"""Plot local low-friction patch results from the robustness benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


def main(summary_file: str, output_file: str | None = None) -> None:
  summary_path = Path(summary_file).expanduser().resolve()
  with summary_path.open(newline="", encoding="utf-8") as file:
    rows = [
      row
      for row in csv.DictReader(file)
      if row["scenario"].startswith("local_friction_")
    ]
  if not rows:
    raise ValueError(f"No local_friction scenarios found in {summary_path}")
  rows.sort(key=lambda row: float(row["local_patch_friction"]))
  friction = [float(row["local_patch_friction"]) for row in rows]

  fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
  axes[0, 0].plot(
    friction,
    [100 * float(row["success_rate"]) for row in rows],
    marker="o",
    label="Episode success",
  )
  axes[0, 0].plot(
    friction,
    [100 * float(row["post_patch_reach_rate"]) for row in rows],
    marker="s",
    linestyle="--",
    label="Reached patch exit",
  )
  axes[0, 0].set(xlabel="Local patch friction", ylabel="Rate (%)")
  axes[0, 0].set_ylim(-5, 105)
  axes[0, 0].legend()

  axes[0, 1].plot(
    friction,
    [float(row["patch_foot_slip_mean_mps_mean"]) for row in rows],
    marker="o",
    color="tab:red",
  )
  axes[0, 1].set(xlabel="Local patch friction", ylabel="Patch contact slip (m/s)")

  axes[1, 0].plot(
    friction,
    [float(row["root_position_rmse_m_mean"]) for row in rows],
    marker="o",
    color="tab:orange",
  )
  axes[1, 0].set(xlabel="Local patch friction", ylabel="Root RMSE (m)")

  axes[1, 1].plot(
    friction,
    [float(row["post_patch_root_position_rmse_m_mean"]) for row in rows],
    marker="o",
    color="tab:green",
  )
  axes[1, 1].set(
    xlabel="Local patch friction",
    ylabel="Post-patch root RMSE (m, reached only)",
  )

  for axis in axes.flat:
    axis.grid(alpha=0.3)
  fig.suptitle("G1 Local Low-Friction Patch Robustness (3 seeds × 64 episodes)")

  output_path = (
    Path(output_file).expanduser().resolve()
    if output_file
    else summary_path.with_name("local_patch_robustness.png")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  print(f"Saved plot to {output_path}")


if __name__ == "__main__":
  tyro.cli(main)
