"""Plot mass, payload, and COM-offset robustness results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


def _read(path: str) -> list[dict[str, str]]:
  with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as file:
    return list(csv.DictReader(file))


def main(
  summary_file: str,
  extra_summary_file: str,
  output_file: str | None = None,
) -> None:
  rows = _read(summary_file) + _read(extra_summary_file)
  rows = list({row["scenario"]: row for row in rows}.values())
  clean = next(row for row in rows if row["scenario"] == "clean")

  mass_rows = [row for row in rows if row["scenario"].startswith("mass_scale_")]
  mass_rows.append(clean)
  mass_rows.sort(key=lambda row: float(row["mass_scale"]))

  payload_rows = [row for row in rows if row["scenario"].startswith("payload_")]
  payload_rows.append(clean)
  payload_rows.sort(key=lambda row: float(row["torso_payload_kg"]))

  com_rows = [row for row in rows if row["scenario"].startswith("com_y_")]
  com_rows.append(clean)
  com_rows.sort(key=lambda row: float(row["torso_com_offset_y_m"]))

  groups = (
    (mass_rows, "mass_scale", "Whole-body mass scale"),
    (payload_rows, "torso_payload_kg", "Torso point payload (kg)"),
    (com_rows, "torso_com_offset_y_m", "Torso COM y offset (m)"),
  )
  fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
  for column, (group_rows, x_key, x_label) in enumerate(groups):
    x = [float(row[x_key]) for row in group_rows]
    axes[0, column].plot(
      x,
      [100 * float(row["success_rate"]) for row in group_rows],
      marker="o",
    )
    axes[0, column].set(xlabel=x_label, ylabel="Success rate (%)")
    axes[0, column].set_ylim(-5, 105)
    axes[1, column].plot(
      x,
      [float(row["root_position_rmse_m_mean"]) for row in group_rows],
      marker="o",
      color="tab:orange",
    )
    axes[1, column].set(xlabel=x_label, ylabel="Root RMSE (m)")

  for axis in axes.flat:
    axis.grid(alpha=0.3)
  fig.suptitle("G1 Inertial Robustness (3 seeds × 64 episodes)")

  output_path = (
    Path(output_file).expanduser().resolve()
    if output_file
    else Path(summary_file).expanduser().resolve().with_name("inertial_robustness.png")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  print(f"Saved plot to {output_path}")


if __name__ == "__main__":
  tyro.cli(main)
