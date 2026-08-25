"""Plot motor-strength and action-delay robustness sweeps."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


def _read(*paths: str) -> list[dict[str, str]]:
  rows: dict[str, dict[str, str]] = {}
  for path in paths:
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as file:
      rows.update({row["scenario"]: row for row in csv.DictReader(file)})
  return list(rows.values())


def main(
  motor_summary_file: str,
  motor_refine_summary_file: str,
  delay_summary_file: str,
  delay_refine_summary_file: str,
  output_file: str,
) -> None:
  motor_rows = _read(motor_summary_file, motor_refine_summary_file)
  delay_rows = _read(delay_summary_file, delay_refine_summary_file)
  clean = next(row for row in motor_rows if row["scenario"] == "clean")
  motor_rows = [row for row in motor_rows if row["scenario"].startswith("motor_scale_")]
  motor_rows.append(clean)
  motor_rows.sort(key=lambda row: float(row["motor_strength_scale"]))
  delay_rows = [row for row in delay_rows if row["scenario"].startswith("delay_")]
  delay_rows.append(clean)
  delay_rows.sort(key=lambda row: float(row["action_delay_ms"]))

  fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
  groups = (
    (motor_rows, "motor_strength_scale", "Motor strength scale"),
    (delay_rows, "action_delay_ms", "Action delay (ms)"),
  )
  for column, (rows, x_key, x_label) in enumerate(groups):
    x = [float(row[x_key]) for row in rows]
    axes[0, column].plot(
      x, [100 * float(row["success_rate"]) for row in rows], marker="o"
    )
    axes[0, column].set(xlabel=x_label, ylabel="Success rate (%)", ylim=(-5, 105))
    axes[1, column].plot(
      x,
      [float(row["foot_slip_mean_mps_mean"]) for row in rows],
      marker="o",
      color="tab:orange",
    )
    axes[1, column].set(xlabel=x_label, ylabel="Mean foot slip (m/s)")

  for axis in axes.flat:
    axis.grid(alpha=0.3)
  fig.suptitle("G1 Actuation Robustness (3 seeds × 64 episodes)")
  output_path = Path(output_file).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  print(f"Saved plot to {output_path}")


if __name__ == "__main__":
  tyro.cli(main)
