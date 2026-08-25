"""Plot single-factor and compound-disturbance robustness results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


LABELS = {
  "clean": "Clean",
  "local_friction_0p2": "Patch",
  "motor_scale_0p58": "Motor",
  "delay_35ms": "Delay",
  "payload_6kg": "Payload",
  "push_0p25": "Push",
  "combo_patch_motor": "Patch+Motor",
  "combo_patch_delay": "Patch+Delay",
  "combo_motor_delay": "Motor+Delay",
  "combo_actuation_payload": "Motor+Delay\n+Payload",
  "combo_all": "All five",
}


def main(summary_file: str, output_file: str | None = None) -> None:
  summary_path = Path(summary_file).expanduser().resolve()
  with summary_path.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
  rows = [row for row in rows if row["scenario"] in LABELS]

  labels = [LABELS[row["scenario"]] for row in rows]
  colors = ["tab:blue" if not row["scenario"].startswith("combo_") else "tab:red" for row in rows]
  fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
  axes[0].bar(labels, [100 * float(row["success_rate"]) for row in rows], color=colors)
  axes[0].set(ylabel="Success rate (%)", ylim=(0, 105))
  axes[1].bar(
    labels,
    [float(row["root_displacement_m_mean"]) for row in rows],
    color=colors,
  )
  axes[1].set(ylabel="Root displacement (m)")
  for axis in axes:
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=20)
  fig.suptitle("G1 Compound Disturbance Robustness (3 seeds × 64 episodes)")

  output_path = (
    Path(output_file).expanduser().resolve()
    if output_file
    else summary_path.with_name("combination_robustness.png")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  print(f"Saved plot to {output_path}")


if __name__ == "__main__":
  tyro.cli(main)
