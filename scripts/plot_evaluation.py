"""Create compact comparison plots from ``scripts/evaluate.py`` output."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


@dataclass(frozen=True)
class PlotConfig:
  summary_csv: str
  output: str | None = None


def main() -> None:
  cfg = tyro.cli(PlotConfig)
  summary_path = Path(cfg.summary_csv).resolve()
  with summary_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No rows in {summary_path}")

  labels = [Path(row["checkpoint"]).stem.replace("model_", "") for row in rows]
  metrics = (
    ("return_mean", "Episode return", True),
    ("velocity_xy_rmse_mean", "XY velocity RMSE (m/s)", False),
    ("fall_rate", "Fall rate", False),
    ("foot_slip_mean_mps_mean", "Contact slip speed (m/s)", False),
  )

  figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
  for axis, (field, title, higher_is_better) in zip(axes.flat, metrics, strict=True):
    values = [float(row[field]) for row in rows]
    color = "#2a9d8f" if higher_is_better else "#457b9d"
    axis.bar(labels, values, color=color)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=35)
    for index, value in enumerate(values):
      axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

  figure.suptitle("Unitree G1 checkpoint evaluation")
  output = Path(cfg.output).resolve() if cfg.output else summary_path.with_suffix(".png")
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=180)
  print(f"[INFO] Plot written to: {output}")


if __name__ == "__main__":
  main()
