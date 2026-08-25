"""Compare matched baseline and candidate robustness benchmark summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


METRICS = (
  "success_rate",
  "length_steps_mean",
  "root_position_rmse_m_mean",
  "body_mpkpe_rmse_m_mean",
  "foot_slip_mean_mps_mean",
  "patch_foot_slip_mean_mps_mean",
  "post_patch_root_position_rmse_m_mean",
  "action_delta_rms_mean",
  "root_displacement_m_mean",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--pair",
    action="append",
    required=True,
    metavar="LABEL:BASELINE_SUMMARY:CANDIDATE_SUMMARY",
  )
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def canonical_scenario(name: str) -> str:
  """Normalize equivalent decimal spellings such as 0p30 and 0p3."""

  def trim_decimal(match: re.Match[str]) -> str:
    digits = match.group(1).rstrip("0") or "0"
    return f"0p{digits}"

  return re.sub(r"0p([0-9]+)", trim_decimal, name)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as file:
    return {canonical_scenario(row["scenario"]): row for row in csv.DictReader(file)}


def as_float(row: dict[str, str], field: str) -> float:
  value = row.get(field, "")
  return float(value) if value else float("nan")


def main() -> None:
  args = parse_args()
  output_rows: list[dict[str, object]] = []
  for pair in args.pair:
    label, baseline_name, candidate_name = pair.split(":", maxsplit=2)
    baseline_path = Path(baseline_name).expanduser().resolve()
    candidate_path = Path(candidate_name).expanduser().resolve()
    baseline_rows = read_rows(baseline_path)
    candidate_rows = read_rows(candidate_path)
    if baseline_rows.keys() != candidate_rows.keys():
      missing_candidate = sorted(baseline_rows.keys() - candidate_rows.keys())
      missing_baseline = sorted(candidate_rows.keys() - baseline_rows.keys())
      raise ValueError(
        f"{label}: unmatched scenarios; missing candidate={missing_candidate}, "
        f"missing baseline={missing_baseline}"
      )
    for scenario in baseline_rows:
      baseline = baseline_rows[scenario]
      candidate = candidate_rows[scenario]
      row: dict[str, object] = {
        "matrix": label,
        "scenario": scenario,
        "baseline_episodes": int(baseline["episodes"]),
        "candidate_episodes": int(candidate["episodes"]),
      }
      for metric in METRICS:
        baseline_value = as_float(baseline, metric)
        candidate_value = as_float(candidate, metric)
        row[f"baseline_{metric}"] = baseline_value
        row[f"candidate_{metric}"] = candidate_value
        row[f"delta_{metric}"] = candidate_value - baseline_value
      row["success_rate_delta_pp"] = (
        as_float(candidate, "success_rate") - as_float(baseline, "success_rate")
      ) * 100.0
      output_rows.append(row)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)
  print(f"Wrote {len(output_rows)} matched scenario comparisons to {args.output}")


if __name__ == "__main__":
  main()
