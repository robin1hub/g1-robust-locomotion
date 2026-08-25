"""Compare the independently selected E2-B0 A/B checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--a", type=Path, required=True)
  parser.add_argument("--b", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  variants = {
    "B0-A_task_fix_only": json.loads(args.a.read_text()),
    "B0-B_reward_fix": json.loads(args.b.read_text()),
  }
  ranking = sorted(
    (
      {
        "variant": name,
        **decision["best"],
      }
      for name, decision in variants.items()
    ),
    key=lambda row: (not row["passed"], row["score"]),
  )
  output = {
    "passed": ranking[0]["passed"],
    "winner": ranking[0],
    "ranking": ranking,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(output, indent=2) + "\n")
  print(json.dumps(output, indent=2))


if __name__ == "__main__":
  main()
