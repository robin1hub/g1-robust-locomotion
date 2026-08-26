#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MOTION_FILE="${MOTION_FILE:-src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed400.npz}"
export LOAD_RUN="${LOAD_RUN:-2026-08-25_10-35-15_tracking_sprint380_antislip_4gpu_8192_each_20}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-model_15.pt}"
export RUN_NAME="${RUN_NAME:-tracking_sprint400_antislip_4gpu_8192_each_20}"

exec "$project_dir/scripts/run_g1_tracking_sprint380_antislip.sh"
