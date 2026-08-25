#!/usr/bin/env bash
set -euo pipefail

worker_id="${1:?usage: $0 <worker_id: 1|2|3> <physical_gpu_id>}"
physical_gpu_id="${2:?usage: $0 <worker_id: 1|2|3> <physical_gpu_id>}"

project_dir="/data/users/yanghao/projects/unitree_rl_mjlab"
python_bin="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python"
checkpoint="$project_dir/logs/rsl_rl/g1_tracking/2026-08-19_06-28-23_t001_joint_dr_full_2gpu_resume6000/model_9999.pt"
motion_file="$project_dir/src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz"
output_root="$project_dir/evaluations/Unitree-G1-Tracking-Robust"

export CUDA_VISIBLE_DEVICES="$physical_gpu_id"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH="/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib"
export MPLCONFIGDIR="/data/users/yanghao/tmp/matplotlib"

cd "$project_dir"

run_matrix() {
  local matrix_name="$1"
  local scenarios="$2"
  "$python_bin" scripts/evaluate_tracking_robustness.py Unitree-G1-Tracking-Robust \
    --checkpoint "$checkpoint" \
    --motion-file "$motion_file" \
    --num-envs 64 \
    --seeds "(42,43,44)" \
    --scenarios "$scenarios" \
    --output-dir "$output_root/$matrix_name"
}

case "$worker_id" in
  1)
    run_matrix t001_e001_global_full_20260819 \
      "('clean','friction_0p2','friction_0p4','friction_0p6','friction_0p8','push_0p25','push_0p5','push_0p75','push_1p0')"
    run_matrix t001_e004_inertial_full_20260819 \
      "('clean','mass_scale_0p8','mass_scale_1p2','mass_scale_1p4','payload_5kg','payload_10kg','payload_15kg','com_y_pos_0p03','com_y_neg_0p03','com_y_pos_0p06','com_y_neg_0p06')"
    run_matrix t001_e005_motor_refine_20260819 \
      "('motor_scale_0p58','motor_scale_0p56','motor_scale_0p54','motor_scale_0p52','motor_scale_0p48','motor_scale_0p45')"
    run_matrix t001_e006_delay_refine_20260819 \
      "('delay_35ms','delay_45ms','delay_50ms','delay_55ms')"
    ;;
  2)
    run_matrix t001_e002_friction_refine_20260819 \
      "('friction_0p25','friction_0p3','friction_0p35')"
    run_matrix t001_e004_inertial_refine_20260819 \
      "('mass_scale_1p25','mass_scale_1p3','mass_scale_1p35','payload_6kg','payload_7kg','payload_8kg','payload_9kg','com_y_pos_0p09','com_y_neg_0p09','com_y_pos_0p12','com_y_neg_0p12')"
    run_matrix t001_e005_motor_full_20260819 \
      "('clean','motor_scale_0p9','motor_scale_0p8','motor_scale_0p7','motor_scale_0p6','motor_scale_0p5')"
    ;;
  3)
    run_matrix t001_e003_local_patch_full_20260819 \
      "('local_friction_0p05','local_friction_0p1','local_friction_0p2','local_friction_0p25','local_friction_0p3','local_friction_0p35')"
    run_matrix t001_e006_delay_full_20260819 \
      "('clean','delay_10ms','delay_20ms','delay_30ms','delay_40ms','delay_60ms','delay_80ms')"
    run_matrix t001_e007_combo_full_20260819 \
      "('clean','local_friction_0p2','motor_scale_0p58','delay_35ms','payload_6kg','push_0p25','combo_patch_motor','combo_patch_delay','combo_motor_delay','combo_actuation_payload','combo_all')"
    ;;
  *)
    echo "worker_id must be 1, 2, or 3" >&2
    exit 2
    ;;
esac
