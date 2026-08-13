# Unitree G1 策略评测

`scripts/play.py` 用于主观观察动作，`scripts/evaluate.py` 用于可复现的无头定量评测。评测脚本采用训练配置而不是 play 配置，因此保留下列训练条件：

- 脚部摩擦随机化；
- 编码器偏置；
- 躯干质心偏移；
- 每 5～6 秒一次的随机推扰；
- 20 秒有限 episode。

评测时仅关闭 curriculum，防止测试过程中改变地形难度或速度范围。每个并行环境只统计第一个完整 episode。

## 单 checkpoint 评测

```bash
cd /data/users/yanghao/projects/unitree_rl_mjlab
source env.sh

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate.py Unitree-G1-Flat \
  --checkpoint=logs/rsl_rl/g1_velocity/2026-08-13_03-21-03_g1_full_baseline/model_10000.pt \
  --num-envs=64 \
  --device=cuda:0
```

默认使用 seed 42。多个 seed 的 tyro 参数需要用 tuple 形式，例如：

```bash
--seeds '(42,43,44)'
```

## 比较多个 checkpoint

Shell 会展开通配符，路径需要加引号：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate.py Unitree-G1-Flat \
  --checkpoint='logs/rsl_rl/g1_velocity/2026-08-13_03-21-03_g1_full_baseline/model_9*.pt' \
  --num-envs=64 \
  --seeds '(42,43,44)' \
  --device=cuda:0
```

输出目录默认为 `evaluations/<task>/<UTC时间>/`，包含：

- `episodes.csv`：每个 episode 的原始指标；
- `summary.csv`：每个 checkpoint 的均值、标准差和跌倒率；
- `results.json`：配置、模型路径及机器可读结果。

生成对比图：

```bash
python scripts/plot_evaluation.py \
  --summary-csv=evaluations/Unitree-G1-Flat/<时间>/summary.csv
```

## 指标说明

| 指标 | 含义 | 方向 |
|---|---|---|
| `return` | 20 秒 episode 累计奖励 | 越高越好 |
| `fall_rate` | 因姿态超过阈值而提前结束的比例 | 越低越好 |
| `timeout_rate` | 成功坚持到时间上限的比例 | 越高越好 |
| `velocity_xy_rmse` | 机体坐标系 XY 速度跟踪 RMSE | 越低越好 |
| `velocity_yaw_rmse` | yaw 角速度跟踪 RMSE | 越低越好 |
| `foot_slip_mean_mps` | 脚接触地面时的平均水平速度 | 越低越好 |
| `action_delta_rms` | 相邻策略动作变化的 RMS | 越低通常越平滑 |
| `base_tilt_rms` | 重力在机体 XY 平面的投影 RMS | 越低通常越直立 |

这些指标用于相同任务和评测设置下的相对比较。它们不能独立证明真机安全性，也不能替代动作视频检查。

## 已完成的基线评测

2026-08-13 使用 `model_10000.pt`，在 seed 42、43、44 下各运行 64 个环境，共完成 192 个独立的 20 秒 episode：

```text
fall_rate             0.000
timeout_rate          1.000
velocity_xy_rmse      0.413 ± 0.112 m/s
velocity_yaw_rmse     0.269 ± 0.045 rad/s
foot_slip_mean_mps    0.085 ± 0.029 m/s
action_delta_rms      0.216 ± 0.014
base_tilt_rms         0.053 ± 0.013
episode return        53.672 ± 2.696
```

结果位于 `evaluations/flat_baseline_model10000_192ep/`。这是后续 Rough 和历史观测策略的 Flat 基线；选择最佳 checkpoint 时仍应比较多个后期 checkpoint。
