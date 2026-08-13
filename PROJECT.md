# Unitree G1 鲁棒运动控制与评测

本仓库基于 Unitree 官方 `unitree_rl_mjlab` 开展 G1 人形机器人运动控制实验。目前已完成平地速度控制基线训练，以及可复现的无头策略评测框架；下一阶段将扩展至复杂地形和历史观测策略。

## 已完成工作

- 基于 MuJoCo、MjLab、RSL-RL 和 PPO 训练 Unitree G1 全身速度控制策略；
- 使用 4096 个并行仿真环境完成约 9.83 亿步交互采样；
- 训练吞吐约 5.6 万 steps/s，总训练时间约 4 小时 44 分钟；
- Actor 使用 `98 → 512 → 256 → 128 → 29` 的 MLP，输出 29 维关节位置目标；
- Critic 使用包含仿真特权信息的 113 维观测；
- 支持 TensorBoard、checkpoint、ONNX 和 Viser 浏览器回放；
- 新增多随机种子、并行 episode 自动评测与结果绘图工具。

## 平地基线结果

评测使用 3 个随机种子，每个种子 64 个并行环境，共计 192 个独立的 20 秒 episode。测试保留训练时的摩擦随机化、编码器偏置、质心偏移，以及每 5～6 秒一次的随机推扰。

| 指标 | 结果 |
|---|---:|
| 跌倒率 | 0.0% |
| 完整回合率 | 100.0% |
| Episode return | 53.672 ± 2.696 |
| XY 速度 RMSE | 0.413 ± 0.112 m/s |
| Yaw 速度 RMSE | 0.269 ± 0.045 rad/s |
| 接触足滑移速度 | 0.085 ± 0.029 m/s |
| 相邻动作变化 RMS | 0.216 ± 0.014 |
| 机身倾斜 RMS | 0.053 ± 0.013 |

详细评测方法参见 [EVALUATION.md](EVALUATION.md)。模型 checkpoint 和大体积训练日志不纳入 Git 仓库。

## 评测示例

```bash
python scripts/evaluate.py Unitree-G1-Flat \
  --checkpoint=/path/to/model_10000.pt \
  --num-envs=64 \
  --seeds '(42,43,44)' \
  --device=cuda:0
```

生成 checkpoint 对比图：

```bash
python scripts/plot_evaluation.py \
  --summary-csv=evaluations/Unitree-G1-Flat/<timestamp>/summary.csv
```

## 后续计划

1. 训练并评测 `Unitree-G1-Rough` 基线；
2. 增加斜坡、楼梯、随机高度场和局部低摩擦测试；
3. 比较单帧 MLP、历史帧拼接和 TCN/GRU 状态编码；
4. 评估历史信息对摩擦变化、动作延迟和突发扰动恢复的作用；
5. 完成消融实验、演示视频和技术报告。

## 上游项目

原始项目：[unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)。本仓库保留上游许可证和提交历史。
