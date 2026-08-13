# 项目计划与实验进度

最后更新：2026-08-13（UTC）

## 项目目标

构建一个可复现、可量化评测的 Unitree G1 鲁棒运动控制项目，按以下主线推进：

```text
Flat PPO 基线
  → Rough 复杂地形基线
  → 摩擦/推扰/质量/延迟鲁棒性测试
  → 历史帧、TCN、GRU 时序策略
  → 显式状态重建
  → 消融实验、演示视频与技术报告
```

项目原则：先建立可靠基线和统一评测，再逐项引入方法改进；所有简历结论必须有对应实验数据支撑。

## 总体状态

| 阶段 | 状态 | 主要产物 |
|---|---|---|
| 1. Flat PPO 基线 | 已完成 | checkpoint、ONNX、TensorBoard、192 episodes 评测 |
| 2. Rough PPO 基线 | 进行中 | smoke test、短训练、全量训练、分地形评测 |
| 3. 鲁棒性 benchmark | 未开始 | 摩擦、推扰、质量、延迟测试曲线 |
| 4. 历史观测策略 | 未开始 | Stack、TCN、GRU 对照 |
| 5. 显式状态重建 | 未开始 | 状态估计辅助损失与消融 |
| 6. 项目包装 | 持续进行 | GitHub、报告、视频、简历描述 |

## 已完成：Flat PPO 基线

训练配置：

```text
任务                 Unitree-G1-Flat
并行环境             4096
PPO iterations       10001（索引 0～10000）
每环境每轮采样       24 steps
总采样               983,138,304 steps
训练吞吐             约 56,000 steps/s
训练时间             约 4小时44分
Actor                MLP 98→512→256→128→29
Critic               MLP 113→512→256→128→1
```

192 episodes 正式评测结果：

| 指标 | 结果 |
|---|---:|
| 跌倒率 | 0.0% |
| 完整回合率 | 100.0% |
| Episode return | 53.672 ± 2.696 |
| XY 速度 RMSE | 0.413 ± 0.112 m/s |
| Yaw 速度 RMSE | 0.269 ± 0.045 rad/s |
| 接触足滑移速度 | 0.085 ± 0.029 m/s |

评测工具：`scripts/evaluate.py`、`scripts/plot_evaluation.py`。详细说明见 `EVALUATION.md`。

## 进行中：Rough PPO 基线

### 目标

训练 G1 使用高度扫描通过平地、楼梯、倒楼梯、斜坡、倒斜坡、随机高度场和波浪地形，为后续历史策略提供复杂地形对照组。

### 验收流程

1. **最小 smoke test**：16 个环境、1 iteration。
2. **短训练**：512～1024 个环境、500～1000 iterations。
3. **全量训练**：根据短测吞吐和显存确定环境数，目标 10001 iterations。
4. **统一评测**：至少 3 seeds × 64 episodes，并扩展为按地形类型统计。
5. **回放检查**：抽查楼梯、斜坡和随机高度场动作，确认没有明显奖励漏洞。

### Smoke test 验收条件

- 环境和高度扫描可正常初始化；
- PPO 完成至少一次采样和更新；
- reward/loss 中没有 NaN 或 Inf；
- terrain curriculum 能注册；
- checkpoint 和 ONNX 正常生成；
- GPU 显存、吞吐和磁盘输出符合服务器规范。

### 实验记录

#### R0：Rough 最小 smoke test

状态：已完成（2026-08-13 09:22 UTC）。

计划命令：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> python scripts/train.py Unitree-G1-Rough \
  --env.scene.num-envs=16 \
  --env.scene.terrain.num-envs=16 \
  --agent.max-iterations=1 \
  --agent.save-interval=1 \
  --agent.run-name=g1_rough_smoke \
  --agent.logger=tensorboard \
  --video=False
```

实际使用 GPU 4，运行目录：

```text
logs/rsl_rl/g1_velocity/2026-08-13_09-22-28_g1_rough_smoke
```

控制台日志：

```text
/data/users/yanghao/logs/unitree_rl_mjlab/g1_rough_smoke_20260813.log
```

结果：

| 项目 | 结果 |
|---|---:|
| 并行环境 | 16 |
| PPO iterations | 1 |
| 每轮样本数 | 384 |
| 首轮吞吐 | 221 steps/s |
| Actor 输入 | 285（含 187 维 height scan） |
| Critic 输入 | 300 |
| Actor 输出 | 29 |
| Curriculum terrain level | 3.6875 |
| NaN/Inf | 未发现 |

结论：环境、高度扫描、接触传感器、terrain curriculum、PPO 更新及日志均正常。首次运行编译崎岖地形 CCD 和 raycast Warp 内核，耗时约一分钟；内核现已进入 `/data/users/yanghao/cache/warp`，后续运行会复用缓存。单轮训练不足以完成 episode，因此 reward 和 termination 的 episode 汇总为 0 属正常现象。

#### R1：Rough 短训练

状态：已完成（2026-08-13 09:24～09:44 UTC）。

目的：测量 1024 个并行环境下的吞吐、显存和早期学习趋势，并据此决定全量训练是否使用 4096 个环境。

计划配置：

```text
GPU                  4
并行环境             1024
PPO iterations       500
checkpoint interval  100
日志后端             TensorBoard
```

实际运行目录：

```text
logs/rsl_rl/g1_velocity/2026-08-13_09-24-13_g1_rough_short_1024
```

tmux 与控制台日志：

```text
tmux: g1_rough_short
/data/users/yanghao/logs/unitree_rl_mjlab/g1_rough_short_1024_20260813.log
```

启动后初步资源数据（iteration 7，仅用于测量性能，不代表最终策略）：

| 项目 | 数值 |
|---|---:|
| 吞吐 | 10,568 steps/s |
| 单轮时间 | 约 2.3 s |
| GPU 4 显存 | 5,403 MiB |
| GPU 4 利用率 | 约 79% |
| 预计 500 iterations 用时 | 约 18.5 分钟 |

早期策略尚未学会站立，mean reward 为 -10.47、mean episode length 为 61.55，terrain curriculum 已从初始较高难度自动降至 1.42。这符合课程学习在训练初期降低难度的预期，需在 500 iterations 后判断学习趋势。

最终结果（iteration 499）：

| 项目 | 数值 |
|---|---:|
| 总采样 | 12,288,000 steps |
| 总用时 | 20分10秒 |
| 稳定吞吐 | 约 10,055 steps/s |
| Mean reward | -6.47 |
| Mean episode length | 231.81 steps |
| XY 速度误差 | 0.3659 |
| Yaw 速度误差 | 0.8344 |
| 滑移速度 | 0.2044 m/s |
| Terrain level | 0.0 |
| 最终 checkpoint | `model_499.pt` |

结论：相较 iteration 7，episode length 从 61.55 提升到 231.81，滑移速度从约 0.444 降到 0.204 m/s，说明策略正在学习；但 terrain curriculum 已退到最低等级，完整 20 秒回合仍极少，500 iterations 不能作为最终 Rough 策略。需要全量训练。

#### R2：Rough 并行规模测试

状态：已完成（2026-08-13 09:49 UTC）。

目的：用 4096 环境运行 10 iterations，测量稳定吞吐和显存，并与 R1 的 1024 环境结果比较；不将该短测试作为策略成果。

运行目录：

```text
logs/rsl_rl/g1_velocity/2026-08-13_09-49-59_g1_rough_scale_4096
```

结果：

| 并行环境 | 每轮样本 | 稳定吞吐 | 单轮时间 |
|---:|---:|---:|---:|
| 1024 | 24,576 | 约 10,055 steps/s | 约 2.44 s |
| 4096 | 98,304 | 约 25,300～25,900 steps/s | 约 3.8 s |

结论：4096 环境的总吞吐约为 1024 环境的 2.5 倍，能更充分利用 A100。全量训练选择 4096 环境；按实测速度估计 10001 iterations 需约 10.5～11 小时，总采样约 9.83 亿步。

#### R3：Rough 全量训练

状态：全量训练已重新启动，运行中（2026-08-13 09:56 UTC）。

计划配置：

```text
任务                 Unitree-G1-Rough
GPU                  4
并行环境             4096
PPO iterations       10001
每轮每环境采样       24 steps
checkpoint interval  100
logger               TensorBoard
video                False
预计用时             10.5～11小时
```

首次启动记录：

```text
运行目录  logs/rsl_rl/g1_velocity/2026-08-13_09-51-56_g1_rough_full_4096
停止位置  iteration 3
停止原因  测试 8192 环境吞吐；不是训练错误
```

正式运行记录：

```text
tmux       g1_rough_full
运行目录   logs/rsl_rl/g1_velocity/2026-08-13_09-56-39_g1_rough_full_4096_final
控制台日志 /data/users/yanghao/logs/unitree_rl_mjlab/g1_rough_full_4096_restart_20260813.log
GPU 4显存  约18.5GB
GPU利用率  约87%
```

首次 iteration 的 ETA 包含初始化影响；稳定后预计回到扩展测试测得的约 10.5～11 小时。

#### R2b：8192 环境扩展测试

状态：已完成（2026-08-13 09:53 UTC）。

| 并行环境 | 稳定吞吐 | 单轮时间 | 稳定显存 |
|---:|---:|---:|---:|
| 4096 | 25.3～25.9k steps/s | 约 3.8 s | 约 18.5GB |
| 8192 | 29.7～30.4k steps/s | 约 6.56 s | 约 37.3GB |

结论：8192 环境吞吐只提升约 17%，但每轮样本翻倍、单轮时间增加约 73%。若仍训练 10001 iterations，总采样量会翻倍且预计用时约 18.2 小时；若将 iterations 减半，则 PPO 参数更新次数和 curriculum 时序都发生变化，不能作为与 Flat 公平可比的基线。最终仍使用 4096 环境完成 Rough baseline。

## 下一阶段预案

Rough baseline 完成后，将扩展自动评测为可控测试场景：

- 固定摩擦系数扫描；
- 标准化前/后/侧向推扰；
- 质量和质心偏移；
- 0～5 个控制周期动作延迟；
- 局部低摩擦地块；
- 单帧 MLP、历史拼接、TCN、GRU 对照。

## 文档更新规则

- 每次启动训练前记录任务、GPU、环境数、iterations 和日志路径；
- smoke/短测/全量训练结束后记录吞吐、显存、关键指标和结论；
- 参数或方向发生变化时记录原因，不覆盖旧结论；
- GitHub 只上传源码、配置和清理后的结果，不上传大型 checkpoint、日志或服务器私有路径。
