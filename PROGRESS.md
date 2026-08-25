# 项目计划与实验进度

最后更新：2026-08-19（UTC）

详细实验配置、过程、失败记录和结果表统一维护在 `EXPERIMENT_LOG.md`；本文件保留项目里程碑总览。

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
| 2. Rough PPO 基线 | 已完成 | smoke test、短训练、全量训练、192 episodes 统一评测 |
| 3. 鲁棒性 benchmark | 第一版完成 | 已完成 E001～E007：摩擦、推扰、惯性、执行器、延迟与组合扰动 |
| 3.5 联合域随机化 MLP | 全量训练中 | T001：保持网络不变，训练组合动力学鲁棒性 |
| 4. 历史观测策略 | 未开始 | Stack、TCN、GRU 对照 |
| 5. 显式状态重建 | 未开始 | 状态估计辅助损失与消融 |
| 6. Marathon 高速耐久跑 | 进行中 | 4 帧历史 Actor、速度自适应步态、能耗约束 |
| 7. 项目包装 | 持续进行 | GitHub、报告、视频、简历描述 |

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

状态：已完成（2026-08-13 20:58 UTC）。

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

最终结果：

| 项目 | 数值 |
|---|---:|
| 总采样 | 983,138,304 steps |
| 总用时 | 10小时56分46秒 |
| 最终吞吐 | 约 25,297 steps/s |
| Mean reward（iteration 10000） | 9.21 |
| Mean episode length | 945.72 / 1000 steps |
| 最终 checkpoint | `model_10000.pt` |
| 导出策略 | `policy.onnx` |

#### R4：Rough 192 episodes 正式评测

状态：已完成（2026-08-18 03:20 UTC）。

评测配置：GPU 6，3 个随机种子 `42/43/44`，每个种子 64 个并行环境，共 192 个 episode。关闭 curriculum，保留训练环境中的摩擦力、编码器偏差、质心偏移和周期推扰随机化。

| 指标 | 结果 |
|---|---:|
| 跌倒率 | 2.08% (4/192) |
| 完整 20 秒回合率 | 97.92% (188/192) |
| Episode return | 44.490 ± 8.212 |
| 平均持续时间 | 19.763 ± 1.783 s |
| XY 速度 RMSE | 0.867 ± 0.366 m/s |
| Yaw 速度 RMSE | 0.336 ± 0.100 rad/s |
| 接触足滑移速度 | 0.052 ± 0.029 m/s |
| Action delta RMS | 0.220 ± 0.013 |
| Base tilt RMS | 0.054 ± 0.027 |

结果目录：

```text
evaluations/rough_model10000_192ep_gpu6/
```

结论：Rough 策略在包含复杂地形、域随机化和周期推扰的统一评测中有较高存活率，但 XY 速度跟踪误差明显高于 Flat baseline，后续需要按地形和指令速度分箱，定位误差主要来自高难地形还是高速指令。

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

## 进行中：Marathon 高速耐久跑

### M0：任务实现与 smoke test

状态：已完成（2026-08-18 03:34 UTC）。

新增任务 `Unitree-G1-Marathon`，与 Flat/Rough 基线隔离。首版配置：

| 项目 | 配置 |
|---|---|
| 赛道 | 平地直线 |
| Episode | 60 s |
| Actor | 4 帧历史，392 维输入，MLP 512-256-128 |
| Critic | 单帧特权观测，113 维输入 |
| 初始目标速度 | 0.5～1.5 m/s |
| 最终目标速度 | 1.5～4.0 m/s |
| 步态 | 周期 0.55→0.30 s，支撑比 0.55→0.38 |
| 新增奖励 | 直立门控前进、归一化机械功率 |

Smoke test 使用 GPU 6、16 环境、1 iteration，运行目录：

```text
logs/rsl_rl/g1_velocity/2026-08-18_03-33-45_g1_marathon_smoke_v2
```

结果：384 steps 采样和 PPO 更新完成，202 steps/s，机械功率指标约 741 W，无 NaN/Inf。因仅采样 0.48 s、没有 episode 结束，Episode Reward 汇总为 0 属正常现象。

### M1：1024 环境短训练

状态：已完成（2026-08-18 03:34～03:44 UTC）。使用 GPU 6、1024 环境、500 iterations。

```text
tmux        g1_marathon_short
运行目录    logs/rsl_rl/g1_velocity/2026-08-18_03-34-44_g1_marathon_short_1024
控制台日志  /data/users/yanghao/logs/unitree_rl_mjlab/g1_marathon_short_1024_20260818.log
```

iteration 18 初步数据：约 21,100 steps/s，GPU 6 显存约 1.0 GB，平均 episode length 66.19 steps，机械功率约 685 W。策略尚处于随机动作和早期站立学习阶段，不用此时指标判断最终跑步效果。

最终结果（iteration 499）：

| 项目 | 结果 |
|---|---:|
| 总采样 | 12,288,000 steps |
| 总用时 | 9分钟 |
| 吞吐 | 24,091 steps/s |
| Mean reward | 96.79 |
| Mean episode length | 2980.15 / 3000 steps |
| 足部滑移速度 | 0.511 m/s |
| 落地冲击 | 316.6 N |
| 机械功率代理 | 789.4 W |
| 最终 checkpoint | `model_499.pt` |

重要结论：策略已学会基本不跌倒并跑完 60 s，但本轮尚不能判定为成功马拉松策略。`error_vel_xy=1.5596` 和 `error_vel_yaw=12.3763` 是按 12 s command 窗口归一化后的 episode 累积量；对 60 s 完整回合粗略除以 5，对应平均 XY 误差约 0.31 m/s，但平均 yaw 角速度误差约 2.48 rad/s。目标 yaw 为 0，这表明存在高速旋转的奖励漏洞风险。下一版必须增强 yaw/航向约束、使用世界坐标赛道进度，并做可视化确认后才能启动全量训练。

### M2：自然跑步动作先验

状态：进行中（2026-08-18）。

选择 LAFAN1 作为首个跑步参考数据源，许可为 CC BY-NC-ND 4.0，只用于本地非商业研究，不将原始或重定向数据上传 GitHub。已下载官方 137 MB 压缩包，并提取：

```text
run1_subject2.bvh  run1_subject5.bvh
run2_subject1.bvh  run2_subject4.bvh
sprint1_subject2.bvh  sprint1_subject4.bvh
```

六段数据均为 30 Hz，每段约 238～273 s。外部数据位于 `data_external/`，已写入 `.gitignore`。Hugging Face 的 `GeorgiaTech/g1_lafan1_50hz` 已确认包含 4 段 run 和 2 段 sprint，但 NPZ/manifest 需要账号接受数据条款后才能下载。当前采用公开 BVH + GMR 的 G1 29-DOF 重定向路线，GMR 使用独立项目目录 `/data/users/yanghao/projects/GMR`。

GMR 独立 Python 3.10 环境已建立在 `/data/users/yanghao/envs/humanoid/gmr-py310`。为节省磁盘，没有安装仅由 `KinematicsModel` 使用的完整 PyTorch/CUDA 依赖；GMR 包入口已将该模块改成可选导入，核心 MuJoCo + Mink IK 重定向不受影响。主项目新增 `scripts/retarget_lafan1_headless.py`，用于服务器上无 GUI 地将 BVH 转为 G1 PKL，并保留精确的帧范围和运行摘要。下一步先验证 100 帧，再根据速度、方向稳定性和周期性筛选跑步子片段。

### M2 数据重定向与 tracking 验证（2026-08-18）

- `run1_subject2.bvh` 全量 7135 帧已重定向完成：约 29.5 s，约 242 frame/s，峰值内存约 632 MB；输出位于 `data_external/g1_lafan1_50hz/retargeted/run1_subject2_full.pkl`。
- 自动筛选出的首个候选为原始帧 `[3360, 3450)`（112–115 s）：平均平面速度约 2.07 m/s，直线度 0.998，根高度标准差约 2.9 cm，3 s 内位移约 6.13 m。
- 50 Hz tracking 数据：`src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz`，149 帧、29 关节、30 个刚体，所有数组均为有限值。
- 运动学预览：`src/assets/motions/g1/lafan1_run1_subject2_112s_115s.mp4`。已检查关键帧，可见躯干前倾、对侧摆臂与交替跑步支撑。`scripts/csv_to_npz.py` 已补充 MP4 实际写盘和相机跟随机器人配置。
- 64 环境、10 iteration 冒烟训练已完成：`logs/rsl_rl/g1_tracking/2026-08-18_07-21-29_lafan1_run_smoke_64_tb`；约 1690 steps/s，mean reward 从 -1.03 改善至 -0.67。默认 WandB 因未登录会失败，已固定使用本地 TensorBoard。
- 策略架构仍是 PPO Actor-Critic MLP：Actor `160→512→256→128→29`，Critic `286→512→256→128→1`，不是 Transformer。
- 第一阶段正式训练脚本：`scripts/run_lafan_tracking_stage1.sh`，计划使用 GPU 6、4096 并行环境、5000 iteration，每 500 iteration 保存 checkpoint。完成后先做视频与定量评估，再决定续训或修改参考片段/奖励。

第一阶段已于 2026-08-18 07:22 UTC 启动：tmux 会话 `g1_lafan_tracking_stage1`，控制台日志 `/data/users/yanghao/logs/unitree_rl_mjlab/g1_lafan_tracking_stage1_20260818.log`，运行目录 `logs/rsl_rl/g1_tracking/2026-08-18_07-22-34_lafan1_run_stage1_4096`。启动后实测约 68k steps/s、每 iteration 约 1.44 s，GPU 6 显存约 2.8 GiB、利用率约 76%，预计约 2 小时完成 5000 iteration。

评估脚本为 `scripts/evaluate_lafan_tracking_checkpoint.sh <model_N.pt>`：它会在 GPU 6 上以 1 个环境录制 500 帧（10 s）策略实际控制视频，视频输出到对应 checkpoint 所在训练目录的 `videos/play/`。首个目标 checkpoint 为 `model_500.pt`；评估时重点检查是否持续前进、是否出现摔倒/拖脚、摆臂与双腿是否同步，以及与参考视频的步态相位是否一致。

### M2 第一阶段 tracking 训练完成（2026-08-18）

- 已完成 `5000` iteration（最终 checkpoint 按零起始编号为 `model_4999.pt`），累计采样 `491,520,000` 环境步；最终训练吞吐约 `66k steps/s`。
- 最终一轮：mean reward `35.44`，mean episode length `493.83/500`；平均刚体位置误差 `0.1048 m`、关节位置误差 `0.5164 rad`、根位置误差 `0.1880 m`。这些是训练时带随机化的统计量，不能单独视为泛化结论。
- 已完成无终止的 10 s 策略控制回放（500 帧），视频为 `logs/rsl_rl/g1_tracking/2026-08-18_07-22-34_lafan1_run_stage1_4096/videos/play/rl-video-step-0.mp4`。人工抽查关键帧：机器人保持站立并呈交替腿/摆臂跑步相位，未见立即摔倒；后续应以多随机种子和扰动评估检验鲁棒性。
- `scripts/evaluate_lafan_tracking_checkpoint.sh` 已固定 `CUDA_VISIBLE_DEVICES=6`，并为当前服务器 Python 环境显式设置其自带 C++ 运行库路径，避免 ICU 与系统 `libstdc++` 的 ABI 冲突。

### M3 鲁棒性 benchmark：第一批结果（2026-08-19）

新增 `scripts/evaluate_tracking_robustness.py`，统一输出逐 episode CSV、场景汇总 CSV 和 JSON 元数据。评测严格限制为一次 149 帧参考片段（实际控制 147 steps，2.94 s），排除了 `MotionCommand` 在片段结束时重写机器人状态造成的伪恢复。初始化时还需主动同步一次 root-relative body target，否则首步会被 `ee_body_pos` 错误终止；脚本已处理该框架细节。

正式矩阵使用最终 `model_4999.pt`、3 seeds × 64 episodes，共 1728 episodes。结果目录：`evaluations/Unitree-G1-Tracking/robustness_full_20260819`。

| 场景 | 成功率 | Body MPKPE RMSE | 足滑 | 根位移 |
|---|---:|---:|---:|---:|
| clean，摩擦 1.0 | 100.0% | 0.041 m | 0.195 m/s | 5.989 m |
| 摩擦 0.2 | 38.0% | 0.108 m | 1.333 m/s | 3.165 m |
| 摩擦 0.4 | 100.0% | 0.045 m | 0.208 m/s | 5.963 m |
| 摩擦 0.6 | 100.0% | 0.042 m | 0.196 m/s | 5.981 m |
| 摩擦 0.8 | 100.0% | 0.041 m | 0.196 m/s | 5.988 m |
| 横向速度推扰 0.25 m/s | 100.0% | 0.042 m | 0.197 m/s | 5.962 m |
| 横向速度推扰 0.50 m/s | 100.0% | 0.044 m | 0.197 m/s | 5.936 m |
| 横向速度推扰 0.75 m/s | 100.0% | 0.046 m | 0.201 m/s | 5.903 m |
| 横向速度推扰 1.00 m/s | 100.0% | 0.048 m | 0.206 m/s | 5.865 m |

首个明确失败边界是极低摩擦：摩擦 0.2 时 192 个 episode 中只有 73 个正常结束，主要终止原因为末端位置和根高度偏差；足滑扩大到 clean 的约 6.8 倍。当前策略对训练范围内及略强的横向速度推扰表现较强，但根位置 RMSE 从 clean 的 0.135 m 随 1.0 m/s 推扰增至 0.433 m，说明“未摔倒”不等于保持精确轨迹。

随后完成摩擦临界区间细化，共 3 个场景 × 3 seeds × 64 episodes，结果目录为 `evaluations/Unitree-G1-Tracking/robustness_friction_refine_20260819`：

| 摩擦系数 | 成功率 | Body MPKPE RMSE | 足滑 | 根位移 |
|---:|---:|---:|---:|---:|
| 0.25 | 77.1% | 0.077 m | 0.687 m/s | 4.829 m |
| 0.30 | 93.2% | 0.059 m | 0.364 m/s | 5.594 m |
| 0.35 | 97.9% | 0.049 m | 0.245 m/s | 5.857 m |

结果显示 `0.25～0.35` 是当前策略的低摩擦失效过渡带：摩擦越低，足滑、轨迹误差和提前终止同步增加。合并后的曲线为 `evaluations/Unitree-G1-Tracking/robustness_full_20260819/robustness_curves_refined.png`。成功口径复核后要求 time-out 且没有同时触发失败终止，因此摩擦 0.2 的严格成功率由旧汇总的 38.5% 修正为 73/192，即 38.0%。

### M3 局部低摩擦地块完成（2026-08-19）

评测脚本已支持 `local_friction_<value>`：根据参考轨迹自动计算前进方向，按左右脚的空间位置分别切换脚部碰撞几何摩擦，策略不能直接看到摩擦值。地块位于起点前方 2.25～3.75 m，半宽 1.0 m。正式矩阵为 6 个摩擦档位 × 3 seeds × 64 episodes，共 1152 episodes，最终目录：`evaluations/Unitree-G1-Tracking/robustness_local_patch_full_20260819_v2`。

| 局部摩擦 | 成功率 | 越过地块率 | 地块内足滑 | 离开后 root RMSE |
|---:|---:|---:|---:|---:|
| 0.05 | 21.9% | 23.4% | 2.139 m/s | 0.417 m |
| 0.10 | 75.5% | 75.5% | 1.206 m/s | 0.267 m |
| 0.20 | 100.0% | 100.0% | 0.304 m/s | 0.141 m |
| 0.25 | 100.0% | 100.0% | 0.203 m/s | 0.121 m |
| 0.30 | 100.0% | 100.0% | 0.175 m/s | 0.114 m |
| 0.35 | 100.0% | 100.0% | 0.173 m/s | 0.110 m |

局部摩擦 0.10 已进入明显失效区，0.05 时大多数 episode 无法离开地块；随着摩擦降低，地块内足滑和离开后的轨迹误差同步增加。下一步测试质量/质心偏移和动作延迟，再冻结完整 baseline benchmark。

### M3 质量、负载与质心 benchmark 完成（2026-08-19）

评测新增整体质量缩放、torso 点负载和 torso 局部 y 方向质心偏移。整体质量使用 pseudo-inertia 同步缩放质量与转动惯量；当前 G1 MJCF 总质量 33.341 kg，torso 为 7.818 kg。第一轮和边界细化各 2112 episodes，结果目录分别为 `robustness_inertial_full_20260819` 和 `robustness_inertial_refine_20260819`。

关键边界：整体质量 1.20×/1.25×/1.30× 的成功率为 88.0%/29.2%/3.6%；torso 负载 6/7/8/9 kg 为 96.4%/79.2%/38.5%/13.0%。质心偏移到 ±12 cm 均未提前终止，但 -12 cm 的 root RMSE 达 0.233 m，高于 +12 cm 的 0.191 m，显示方向不对称。训练随机化建议从质量 0.8×～1.25×、torso 负载 0～8 kg、质心 ±12 cm 开始并使用 curriculum。合并曲线为 `evaluations/Unitree-G1-Tracking/robustness_inertial_full_20260819/inertial_robustness.png`。

注意：极端质量或负载下 episode 很早终止，因此其全程平均 RMSE 不一定继续单调上升；判断失效必须结合成功率、episode 长度和根位移，不能单独看 RMSE。

### M3 执行器、延迟与组合 benchmark 完成（2026-08-19）

- 电机强度：保持 60% 标称力矩时成功率 99.5%，主要失效带为 56%～48%。
- 动作延迟：30 ms 成功率 100%，35 ms 为 99.0%；45/50/55 ms 快速降至 55.7%/29.2%/3.6%，主要失效带为 35～55 ms。
- 组合扰动：单独施加局部摩擦 0.20、电机 58%、延迟 35 ms、负载 6 kg、侧推 0.25 m/s 时，成功率均为 95.8%～100%；但局部摩擦+延迟为 59.4%，电机+延迟为 18.8%，电机+延迟+负载为 0%。这证明当前策略存在显著的组合动力学失配，而不是只对某一个极端变量脆弱。
- 第一版 E001～E007 baseline benchmark 已冻结。下一步 T001 训练联合域随机化 MLP 基线，随后 T002 加短历史观测，并使用完全相同的 E001～E007 矩阵做对照。

结果目录：`robustness_motor_full_20260819`、`robustness_motor_refine_20260819`、`robustness_delay_full_20260819`、`robustness_delay_refine_20260819`、`robustness_combo_full_20260819`。详细表格、失败记录和复现命令见 `EXPERIMENT_LOG.md`。

### T001 联合域随机化 MLP 启动（2026-08-19）

新增 `Unitree-G1-Tracking-Robust`，网络、观测、奖励与 PPO 均保持不变，只在每个 episode 联合随机化摩擦、电机强度、动作延迟和 torso 负载，并使用四阶段课程逐步扩大范围。有效 smoke 和 512 环境 × 200 iterations 短训练均已完成；短训练最终 mean episode length 为 478.36/500、mean reward 31.90。

4096 环境、5000 additional iterations 的全量微调最初在 GPU 6 启动，运行到约 iteration 6257。双卡等总批量测试表明 GPU 5+7 明显快于单卡，因此已从落盘的 `model_6000.pt` 切换为双卡续训；每卡 2048 环境，全局 rollout 规模仍为 4096 环境。新运行目录为 `logs/rsl_rl/g1_tracking/2026-08-19_06-28-23_t001_joint_dr_full_2gpu_resume6000`，当前 tmux 为 `g1_tracking_t001_2gpu`。训练完成后将以相同 E001～E007 矩阵与原始 baseline 对照，重点检查 `combo_motor_delay` 和 `combo_actuation_payload`。

最新状态（2026-08-19）：T001 全量训练已正常完成，最终 checkpoint 为 `model_9999.pt`，累计约 3.93 亿环境步，两个训练 worker 均无错误退出。完整 E001～E007 对照评测已于 11:09 UTC 启动，使用 GPU 5/6/7 三个 worker 并行，但每个场景仍严格保持 3 seeds × 64 episodes。会话为 `t001_eval_w1`、`t001_eval_w2`、`t001_eval_w3`，输出写入 `evaluations/Unitree-G1-Tracking-Robust/t001_e*_20260819`。评测结束后将自动汇总与原 baseline 的差异，并据此进入下一项实验。

T001 全部 74 个匹配场景已评测完成：平均成功率提升 21.94 个百分点，没有场景下降超过 1 个百分点。最关键的 `combo_motor_delay` 从 18.8% 提升到 99.0%，`combo_actuation_payload` 从 0% 提升到 77.6%，五项全组合从 0% 提升到 65.6%；clean 仍为 99.5%，Body MPKPE 由 4.1 cm 降到 3.5 cm。T001 已形成有效的联合域随机化强基线。

T002 已进入短训练阶段：新增 `Unitree-G1-Tracking-Robust-History`，只给 actor 的本体感觉和动作增加 4 帧短历史，输入从 160 维变为 439 维，critic、PPO、奖励和随机化保持不变。通过权重/normalizer/Adam 状态扩展从 T001 等价初始化，随机输入最大输出差仅 `2.38e-6`；16 环境 smoke 已通过。下一步为 2048 环境 × 200 iterations 短训练及 E007 回归，通过后启动全量训练。

T002 短训练最终停留在约 5 steps 的 episode length，没有从初期崩溃恢复，已停止进入全量阶段并作为失败实验保留。后续若恢复该方向，需针对 optimizer 迁移和历史编码器单独设计，而不是直接扩大输入后继续 PPO。

### S001 Sprint-v2 启动（2026-08-20）

新增独立任务 `Unitree-G1-Sprint-v2`，修复 Marathon-v1 的旋转奖励漏洞：世界 `+X` 直线进度、0.9 m 半宽跑道、世界横向位置/速度、朝向和 yaw rate 约束，出界或倒跑提前终止。速度课程为 `0.8～1.8 → 1.5～2.5 → 2.2～3.2 → 2.8～4.0 m/s`；训练早期只使用平地和轻摩擦随机化，高速稳定后再加入组合扰动。

首次 smoke 因本地终止函数导入命名空间错误在环境创建前失败，修正后 16 环境 × 2 iterations 有效 smoke 已完成：Actor 392 维、Critic 113 维、22 项奖励、4 项终止和两个 curriculum 均正常，PPO 与 checkpoint 保存无异常。下一步运行 GPU 5、2048 环境 × 500 iterations 短训练，并通过固定速度回放排查旋转、漂移和动作异常。

Sprint-v2 短训练已于 02:14 UTC 在 GPU 5 启动，tmux 为 `g1_sprint_v2_short`，目录为 `logs/rsl_rl/g1_velocity/2026-08-20_02-14-50_sprint_v2_short_2048_500`。早期吞吐约 38k steps/s；旧 Marathon 策略在新跑道约束下频繁出界，当前处于纠正旋转和漂移的适应阶段。

500 iterations 短训完成后，相同 1.5 m/s、64 episodes 对照显示：世界前进速度由旧策略 0.93 提升到 1.28 m/s，yaw rate RMS 由 1.77 降到 0.39 rad/s，平均前进距离由 1.97 增至 5.65 m；但出界率仍为 98.4%。因此暂不扩速，继续在 stage 0 的 0.8～1.8 m/s 训练 1000 iterations，先解决直线完赛率。

stage 0 续训已于 02:42 UTC 在 GPU 5 启动，tmux `g1_sprint_v2_stage0`，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_02-42-10_sprint_v2_stage0_2048_1000`。早期吞吐约 39k steps/s，预计约 21 分钟完成；结束后自动沿用固定 1.5 m/s benchmark 决定是否解锁下一速度阶段。

stage 0 续训已于 03:05 UTC 正常完成，最终 checkpoint 为 `model_1997.pt`，累计训练到约 49.15M environment steps，本段耗时 22 分 34 秒、平均吞吐约 36.2k steps/s，未出现 NaN/Inf 或异常退出。自动评测最初因 tmux 的前缀匹配把 `g1_sprint_v2_stage0_eval` 误识别为仍存活的训练会话而持续等待；脚本现已改用 `-t =session_name` 精确匹配并完成全部评测。

固定 seed 42、每档 64 episodes 的结果：命令 1.0/1.5/1.8 m/s 下，实际世界前进速度分别为 0.946/1.366/1.634 m/s，跌倒率均为 0%，yaw-rate RMS 为 0.310/0.343/0.376 rad/s，足滑为 0.217/0.281/0.337 m/s，平均前进距离为 7.89/8.05/6.75 m。但出界率仍为 100%/98.4%/100%，因此暂不解锁 stage 1。与 `model_998.pt` 的固定 1.5 m/s 结果相比，`model_1997.pt` 的速度 1.280→1.366 m/s、存活 4.40→5.89 s、距离 5.65→8.05 m、yaw RMS 0.388→0.343、足滑 0.301→0.281，说明训练有效但没有解决长期横向漂移。

关键诊断：当前 actor 输入没有世界赛道横向位置或世界航向误差；奖励虽然惩罚出界和偏航，但策略无法从自身观测辨认自己位于跑道哪一侧，也无法直接辨认相对世界 `+X` 的航向。这是部分可观测性问题，继续单纯增加 stage 0 iterations 的收益有限。下一步先新增赛道相对观测（横向位置、航向误差及世界横向/前向速度），采用输入层扩展方式继承现有策略，再做 smoke 和短训对照；只有固定 1.5 m/s 出界率明显下降后才进入更高速课程。

### S002 Sprint-v3 可观测赛道闭环（2026-08-20）

已新增独立任务 `Unitree-G1-Sprint-v3`。旧 392 维历史观测顺序保持不变，末尾加入当前帧 5 维赛道状态：归一化横向位置、航向 cos/sin、归一化世界前向/横向速度，Actor 输入变为 397 维；Critic 和 PPO 结构不变。`scripts/expand_sprint_checkpoint_track_state.py` 已将 `model_1997.pt` 的第一层和观测归一化统计扩展，新输入列及 Adam 动量均以零初始化，因此转换瞬间策略动作与 v2 等价。提速阈值从 absolute iteration 2000 延后到 2500，先预留约 500 iterations 学习纠偏，再根据固定 1.5 m/s 出界率决定是否进入 1.5～2.5 m/s。

16 环境 × 2 iterations smoke 已通过，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_06-14-22_sprint_v3_track_obs_smoke_16`。环境确认 Actor `(397,)`、Critic `(113,)`、新增 `track_state (5,)`，成功加载转换后的 `model_1997.pt` 和 Adam 状态；课程保持 0.8～1.8 m/s，两轮采样与更新无异常。

纠偏适应训练已于 06:15 UTC 在 GPU 5 启动：tmux `g1_sprint_v3_track_adapt`，2048 环境 × 500 additional iterations，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_06-15-32_sprint_v3_track_adapt_2048_500`，日志 `logs/benchmarks/g1_sprint_v3_track_adapt_20260820.log`。早期吞吐约 34k steps/s，ETA 约 11.5 分钟，课程保持 0.8～1.8 m/s。

本轮于 06:27 UTC 完成，自动评测于 06:29 完成，因此之后 GPU 5 为空闲而不是训练丢失。最佳 `model_2300.pt` 在固定 1.5 m/s 下跌倒率 0%、出界率 67.2%、完整回合率 32.8%、实际速度 1.430 m/s、平均存活 11.08 s、距离 16.07 m、足滑 0.335 m/s。相比 v2 出界率 98.4% 已明显改善，证明赛道观测有效，但未达到≤20%的提速门槛；后期 checkpoint 出界率回升，显示 PPO 适应过程振荡，自动门控正确地没有启动 stage 1。

下一轮新增 `Unitree-G1-Sprint-v3-Lane`，从最佳 `model_2300.pt` 继续：保留原二次横向位置惩罚，新增接近 0.9 m 边界时四次增长的 barrier，以及离开中心线方向的横向速度惩罚；学习率由 1e-3 降至 3e-4、熵系数由 0.01 降至 0.005，以减少后期策略振荡。首个提速节点调整到 iteration 2800，本轮先用 500 iterations 完成低速纠偏，达标后再提速。

S003 正式训练已于 07:48 UTC 在 GPU 5 启动：PID 2206578、tmux `g1_sprint_v3_lane_adapt`，2048 环境 × 500 additional iterations，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_07-48-17_sprint_v3_lane_adapt_2048_500`。iteration 2322 吞吐约 35.0k steps/s、ETA 约 11 分钟；GPU 5 显存约 1.58 GiB，新增两项奖励均产生非零训练信号，课程仍为 0.8～1.8 m/s。

资源复核显示 2048 环境仅占 GPU 5 约 1.58 GiB、利用率约 62%、吞吐约 35k steps/s。为利用剩余容量，将从已落盘 `model_2400.pt` 切换到 4096 环境续训到相同 absolute iteration 2799，并比较切换后的吞吐；原 2048 进程在安全 checkpoint 后停止，未覆盖任何结果。

4096 环境续训已于 07:52 UTC 启动：PID 2209195、tmux `g1_sprint_v3_lane_4096`，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_07-52-59_sprint_v3_lane_adapt_4096_resume2400`。实测显存约 2.95 GiB、GPU 利用率约 74%，稳定吞吐约 61.6k steps/s，相比 2048 环境约 35k 提升约 76%；单 iteration 由约 1.4 s 增至 1.6 s，但每轮样本翻倍。自动评测门控已改绑新会话和新目录。

4096 环境续训已于 08:05 UTC 正常完成，最终 `model_2799.pt`，本段 39.322M environment steps、耗时 11分52秒；后段吞吐约 54k steps/s。自动评测于 08:06 完成。最佳 checkpoint 不是最终模型，而是 `model_2700.pt`：固定 1.5 m/s 下跌倒率 0%、出界率 50%、完整 20 s 回合率 50%、实际速度 1.358 m/s、平均存活 12.67 s、距离 17.54 m、足滑 0.319 m/s。相较上一轮最佳 `model_2300` 的出界 67.2% 和完整率 32.8% 继续改善，但仍未达到出界≤20%的门槛，因此自动门控没有启动 stage 1；GPU 5 已释放。

### E0 多 seed 赛道偏置诊断（2026-08-20）

评测器已新增航向误差 RMS/绝对均值（度）、终止横向位置/速度/航向、横向漂移斜率 `dy/dx`，并分别汇总 +Y/-Y 出界率；新增 `--clean` 可关闭 actor observation corruption 及摩擦、编码器偏置、质心随机化。4 环境 clean smoke 已通过。正式 E0 使用 `model_2700.pt`、固定 1.5 m/s、seeds 11/23/42/67/89、每 seed 128 episodes，分别运行 clean 与训练随机化，共 1280 episodes。

E0 已完成。clean 640 episodes 全部跑满 20 s、无跌倒/出界，速度 1.332 m/s、航向 RMS 2.98°、`dy/dx=-0.0167`；DR 640 episodes 无跌倒，出界率 42.19%、完整率 57.81%、速度 1.360 m/s、航向 RMS 7.04°、`dy/dx=-0.0755`。DR 的 270 次出界中 266 次为 -Y、4 次为 +Y，即 98.5% 向同一侧，且五个 seed 一致，明确存在随机化下暴露的单侧策略偏置。后续必须加入镜像/对称约束；为保持单变量实验顺序，先执行 E1 修复 phase，再在 E2 的三维命令训练中加入 symmetry。

E1 已新增 `Unitree-G1-Sprint-v4-AdaptivePhase`：Actor 的 phase 与 gait reward 共用 `running_gait_period()`，速度从 0.5→4.0 m/s 时周期同步由 0.55→0.30 s，消除旧固定 0.6 s 时钟冲突。将从 `model_2700.pt` 用 4096 环境训练 300 iterations，每 50 iterations 保存；学习率 1e-4、clip 0.1、desired KL 0.005、entropy 0.004，并保持 0.8～1.8 m/s 以隔离 phase 变量。

E1 的 16 环境 × 2 iterations smoke 已通过，目录 `logs/rsl_rl/g1_velocity/2026-08-20_09-25-53_sprint_e1_adaptive_phase_smoke_16`；模型、优化器和自适应 phase 均正常，无 NaN/Inf。

E1 正式训练已于 09:27 UTC 启动：GPU 5、4096 环境、300 additional iterations，tmux `g1_sprint_e1_phase`，目录 `logs/rsl_rl/g1_velocity/2026-08-20_09-27-50_sprint_e1_adaptive_phase_4096_300`。iteration 2725 吞吐约 58k steps/s、ETA 约 8 分钟，显存约 2.95 GiB。完成后先用 seed 42 筛选每 50 iterations 的 checkpoint，再对最佳模型自动执行与 E0 相同的 clean/DR 5-seed 评测。

E1 训练于 09:37 UTC 正常完成，最终 `model_2999.pt`，29.491M environment steps、耗时 8分58秒；自动评测于 09:43 完成。结果需要分两部分理解：仅把原 `model_2700.pt` 放入自适应 phase 环境、不更新权重时，clean 航向 RMS 2.98→2.40°，DR 出界率 42.19→35.31%、完整率 57.81→64.53%、航向 RMS 7.04→6.59°，说明相位一致化本身有效；但从 `model_2750` 起所有微调 checkpoint 均明显退化，出界率约 98.4%～100%，最终航向 RMS 接近 17°。因此 E1 输出保留原 `model_2700.pt` 配合自适应 phase，不采用训练后 checkpoint。下一阶段应在宽跑道/无窄边界终止的三维命令技能训练中重新适应 phase，并加入左右对称约束。

### E2-A 三维命令基线（2026-08-20）

新增 `Unitree-G1-Sprint-E2A-Command`：50% 环境强制 `vy=yaw=0`，其余按课程扩展为 `vy ±0.10/0.20/0.30 m/s`、yaw `±0.15/0.30/0.50 rad/s`；世界赛道 shaping 已移除，安全边界放宽至 ±2 m、严重后退阈值放宽至 2 m。新增 `--weights-only-resume`，只加载 `model_2700.pt` 的 Actor/Critic 与 normalizer，明确重置 Adam、iteration 和 curriculum counter。PPO 使用 lr 1e-4、clip 0.1、KL 0.005、entropy 0.004、8 mini-batches。首次 smoke 因新增命令类插入位置造成缩进错误，在导入阶段失败且未创建环境；修正后的 16-env × 2-iteration smoke `2026-08-20_10-04-16_sprint_e2a_command_smoke_16_v2` 已通过，日志确认从 iteration 0 开始且课程第一档正确。

E2-A 正式训练已于 10:05 UTC 启动：GPU 5、4096 环境 × 600 iterations，tmux `g1_sprint_e2a`，目录 `logs/rsl_rl/g1_velocity/2026-08-20_10-05-45_sprint_e2a_command_4096_600`。iteration 17 吞吐约 50～58k steps/s、ETA 约 20 分钟，显存约 2.58 GiB。评测器已扩展分轴 vx/vy/yaw RMSE、方向正确率与实际 body-frame 均速；完成后自动对所有 50-iteration checkpoint 运行 straight、vy±0.3、yaw±0.5 矩阵并选优。

E2-A 于 10:26 UTC 正常完成：600 iterations、58.982M environment steps、耗时 20分08秒，最终 `model_599.pt`；自动矩阵于 10:38 完成。严格门槛未通过，自动选出的综合最佳为 `model_350.pt`：五类固定命令测试均无跌倒，直线速度 1.515 m/s（相对起点保留 110.8%），横移两方向最低正确率 97.7%、最差 RMSE 0.133 m/s，说明 `vy` 技能已经学会；但转向两方向最低正确率仅 50.5%、最差 yaw-rate RMSE 0.791 rad/s，距离 95%/0.20 门槛很远。最终 `model_599.pt` 的转向也未继续改善（52.2%、0.895 rad/s）。

对 E2-A 任务定义复核后修正了原结论：低层任务仍保留世界坐标 ±2 m `outside_lane` 和后退终止，正确执行 1.5 m/s、0.5 rad/s 转向时约 3 秒即可产生 2.79 m 世界横移，因此“忽略 yaw、保持直行”反而能延长 episode；指数 yaw 奖励在 0.5 rad/s 初始误差下也已接近饱和。这是优先于 symmetry 的任务冲突。下一步改为 E2-B0：从已学会直线/横移的 `model_350.pt` 权重级热启动，删除世界跑道终止，使用 35% 直线/25% 横移/30% 转向/10% 组合的互斥采样，yaw 奖励改为 weight 3.0、std 0.5 并增加 -0.5×平方误差，角动量惩罚排除 z 分量；先做 4096×200、每25轮保存的 ±0.15/±0.30 探针，通过后才进入 B1，symmetry 延后到 E2-C。

E2-B0 独立任务 `Unitree-G1-Sprint-E2B0-Yaw-Probe` 已完成配置与 16-env × 2-iteration smoke（`2026-08-20_10-54-44_sprint_e2b0_yaw_probe_smoke_16_v2`）。运行时确认只有 `time_out/fell_over/illegal_contact` 三项终止、18 项奖励、四类命令采样和正确课程；Actor/Critic 仍为 397/113 维，`model_350.pt` 权重级加载成功，无 NaN/Inf。首次 smoke 因沙箱看不到 CUDA 在创建环境前失败，不属于代码或实验失败；获得 GPU 权限后的 v2 有效完成。

4096×200 正式探针及自动评测脚本均已就绪，但 10:57 UTC 的后台启动请求因平台权限审批服务连接失败而未执行；已复核没有残留训练/tmux 会话，也没有生成正式 run。需在下一次获得明确 GPU 后台运行批准后直接启动，无需改动实验配置。

### E2-B 网络方案复核与实验设计（2026-08-24）

公开方案进一步支持“先修任务、后 symmetry”的顺序。HUGWBC 将低层任务命令明确设为机体坐标 `(vx, vy, yaw-rate)`，early termination 只包含躯干/其他连杆触地和大倾角，并使用速度课程；其正则项只惩罚 roll/pitch 角速度。Unitree 官方 velocity policy 同样直接暴露三维速度命令，部署范围覆盖 `vy ±0.5、yaw ±1.0`。Booster Gym 将 x/y/yaw 跟踪分开，并同样只惩罚 x/y 角速度。这说明世界跑道边界不应进入低层转向技能任务。

正式实验改成可归因的两步探针，而非一次混合所有修改：B0-A 从同一 `model_350.pt` 出发，只修复终止冲突并使用分类命令采样，保持 E2-A yaw reward；B0-B 从同一起点加入宽核 yaw reward、非饱和平方误差和 z 角动量豁免。两组各 4096×100、每25保存，先只学 ±0.15；用 clean seed42 筛选后以 clean/DR 3 seeds×64 复评。若达到 yaw 方向≥90%、RMSE≤0.30、失败≤2%、直线保留≥95%，选择较优组再训练100轮至 ±0.30。这样可区分改善来自任务定义还是奖励形状。

启动前还需处理一个架构细节：当前 E2-A/B0 Actor 仍含 5 维世界 `track_state`。低层控制器最终不应依赖世界赛道状态；为兼容 `model_350` 的 397 维输入，B0 探针先对这 5 维做零掩码并保留网络尺寸，B1 再通过输入层映射正式移除。课程不再固定到点自动升级，而由阶段评测门控。B1 扩到 yaw ±0.50 和组合命令，B1 通过后 E2-C 才加入 mirror/symmetry，之后接高层世界赛道控制器。

E2-B0 A/B 已实现：`Unitree-G1-Sprint-E2B0A-Task-Fix` 只修复终止、分类采样和 `track_state` 零掩码，保留 E2-A yaw reward；`Unitree-G1-Sprint-E2B0B-Reward-Fix` 在相同基础上增加宽核/平方 yaw reward 并豁免 z 角动量。两组 16-env×2 smoke 于 02:32 UTC 同时通过，均从 `model_350.pt` weights-only 成功加载、保持397/113维、yaw范围仅±0.15，且无跌倒/非法接触。

两组正式实验于 2026-08-24 02:33 UTC 并行启动：B0-A 使用空闲 GPU 4，目录 `2026-08-24_02-33-07_sprint_e2b0a_task_fix_4096_100`；B0-B 使用空闲 GPU 5，目录 `2026-08-24_02-33-07_sprint_e2b0b_reward_fix_4096_100`。各为4096 envs×100 iterations、每25保存，同 seed、同 PPO、同 `model_350` 起点。自动评测会话 `g1_sprint_e2b0_eval` 已启动，等待两组结束后测试 straight、vy±0.3、yaw±0.15 并生成 A/B 决策。

两组训练均于02:37 UTC正常完成，各9.830M steps、约3分24秒，最终 `model_99.pt`，训练期间无跌倒/非法接触。自动评测首次因固定命令逻辑只清零 `rel_straight`、破坏四类概率和而失败；修复为评测时100% combined后，第二次因失败残留目录的防覆盖检查退出，已将残留目录保留归档。最终A/B矩阵通过GPU4/5并行补跑完成，完整决策为 `evaluations/Unitree-G1-Sprint-E2B0-AB/e2b0_ab_seed42/decision_parallel.json`。

B0-A未通过：最佳 `model_99` 无失败、横移正确率98.5%/RMSE0.084，但yaw最低方向正确率40.3%、RMSE0.638，说明仅删除冲突终止不足以让策略学会转向。B0-B有显著且单调的正向信号：`model_0→25→50→75→99` 的yaw RMSE约0.669→0.456→0.371→0.344→0.297，最佳方向正确率59.5%，实际均值为目标±0.15下约+0.074/-0.106 rad/s；无跌倒/非法接触，横移正确率98.4%/RMSE0.097。直线实际1.451 m/s，虽相对零掩码后的过冲基准1.658仅保留87.5%，但更接近1.5 m/s命令且绝对值已高于后续1.44门槛，因此“retention失败”主要是旧判据对过冲基准不合理。结论：任务修复必要但不充分，宽核+非饱和yaw信号有效；当前仍是偏弱且振荡的初级转向，不能升级到±0.30。

按失败分支直接启动B0-B2 yaw-focus：奖励、终止、yaw±0.15和其余PPO均不变，只把命令比例从35/25/30/10改为20%直线、10%横移、60%纯yaw、10%组合，使yaw-active从40%提高到70%。从B0-B `model_99.pt` weights-only启动。16-env×2 smoke `2026-08-24_02-56-47_sprint_e2b0b2_yaw_focus_smoke_16`通过；正式4096×100于02:57 UTC在GPU5启动，目录`2026-08-24_02-57-13_sprint_e2b0b2_yaw_focus_4096_100`，初始约52.4k steps/s、无跌倒/非法接触，自动评测会话已挂载。

B0-B2训练于03:00 UTC正常完成，最终`model_99.pt`。自动评测在完成straight/lat±后，于创建yaw_pos目录后无错误栈退出；空目录已保留为`yaw_pos_015_failed_empty`，正负yaw随后在GPU4/5并行补跑并生成完整decision。最佳`model_99`：failure0%，straight1.454 m/s（相对B起点100.25%），lateral方向98.57%/RMSE0.0859；瞬时yaw方向最低65.96%、RMSE0.250，未达到原90%方向门槛，但相较B的59.48%/0.297继续改善。更关键的是目标±0.15下episode平均yaw已达+0.128/-0.121 rad/s，即稳态增益约0.86/0.81，且正负方向均无跌倒。

因此当前不再解释为“不会转”，而是“整体转向正确、步态内瞬时躯干yaw振荡较大”。原方向正确率按50 Hz瞬时yaw计数，会把正常左右摆动也记为错误；下一步先新增低通/单步态周期平均yaw、累计heading变化率和响应时间指标，再决定是否需要继续训练。若滤波后跟踪合格，则B0通过并进入±0.30；若滤波后仍振荡过大，再把yaw reward改为滤波角速度并加入轻量yaw加速度/jerk正则，避免继续单纯增加采样比例。

### E2-B0-B2 滤波转向审计与 B3 启动（2026-08-24）

评测器已加入0.3秒低通yaw、低通方向正确率/RMSE、累计heading变化率、稳态增益、80%响应时间和超调量。B2 `model_99` 在DR seed42、正负各64 episodes下通过门控：+0.15的滤波方向99.55%、RMSE0.0506、增益0.835；-0.15为96.45%、0.0613、0.787；两向失败率均0%、前进速度均约1.451 m/s。累计heading-rate为+0.131/-0.121 rad/s，与滤波结果一致，证明瞬时65.96%的低方向分数主要由步态内摆动造成，而非没有执行转向。

按门控结果新增 `Unitree-G1-Sprint-E2B0B3-Yaw030`：仅把yaw训练范围从±0.15扩至±0.30，保留70% yaw-active采样、奖励、终止、随机化与PPO不变，从B2 `model_99` weights-only继续。16-env×2 smoke `2026-08-24_03-15-35_sprint_e2b0b3_yaw030_smoke_16`已通过，运行时确认课程范围±0.30、无跌倒/非法接触/NaN。正式计划为4096×100、每25保存，完成后按同一滤波指标评测±0.15与±0.30，并回归straight和vy±0.30。

B3正式训练已于03:17 UTC在GPU5启动，运行目录 `2026-08-24_03-17-31_sprint_e2b0b3_yaw030_4096_100`。iteration 1时GPU利用率约95%、显存5.17 GiB，训练无跌倒/非法接触，预计约7～8分钟完成。第一次tmux启动曾生成 `03-16-47` 目录和 `model_0.pt` 后会话退出，该不完整run保留为工程失败记录，不参与比较；有效正式run为 `03-17-31`。

B3有效run已于03:24 UTC完成：100 iterations、9.830M environment steps、最终 `model_99.pt`，末轮平均回合600/600、fall/illegal均0。DR seed42的7类固定命令、每类64 episodes回归全部通过：直线速度1.448 m/s；横移最低方向98.51%、最大RMSE0.0804 m/s；yaw±0.15最低滤波方向97.74%、最大RMSE0.0552；yaw±0.30最低滤波方向99.60%、最大RMSE0.0763、稳态增益0.836～0.851。448个episodes无失败，B3正式验收，下一阶段扩至yaw±0.50短探针。

E2-B1 `Unitree-G1-Sprint-E2B1-Yaw050` 已实现，只把yaw范围扩大到±0.50，其余保持B3不变。16-env×2 smoke `2026-08-24_03-29-36_sprint_e2b1_yaw050_smoke_16`通过；正式4096×100于03:30 UTC在GPU5启动，目录 `2026-08-24_03-30-01_sprint_e2b1_yaw050_4096_100`。iteration5约47.4k steps/s、无fall/illegal，ETA约3.4分钟。

E2-B1于03:33 UTC完成：100 iterations、9.830M steps、最终`model_99.pt`，末轮reward71.14、episode length600/600，fall/illegal均0。DR seed42的9类固定命令、每类64 episodes已完成：yaw±0.50滤波方向99.83%/99.81%、RMSE0.0813/0.1130、稳态增益0.894/0.828，说明完整转向技能已学会；直线1.442 m/s，横移最低方向98.80%、最大RMSE0.0743，均保留。

严格左右一致性门槛未通过：yaw +0.15/-0.15稳态增益为1.037/0.705，负向小角速度明显偏弱；±0.30也为0.940/0.794。所有576个episodes虽零失败，但该幅值依赖的不对称会影响后续高层赛道纠偏。B1保留`model_99.pt`，下一步进入E2-C symmetry，不再扩大yaw范围或继续堆yaw奖励。

### E2-C 最后一次 yaw 相关实验：镜像增强（2026-08-24）

已明确停止规则：E2-C只运行一次4096×100，无论是否把左右差距完全消除，之后都转入高层赛道控制器；残余不对称由高层左右增益补偿，不再继续调yaw奖励。

实现使用RSL-RL原生 symmetry data augmentation，不启用额外mirror-loss权重。新增G1矢状面镜像：29关节左右交换，roll/yaw轴取反、pitch轴保号；机体角速度按伪向量变换，重力/线速度按极向量变换，`vy/yaw_cmd`取反，相位平移半周期，Critic足部与接触力同步交换。397维Actor、113维Critic和29维动作均通过“镜像两次逐元素恢复”测试，TensorDict原始+镜像批次测试也通过。

第一次16-env smoke在更新前因RSL-RL向agent配置注入实时`_env`对象、YAML无法序列化而退出；未产生算法更新。`train.py`现将静态记录快照与运行时配置分离。修复后的smoke `2026-08-24_03-56-11_sprint_e2c_symmetry_smoke_16_v2`完成2轮，日志确认symmetry augmentation生效，无fall/illegal/NaN。

E2-C正式训练于03:56 UTC在GPU5启动：4096×100、每25保存，从B1 `model_99.pt` weights-only开始，目录 `2026-08-24_03-56-37_sprint_e2c_symmetry_aug_4096_100`。iteration1约45.4k steps/s、无fall/illegal，ETA约4分钟。

E2-C于04:00 UTC正常完成：100 iterations、9.830M steps、最终`model_99.pt`，末轮reward71.98、episode length600/600、fall/illegal均0。DR seed42的9类固定命令、每类64 episodes全部完成，576 episodes零失败；直线速度1.442 m/s，横移最低方向98.96%、最大RMSE0.0715 m/s。

镜像增强改善明确但没有完全消除低速不对称。yaw正负稳态增益差在±0.15从0.332降至0.267，在±0.30从0.146降至0.112，在±0.50从0.066降至0.036；±0.50增益为0.876/0.841，已非常接近。E2-C没有造成生存退化，采用其`model_99.pt`作为最终低层三维命令策略。yaw专项到此结束，下一阶段正式进入高层赛道闭环；残余左右差异由高层不同方向增益补偿。

### S1 第一档提速：1.5～2.2 m/s（2026-08-24）

已从 E2-C `model_99.pt` 启动第一档提速，而不是直接跳到 3～4 m/s。新任务 `Unitree-G1-Sprint-S1-Speed220` 将主命令范围设为 1.5～2.2 m/s，并让 20% 环境回放 1.0～1.8 m/s，避免高速微调遗忘原有稳定步态；直线/横移/转向/组合采样比例为 60/10/20/10，继续保留三维命令能力和 symmetry augmentation。

奖励新增非饱和前向速度平方误差（weight -0.5），躯干姿态目标改为随速度从前倾 2° 平滑增加到 8°；其余安全、足滑、落地冲击和功率约束保持不变。PPO 仍使用 weights-only、lr 1e-4、clip 0.1、KL 0.005、entropy 0.004、8 mini-batches，避免继承旧 Adam 状态。

16-env×2 smoke `2026-08-24_06-26-55_sprint_s1_speed220_smoke_16` 已通过，运行时确认速度范围、回放、新奖励和前倾指标均生效，且无跌倒/非法接触。正式 run 于 06:28 UTC 在 GPU 4 启动：`2026-08-24_06-28-40_sprint_s1_speed220_4096_300`，4096×300、每25轮保存；iteration 5 吞吐约46k steps/s，ETA约11分钟。此前 GPU 5/4 各有一次受限执行环境看不到 CUDA 的启动失败，均在环境构造前退出、未发生训练更新；改用服务器 GPU 权限后正常运行。

完成后对 `model_0/25/.../299` 独立评测固定 1.5、2.0、2.2 m/s，并回归 `vy±0.3`、`yaw±0.5`。选优优先级为失败率、实际速度/命令速度、速度 RMSE，其次才是训练 reward；同时检查足滑、落地冲击、功率和动作平滑，防止用摔倒前冲或滑行换速度。

S1 正式训练于 06:39 UTC 完成：300 iterations、29.491M environment steps，最终 `model_299.pt`，训练末轮 reward 70.60、episode length 600/600、fall/illegal 均0；13个25轮间隔 checkpoint 与 ONNX 均完整落盘。2.2 m/s 的 seed42×64 全 checkpoint 筛选中，`model_25` 起全部零失败；`model_299` 实际2.126 m/s、速度RMSE 0.359、足滑0.417 m/s，为综合最佳。

`model_299` 的回归矩阵共384 episodes全部零失败：1.5/2.0 m/s实际1.466/1.932 m/s；横移±0.3最低方向98.90%、最大RMSE0.0697 m/s；yaw±0.5低通方向最低99.78%、最大RMSE0.0837 rad/s、稳态增益+0.935/-0.949。相比E2-C的yaw±0.5增益+0.876/-0.841没有遗忘，反而更接近命令。

另以seeds 11/23/67/89、每seed64 episodes复评2.2 m/s：256/256全部跑满，零跌倒/非法接触；实际速度2.131±0.028 m/s，达到命令96.9%，速度RMSE0.358±0.007，足滑0.419 m/s。S1正式通过，采用`model_299.pt`，下一步可进入重叠速度区间2.0～2.8 m/s的S2；评测决策写入 `evaluations/Unitree-G1-Sprint-S1-Speed220/decision.json`。

### S2 第二档提速：2.0～2.8 m/s（2026-08-25）

S1通过后新增独立任务`Unitree-G1-Sprint-S2-Speed280`，从S1最佳`model_299.pt`权重级热启动。主速度范围扩到2.0～2.8 m/s，25%环境回放1.5～2.2 m/s；直线/横移/转向/组合仍为60/10/20/10，vy±0.30、yaw±0.50及symmetry均保留。速度误差、足滑、冲击、功率等权重暂不改变，以保证本阶段结果主要归因于速度课程。

前倾目标沿连续曲线从1.5 m/s的2°扩展到2.8 m/s的12°，避免回放旧速度时突然改变姿态目标。16-env×2 smoke `2026-08-25_02-59-20_sprint_s2_speed280_smoke_16`已通过：速度范围、回放、前倾、weights-only reset和symmetry均生效，无fall/illegal/NaN。

正式run于02:59 UTC在GPU1启动：`2026-08-25_02-59-46_sprint_s2_speed280_4096_400`，4096×400、每25轮保存，PPO继续使用lr1e-4、clip0.1、KL0.005、entropy0.004、8 mini-batches。iteration5吞吐约46k steps/s、ETA约15分钟；早期探索fall/illegal已从iteration2～4的少量事件降至iteration5各0.0417，尚未形成持续退化。完成后筛选2.8 m/s checkpoint，再评2.0/2.4/2.8及低速、横移、yaw回归；S2门槛为实际速度≥命令90%、失败≤2%，并重点约束足滑不超过约0.55 m/s。

S2正式训练于03:14 UTC正常完成：400 iterations、39.322M environment steps，最终`model_399.pt`；第50轮后所有训练采样均保持600/600、fall/illegal为0，末轮reward65.93。训练足滑由iteration25的0.558降至末轮0.534 m/s，实际/目标平均前倾由5.46°/8.94°改善到8.12°/8.93°，没有后期策略崩坏。

固定2.8 m/s的17-checkpoint筛选共1088 episodes全部零失败。选择最终`model_399`：seed42实际速度2.677 m/s（95.6%跟踪）、vx RMSE0.462、足滑0.501 m/s、return75.11；它不是瞬时均速最高的模型，但同时具有最低RMSE、最低足滑与最高回报，综合优于早期checkpoint。

`model_399`的7类回归共448 episodes全部零失败：1.5/2.0/2.4 m/s实际1.504/1.950/2.312 m/s；横移最低方向98.87%、最大RMSE0.0778；yaw最低低通方向99.71%、最大RMSE0.1177，+/-0.5稳态增益1.002/1.127。负yaw有约12.7%过冲，但仍显著满足RMSE≤0.20门槛，不阻塞本阶段验收。

DR seeds11/23/67/89、每seed64 episodes的2.8 m/s复评全部跑满：实际2.684±0.032 m/s、跟踪率95.9%、vx RMSE0.461±0.008、足滑0.508±0.025 m/s，零fall/illegal。S2正式通过并采用`model_399.pt`；结构化结果为`evaluations/Unitree-G1-Sprint-S2-Speed280/decision.json`。下一档建议2.5～3.4 m/s，并继续把足滑0.55 m/s作为硬门槛。

### S3 第三档提速：2.5～3.4 m/s（2026-08-25）

S2通过后新增`Unitree-G1-Sprint-S3-Speed340`，从S2最佳`model_399.pt`权重级热启动。主速度2.5～3.4 m/s，30%环境回放2.0～2.8 m/s，三维命令比例、vy±0.30、yaw±0.50、symmetry及所有奖励权重保持不变。前倾目标在2.0～3.4 m/s间由6°连续增至16°，与S2重叠区间基本连续。

16-env×2 smoke `2026-08-25_03-31-15_sprint_s3_speed340_smoke_16`通过，运行时确认3.4 m/s上限、约13°平均目标前倾和weights-only reset均正确，无fall/illegal/NaN；短样本足滑一度0.825 m/s，已标为本阶段主要风险。

正式run于03:31 UTC在GPU1启动：`2026-08-25_03-31-41_sprint_s3_speed340_4096_400`，4096×400、每25保存，PPO保持lr1e-4、clip0.1、KL0.005、entropy0.004、8 mini-batches。任务切换初期iteration4～7出现较多重置；到iteration38平均回合已恢复579/600、fall0.375/illegal0.0833（训练窗口统计），但足滑仍约0.666 m/s。训练继续完成；最终只有3.4 m/s跟踪≥90%、独立评测失败≤2%且足滑≤0.55 m/s才通过，否则从最佳速度checkpoint进入S3-slip专项，不再直接扩到4.0 m/s。

## 文档更新规则

- 每次启动训练前记录任务、GPU、环境数、iterations 和日志路径；
- smoke/短测/全量训练结束后记录吞吐、显存、关键指标和结论；
- 参数或方向发生变化时记录原因，不覆盖旧结论；
- GitHub 只上传源码、配置和清理后的结果，不上传大型 checkpoint、日志或服务器私有路径。
