# Unitree G1 鲁棒跑步实验日志

最后更新：2026-08-19（UTC）

## 1. 研究问题

本项目研究：使用 LAFAN1 跑步动作训练出的 Unitree G1 motion-tracking 策略，在摩擦突变、外部推扰、模型参数误差和控制延迟下能否继续稳定、准确地跑步；以及历史状态和域随机化能否提升这种鲁棒性。

当前策略是 PPO Actor-Critic，actor/critic 均为 MLP，不是 Transformer。当前阶段先建立可复现 benchmark，找到失效边界，再训练改进策略，避免在没有明确问题和对照指标时盲目增加模型复杂度。

## 2. 固定实验资产

| 项目 | 内容 |
|---|---|
| 代码仓库 | `/data/users/yanghao/projects/unitree_rl_mjlab` |
| 任务 | `Unitree-G1-Tracking` |
| 参考动作 | `src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz` |
| 动作长度 | 149 帧，50 Hz |
| 基线 checkpoint | `logs/rsl_rl/g1_tracking/2026-08-18_07-22-34_lafan1_run_stage1_4096/model_4999.pt` |
| 评测并行数 | 正式评测每个 seed 64 个环境 |
| 正式随机种子 | 42、43、44 |
| Python | `/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python` |
| 主要 GPU | 服务器空闲卡，当前评测固定映射为进程内 `cuda:0` |

## 3. 评测原则

1. 同一组实验固定 checkpoint、动作片段、随机种子和 episode 数，只改变待测试因素。
2. 每个 episode 只评测一次参考片段，不跨越片段循环边界。
3. `MotionCommand` 到达片段末尾会重写机器人状态，因此评测只运行 147 个控制步（2.94 s），避免把状态重置误判为策略恢复。
4. 初始化后主动更新一次 root-relative body target，避免第一步被错误的 `ee_body_pos` 终止。
5. 同时保存逐 episode 数据、场景汇总和 JSON 元数据，不能只依据单个视频得出结论。
6. 成功的定义是坚持到时间上限；同时必须检查轨迹误差、足滑和位移，因为“没有摔倒”不等于精确跟踪。

## 4. 指标解释

| 指标 | 含义 | 趋势 |
|---|---|---|
| success rate | 未触发姿态/位置等提前终止的比例 | 越高越好 |
| Body MPKPE RMSE | 各关键刚体相对参考动作的位置误差 | 越低越好 |
| root position RMSE | 躯干根节点相对参考轨迹的位置误差 | 越低越好 |
| foot slip | 脚与地面接触时的水平滑动速度 | 越低越好 |
| root displacement | episode 内机器人根节点移动距离 | 需结合参考距离判断 |
| action delta RMS | 相邻控制动作的变化幅度 | 通常越低越平滑 |
| patch foot slip | 脚位于局部低摩擦区且接触地面时的滑动速度 | 越低越好 |
| post-patch reach rate | episode 中真正越过低摩擦区域的比例 | 越高越好 |
| post-patch root RMSE | 离开局部低摩擦区后的根轨迹误差 | 越低越好 |

## 5. 实验记录

### E001：全局摩擦与横向推扰基线

- 日期：2026-08-19
- 状态：完成
- 目的：确定原始策略的基本鲁棒性和明显失败条件。
- 配置：9 个场景 × 3 seeds × 64 episodes，共 1728 episodes。
- 输出：`evaluations/Unitree-G1-Tracking/robustness_full_20260819`

| 场景 | 成功率 | Body MPKPE | 足滑 | 根位移 |
|---|---:|---:|---:|---:|
| clean，摩擦 1.0 | 100.0% | 0.041 m | 0.195 m/s | 5.989 m |
| 全局摩擦 0.2 | 38.0% | 0.108 m | 1.333 m/s | 3.165 m |
| 全局摩擦 0.4 | 100.0% | 0.045 m | 0.208 m/s | 5.963 m |
| 全局摩擦 0.6 | 100.0% | 0.042 m | 0.196 m/s | 5.981 m |
| 全局摩擦 0.8 | 100.0% | 0.041 m | 0.196 m/s | 5.988 m |
| 横向推扰 0.25 m/s | 100.0% | 0.042 m | 0.197 m/s | 5.962 m |
| 横向推扰 0.50 m/s | 100.0% | 0.044 m | 0.197 m/s | 5.936 m |
| 横向推扰 0.75 m/s | 100.0% | 0.046 m | 0.201 m/s | 5.903 m |
| 横向推扰 1.00 m/s | 100.0% | 0.048 m | 0.206 m/s | 5.865 m |

结论：全局摩擦 0.2 是明确失败条件；横向推扰 1.0 m/s 虽未造成摔倒，但 root RMSE 从 0.135 m 增加到 0.433 m，说明跟踪精度明显下降。旧 summary 只检查 time-out，曾将一个同时触发 time-out 和 `anchor_pos` 的 episode 计为成功；按后续统一的严格口径，成功数为 73/192，即 38.0%。

### E002：全局低摩擦临界区细化

- 日期：2026-08-19
- 状态：完成
- 目的：确定摩擦 0.2～0.4 之间的失效过渡带。
- 配置：3 个场景 × 3 seeds × 64 episodes，共 576 episodes。
- 输出：`evaluations/Unitree-G1-Tracking/robustness_friction_refine_20260819`

| 摩擦 | 成功率 | Body MPKPE | 足滑 | 根位移 |
|---:|---:|---:|---:|---:|
| 0.25 | 77.1% | 0.077 m | 0.687 m/s | 4.829 m |
| 0.30 | 93.2% | 0.059 m | 0.364 m/s | 5.594 m |
| 0.35 | 97.9% | 0.049 m | 0.245 m/s | 5.857 m |

结论：`0.25～0.35` 是当前策略的低摩擦失效过渡带，后续局部低摩擦实验应覆盖这一范围，并加入更强的 0.2 场景。

### E003：局部低摩擦地块

- 日期：2026-08-19
- 状态：完成
- 目的：模拟正常路面中突然出现一小块湿滑区域，测量单脚踩滑、双脚经过和离开地块后的恢复能力。
- 设计：参考起点前方 2.25 m 开始，长度 1.5 m，左右半宽 1.0 m；地块以外摩擦为 1.0。
- 实现方式：从参考动作起终点自动计算水平前进方向，再根据左右脚 site 在“沿轨迹/垂直轨迹”坐标系中的位置，分别更新对应 7 个脚部 collision geom 的切向摩擦。某只脚进入区域时只降低该脚摩擦，另一只脚不受影响；策略不直接获得摩擦值。
- 正式场景：`local_friction_0p05`、`local_friction_0p1`、`local_friction_0p2`、`local_friction_0p25`、`local_friction_0p3`、`local_friction_0p35`。加入 0.05 和 0.10 是因为局部区域持续时间短，0.2 的 smoke 没有提前终止，需要更强条件才能寻找局部失效边界。
- 新增指标：地块暴露时间、地块内接触足滑、离开地块后的 root position RMSE。
- 验证顺序：静态检查 → 4 环境 smoke → 检查脚部摩擦确实按位置切换 → 3 seeds × 64 正式评测 → 更新本节结果。

失败记录 1：第一次 smoke 假设动作沿世界 `+x` 前进，但该 LAFAN1 片段实际主要沿 `+y` 前进，因此地块暴露时间为 0。没有使用这组结果做性能结论。修正为根据参考轨迹自动计算前进方向，使同一实现可用于不同朝向的动作片段。

有效 smoke：4 环境、seed 42。摩擦 0.2 和 0.3 均为 100% 成功，平均地块暴露约 0.99 s；区域内足滑分别为 0.264 m/s 和 0.179 m/s，离开地块后的 root RMSE 分别为 0.144 m 和 0.121 m。结果目录：`evaluations/Unitree-G1-Tracking/robustness_local_patch_smoke_20260819_v2`。

失败记录 2：第一轮正式矩阵完成后发现两个统计边界问题：（1）最后一步同时触发 time-out 和失败终止时，旧逻辑只检查 time-out，可能误记为成功；（2）未能离开地块的 episode 没有 post-patch 样本，旧逻辑会把恢复误差写成 0。已将成功改为“time-out 且没有失败终止”，新增 `reached_post_patch` 和 post-patch reach rate，未离开地块的恢复误差记为 NaN 并从条件均值中排除。第一轮目录 `robustness_local_patch_full_20260819` 仅用于调试，不作为最终正式结果。

最终正式矩阵：6 个局部摩擦场景 × 3 seeds × 64 episodes，共 1152 episodes。结果目录：`evaluations/Unitree-G1-Tracking/robustness_local_patch_full_20260819_v2`。

正式运行命令（服务器物理 GPU 6 映射为进程内 `cuda:0`）：

```bash
env CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/evaluate_tracking_robustness.py Unitree-G1-Tracking \
  --checkpoint logs/rsl_rl/g1_tracking/2026-08-18_07-22-34_lafan1_run_stage1_4096/model_4999.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --num-envs 64 --seeds "(42,43,44)" \
  --scenarios "('local_friction_0p05','local_friction_0p1','local_friction_0p2','local_friction_0p25','local_friction_0p3','local_friction_0p35')" \
  --output-dir evaluations/Unitree-G1-Tracking/robustness_local_patch_full_20260819_v2
```

绘图命令：

```bash
/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/plot_local_patch_robustness.py \
  --summary-file evaluations/Unitree-G1-Tracking/robustness_local_patch_full_20260819_v2/summary.csv
```

| 局部摩擦 | 成功率 | 越过地块率 | 地块内足滑 | Body MPKPE | 离开后 root RMSE | 根位移 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 21.9% | 23.4% | 2.139 m/s | 0.081 m | 0.417 m | 3.629 m |
| 0.10 | 75.5% | 75.5% | 1.206 m/s | 0.062 m | 0.267 m | 5.155 m |
| 0.20 | 100.0% | 100.0% | 0.304 m/s | 0.045 m | 0.141 m | 5.934 m |
| 0.25 | 100.0% | 100.0% | 0.203 m/s | 0.043 m | 0.121 m | 5.959 m |
| 0.30 | 100.0% | 100.0% | 0.175 m/s | 0.042 m | 0.114 m | 5.970 m |
| 0.35 | 100.0% | 100.0% | 0.173 m/s | 0.042 m | 0.110 m | 5.977 m |

终止原因：摩擦 0.05 的 192 个 episode 中，42 个正常完成，72 个触发 `ee_body_pos`、51 个触发 `anchor_pos`、27 个同时触发两者；摩擦 0.10 有 145 个正常完成，其余主要为末端位置或根位置偏差。摩擦 0.20 以上全部正常完成。

结论：局部地块比全局低摩擦更能区分“短暂踩滑后恢复”和“持续打滑”。当前策略在局部摩擦 0.20 时能够通过，但 0.10 已进入明显失效区，0.05 大多无法离开地块。随着摩擦降低，地块内足滑和离开后的轨迹误差同步升高。后续训练应把 0.05～0.30 的局部摩擦随机化纳入训练，并重点优化打滑后的恢复，而不是只优化全局低摩擦下的存活。

结果曲线：`evaluations/Unitree-G1-Tracking/robustness_local_patch_full_20260819_v2/local_patch_robustness.png`。

## 6. 后续实验队列

| 编号 | 实验 | 状态 | 目的 |
|---|---|---|---|
| E004 | 质量与质心偏移 | 完成 | 验证负载和模型误差鲁棒性 |
| E005 | 电机强度变化 | 完成 | 验证执行器输出误差鲁棒性 |
| E006 | 动作延迟 | 完成 | 验证通信和执行延迟鲁棒性 |
| E007 | 组合扰动 | 完成 | 验证真实复杂条件下的泛化 |
| T001 | 域随机化鲁棒 MLP | 完成 | 建立改进训练基线 |
| T002 | MLP + 短历史观测 | 短训练中 | 验证历史状态是否帮助识别打滑与动力学失配 |
| T003 | 隐状态重建/时序编码 | 待开始 | 估计摩擦、负载、延迟等隐变量 |
| A001 | 对照与消融实验 | 待开始 | 证明提升来源，而非训练随机性 |

### E004：质量、负载与质心偏移

- 日期：2026-08-19
- 状态：完成
- 目的：检验策略对机器人模型标定误差、背负负载和左右不对称负载的适应能力。
- 控制变量：所有场景保持摩擦 1.0，不施加外部推扰，不加入编码器偏差；每个场景只改变一种惯性因素。
- 标称质量：当前 G1 MJCF 总质量为 33.341 kg，`torso_link` 为 7.818 kg。因此 5/10/15 kg 躯干负载分别约为机器人总质量的 15.0%/30.0%/45.0%，但由于负载集中在躯干，其效果不能等同于全身均匀增重。
- 整体质量：`mass_scale_0p8`、`mass_scale_1p2`、`mass_scale_1p4`。使用物理一致的 pseudo-inertia 变换，同时缩放质量和转动惯量。
- 躯干负载：`payload_5kg`、`payload_10kg`、`payload_15kg`。把负载建模为位于 torso COM 的点质量，因此增加质量但不额外增加转动惯量。
- 不对称质心：`com_y_pos_0p03`、`com_y_neg_0p03`、`com_y_pos_0p06`、`com_y_neg_0p06`，分别表示躯干局部 y 轴左右偏移 3 cm 和 6 cm。
- 验证顺序：静态检查 → 每类至少一个场景 smoke → clean 加 10 个扰动场景 × 3 seeds × 64 episodes 正式评测 → 分组绘图和结论。

有效 smoke：4 环境、seed 42。`mass_scale_1p4` 和 `payload_15kg` 均为 0% 成功，平均根位移分别为 1.477 m 和 1.737 m，主要触发末端位置误差；`com_y_pos_0p06` 为 100% 成功，根位移 6.066 m。结果目录：`evaluations/Unitree-G1-Tracking/robustness_inertial_smoke_20260819`。

第一轮正式矩阵：clean + 10 个扰动场景 × 3 seeds × 64 episodes，共 2112 episodes。结果目录：`evaluations/Unitree-G1-Tracking/robustness_inertial_full_20260819`。

| 场景 | 成功率 | Body MPKPE | 根 RMSE | 足滑 | 根位移 |
|---|---:|---:|---:|---:|---:|
| clean | 100.0% | 0.041 m | 0.135 m | 0.195 m/s | 5.990 m |
| 整体质量 0.8× | 100.0% | 0.030 m | 0.138 m | 0.213 m/s | 6.018 m |
| 整体质量 1.2× | 88.0% | 0.077 m | 0.155 m | 0.251 m/s | 5.607 m |
| 整体质量 1.4× | 0.0% | 0.127 m | 0.174 m | 0.409 m/s | 1.539 m |
| 躯干负载 5 kg | 100.0% | 0.066 m | 0.157 m | 0.233 m/s | 6.008 m |
| 躯干负载 10 kg | 3.6% | 0.114 m | 0.191 m | 0.396 m/s | 2.175 m |
| 躯干负载 15 kg | 0.0% | 0.134 m | 0.185 m | 0.422 m/s | 1.453 m |
| 质心 y = +3 cm | 100.0% | 0.041 m | 0.134 m | 0.195 m/s | 5.995 m |
| 质心 y = -3 cm | 100.0% | 0.042 m | 0.147 m | 0.198 m/s | 5.968 m |
| 质心 y = +6 cm | 100.0% | 0.042 m | 0.144 m | 0.194 m/s | 6.013 m |
| 质心 y = -6 cm | 100.0% | 0.042 m | 0.166 m | 0.198 m/s | 5.953 m |

阶段判断：整体质量的失效边界位于 1.2×～1.4×，集中躯干负载边界位于 5～10 kg；负 y 方向质心偏移造成的根误差大于同幅正 y，但 ±6 cm 尚未造成提前终止。下一轮细化整体质量 1.25/1.30/1.35×、负载 6/7/8/9 kg，以及质心 ±9/±12 cm。

边界细化使用 11 个场景 × 3 seeds × 64 episodes，共 2112 episodes。结果目录：`evaluations/Unitree-G1-Tracking/robustness_inertial_refine_20260819`。

| 场景 | 成功率 | Body MPKPE | 根 RMSE | 足滑 | 根位移 |
|---|---:|---:|---:|---:|---:|
| 整体质量 1.25× | 29.2% | 0.098 m | 0.170 m | 0.328 m/s | 3.881 m |
| 整体质量 1.30× | 3.6% | 0.111 m | 0.171 m | 0.368 m/s | 2.263 m |
| 整体质量 1.35× | 0.0% | 0.118 m | 0.171 m | 0.394 m/s | 1.626 m |
| 躯干负载 6 kg | 96.4% | 0.075 m | 0.169 m | 0.264 m/s | 5.941 m |
| 躯干负载 7 kg | 79.2% | 0.086 m | 0.185 m | 0.305 m/s | 5.498 m |
| 躯干负载 8 kg | 38.5% | 0.103 m | 0.203 m | 0.352 m/s | 4.263 m |
| 躯干负载 9 kg | 13.0% | 0.110 m | 0.199 m | 0.392 m/s | 2.923 m |
| 质心 y = +9 cm | 100.0% | 0.043 m | 0.163 m | 0.193 m/s | 6.031 m |
| 质心 y = -9 cm | 100.0% | 0.044 m | 0.196 m | 0.197 m/s | 5.939 m |
| 质心 y = +12 cm | 100.0% | 0.045 m | 0.191 m | 0.192 m/s | 6.048 m |
| 质心 y = -12 cm | 100.0% | 0.046 m | 0.233 m | 0.198 m/s | 5.928 m |

最终结论：整体均匀增重的主要失效带为 1.20×～1.30×，其中成功率从 88.0% 降到 3.6%；集中在躯干的点负载主要失效带为 6～9 kg，说明负载位置会显著改变影响，不能只看总质量比例。±12 cm 质心偏移没有造成提前终止，但负 y 偏移的根 RMSE 明显更高，显示当前跑步动作和策略存在方向不对称性。训练随机化可先使用质量 0.8×～1.25×、躯干负载 0～8 kg 和质心 ±12 cm，并用 curriculum 逐步扩大，避免一开始大量采样必失败条件。

解释注意：质量 1.35×～1.40×、负载 10～15 kg 的 episode 很早终止，所以其“终止前平均 root RMSE”可能不会继续上升。严重程度应以成功率、episode 长度和根位移为主，RMSE 只在相近存活时长下直接比较。

正式矩阵复现命令：

```bash
env CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/evaluate_tracking_robustness.py Unitree-G1-Tracking \
  --checkpoint logs/rsl_rl/g1_tracking/2026-08-18_07-22-34_lafan1_run_stage1_4096/model_4999.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --num-envs 64 --seeds "(42,43,44)" \
  --scenarios "('clean','mass_scale_0p8','mass_scale_1p2','mass_scale_1p4','payload_5kg','payload_10kg','payload_15kg','com_y_pos_0p03','com_y_neg_0p03','com_y_pos_0p06','com_y_neg_0p06')" \
  --output-dir evaluations/Unitree-G1-Tracking/robustness_inertial_full_20260819
```

边界细化只需将 `--scenarios` 替换为：

```text
('mass_scale_1p25','mass_scale_1p3','mass_scale_1p35','payload_6kg','payload_7kg','payload_8kg','payload_9kg','com_y_pos_0p09','com_y_neg_0p09','com_y_pos_0p12','com_y_neg_0p12')
```

合并曲线：`evaluations/Unitree-G1-Tracking/robustness_inertial_full_20260819/inertial_robustness.png`。

### E005：电机强度变化

- 日期：2026-08-19
- 状态：完成
- 目的：模拟电机峰值力矩下降、模型中的力矩上限误差和硬件性能衰减。
- 实现：保持 PD 增益不变，同时按比例缩放全部关节执行器的正负 effort limit；策略不能看到缩放比例。
- 计划档位：`motor_scale_0p9`、`0p8`、`0p7`、`0p6`、`0p5`，分别保留标称最大力矩的 90%～50%。
- 控制变量：摩擦 1.0、无推扰、无额外负载、无动作延迟。
- 验证顺序：最强 0.5× smoke → clean 加 5 档正式评测 → 如有需要细化失效边界。

失败记录 1：第一次 smoke 使用 `actuator_names=(".*",)`，配置解析得到 29 个单关节控制索引，但 `dr.effort_limits` 按 6 个 actuator group 访问，初始化时发生越界，物理仿真尚未开始。已改用 `SceneEntityCfg("robot")` 的默认 group slice，一次选择全部执行器组；失败目录 `robustness_motor_smoke_20260819` 不作为结果。

正式与细化结果目录分别为 `robustness_motor_full_20260819` 和 `robustness_motor_refine_20260819`，每个目录均为 6 场景 × 3 seeds × 64 episodes，即各 1152 episodes。

| 电机力矩上限 | 成功率 | 根 RMSE | 足滑 | 根位移 |
|---:|---:|---:|---:|---:|
| 90% | 100.0% | 0.140 m | 0.193 m/s | 5.974 m |
| 80% | 100.0% | 0.146 m | 0.193 m/s | 5.950 m |
| 70% | 100.0% | 0.155 m | 0.192 m/s | 5.922 m |
| 60% | 99.5% | 0.174 m | 0.200 m/s | 5.871 m |
| 58% | 96.9% | 0.177 m | 0.205 m/s | 5.763 m |
| 56% | 89.6% | 0.185 m | 0.222 m/s | 5.445 m |
| 54% | 72.4% | 0.193 m | 0.259 m/s | 4.834 m |
| 52% | 54.2% | 0.198 m | 0.289 m/s | 4.107 m |
| 50% | 33.9% | 0.203 m | 0.320 m/s | 3.345 m |
| 48% | 17.7% | 0.197 m | 0.349 m/s | 2.595 m |
| 45% | 1.0% | 0.192 m | 0.374 m/s | 1.704 m |

结论：当前策略在 60% 标称峰值力矩下基本稳定，主要失效带为 56%～48%。训练随机化可先覆盖 60%～100%，再通过 curriculum 下探到约 52%，避免初始训练被大量不可完成 episode 主导。

### E006：动作延迟

- 日期：2026-08-19
- 状态：完成
- 目的：测试从策略给出关节位置目标到执行器使用该目标之间的延迟容忍度。
- 实现：用 `DelayedActuatorCfg` 包装原有 6 组位置执行器，只延迟 position target；物理步长为 5 ms，因此延迟严格量化为 5 ms 的整数倍。
- 计划档位：10、20、30、40、60、80 ms；策略控制周期为 20 ms，因此分别对应 0.5～4 个控制周期。
- 控制变量：摩擦 1.0、100% 电机力矩、无负载、无推扰。

80 ms smoke（4 episodes）为 0% 成功，证明延迟配置已实际生效。正式粗扫使用 clean 加 6 个延迟档位 × 3 seeds × 64 episodes，共 1344 episodes；结果目录为 `evaluations/Unitree-G1-Tracking/robustness_delay_full_20260819`。

| 动作延迟 | 控制周期倍数 | 成功率 | 根 RMSE | 足滑 | 根位移 |
|---:|---:|---:|---:|---:|---:|
| 0 ms | 0.00× | 100.0% | 0.135 m | 0.195 m/s | 5.989 m |
| 10 ms | 0.50× | 100.0% | 0.147 m | 0.207 m/s | 6.121 m |
| 20 ms | 1.00× | 100.0% | 0.176 m | 0.235 m/s | 6.206 m |
| 30 ms | 1.50× | 100.0% | 0.229 m | 0.261 m/s | 6.314 m |
| 40 ms | 2.00× | 86.5% | 0.299 m | 0.385 m/s | 5.827 m |
| 60 ms | 3.00× | 1.6% | 0.200 m | 0.662 m/s | 1.800 m |
| 80 ms | 4.00× | 0.0% | 0.130 m | 0.695 m/s | 0.866 m |

阶段判断：30 ms 仍能完成全部 episode，但轨迹误差已经明显增大；40～60 ms 之间成功率快速崩溃。由于严重失败会很早终止，60/80 ms 的终止前 root RMSE 反而低于 40 ms，不能据此认为跟踪更好，应结合成功率、episode 长度和根位移解释。下一轮以 5 ms 精度补测 35/45/50/55 ms。

边界细化结果目录为 `evaluations/Unitree-G1-Tracking/robustness_delay_refine_20260819`，共 4 场景 × 3 seeds × 64 episodes，即 768 episodes。

| 动作延迟 | 成功率 | Body MPKPE | 根 RMSE | 足滑 | 根位移 |
|---:|---:|---:|---:|---:|---:|
| 35 ms | 99.0% | 0.070 m | 0.265 m | 0.295 m/s | 6.323 m |
| 45 ms | 55.7% | 0.105 m | 0.305 m | 0.476 m/s | 4.716 m |
| 50 ms | 29.2% | 0.122 m | 0.298 m | 0.571 m/s | 3.680 m |
| 55 ms | 3.6% | 0.132 m | 0.247 m | 0.647 m/s | 2.480 m |

最终结论：策略能容忍约 1.5 个控制周期的纯位置目标延迟，但从 35 到 55 ms 成功率由 99.0% 快速降至 3.6%。训练时可先随机化 0～35 ms，再逐步扩到 45～50 ms；超过 55 ms 对当前步态基本不可用。E007 选用 35 ms 作为“单独近安全”的组合测试值。

### E007：近安全边界的组合扰动

- 日期：2026-08-19
- 状态：完成
- 核心问题：若每个扰动单独施加时成功率约为 96%～100%，组合后是否仍能成功，还是出现明显的非线性性能崩溃。
- 单项边界：局部低摩擦 0.20（100%）、电机强度 58%（96.9%）、动作延迟 35 ms（99.0%）、躯干负载 6 kg（96.4%）、侧向推扰 0.25 m/s（既有评测为安全档）。
- 组合预设：
  - `combo_patch_motor`：局部摩擦 0.20 + 电机 58%；
  - `combo_patch_delay`：局部摩擦 0.20 + 延迟 35 ms；
  - `combo_motor_delay`：电机 58% + 延迟 35 ms；
  - `combo_actuation_payload`：电机 58% + 延迟 35 ms + 负载 6 kg；
  - `combo_all`：上述四项再加 0.25 m/s 侧向推扰。
- 实验设计：先用 `combo_all` smoke 验证多类配置可同时生效；随后在同一批次重新测 clean、5 个单项和 5 个组合，每个场景 3 seeds × 64 episodes，避免跨批次比较误差。

失败记录 1：首次 `combo_all` smoke 在环境初始化时失败，原因是 `dr.effort_limits` 只接受原始位置执行器，而动作延迟会把它包装成 `DelayedActuator`；物理仿真尚未开始。修正为组合场景先在 6 组原始执行器配置上确定性缩放 effort limit，再包装延迟执行器，物理含义与单独的电机强度事件一致。失败目录 `robustness_combo_smoke_20260819` 不作为结果。

修正后的 `combo_all` smoke（8 episodes）成功执行，结果为 0% 成功。正式矩阵使用 11 场景 × 3 seeds × 64 episodes，共 2112 episodes；结果目录为 `evaluations/Unitree-G1-Tracking/robustness_combo_full_20260819`。

| 场景 | 成功率 | 平均步数 / 147 | Body MPKPE | 足滑 | 根位移 |
|---|---:|---:|---:|---:|---:|
| clean | 100.0% | 147.0 | 0.041 m | 0.195 m/s | 5.989 m |
| 局部摩擦 0.20 | 100.0% | 147.0 | 0.045 m | 0.220 m/s | 5.934 m |
| 电机 58% | 96.9% | 144.1 | 0.069 m | 0.204 m/s | 5.764 m |
| 延迟 35 ms | 99.0% | 146.0 | 0.070 m | 0.295 m/s | 6.324 m |
| 负载 6 kg | 95.8% | 144.6 | 0.074 m | 0.264 m/s | 5.936 m |
| 侧推 0.25 m/s | 100.0% | 147.0 | 0.042 m | 0.197 m/s | 5.962 m |
| 局部摩擦 + 电机 | 94.3% | 143.1 | 0.076 m | 0.301 m/s | 5.643 m |
| 局部摩擦 + 延迟 | 59.4% | 119.9 | 0.096 m | 0.558 m/s | 5.100 m |
| 电机 + 延迟 | 18.8% | 86.5 | 0.122 m | 0.422 m/s | 3.709 m |
| 电机 + 延迟 + 负载 | 0.0% | 26.5 | 0.142 m | 0.411 m/s | 1.040 m |
| 五项全组合 | 0.0% | 27.2 | 0.140 m | 0.414 m/s | 1.062 m |

交互效应判断：若粗略按单项成功率独立相乘，局部摩擦+延迟、电机+延迟、电机+延迟+负载的预期成功率约为 99.0%、95.9%、91.9%，实际仅为 59.4%、18.8%、0%。独立乘积不是严格统计模型，但可作为直观参照，结果足以证明存在强烈的组合放大效应。延迟是关键耦合因素：局部摩擦+电机仍有 94.3%，一旦加入 35 ms 延迟，跟踪与恢复能力迅速崩溃。

终止模式：`combo_motor_delay` 的 192 个 episode 中只有 36 个纯 time-out；其余主要触发 `anchor_pos` 或 `ee_body_pos`。`combo_actuation_payload` 的 192 个 episode 中 177 个由 `ee_body_pos` 单独触发，说明执行器在延迟和额外惯性下无法及时维持参考末端轨迹。全组合已经受到 0% 成功的地板效应，不能进一步量化局部摩擦和侧推的边际影响；后续训练后应原样重测。

结论：当前基线不是对所有单项都脆弱，而是缺少对多个隐含动力学因素同时变化的适应能力。下一阶段 T001 不应只做单因素随机化，应联合随机化摩擦、电机、延迟和负载，并采用从单项到组合的 curriculum；T002 再加入短历史观测，检验历史信息能否帮助策略在线识别有效执行能力与延迟。

图表：`robustness_delay_full_20260819/actuation_robustness.png` 汇总 E005/E006；`robustness_combo_full_20260819/combination_robustness.png` 展示 E007 单项与组合差异。

E007 正式矩阵复现命令：

```bash
env CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=/data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/lib \
  MPLCONFIGDIR=/data/users/yanghao/tmp/matplotlib \
  /data/users/yanghao/envs/humanoid/unitree_rl_mjlab-py311/bin/python \
  scripts/evaluate_tracking_robustness.py Unitree-G1-Tracking \
  --checkpoint logs/rsl_rl/g1_tracking/2026-08-18_07-22-34_lafan1_run_stage1_4096/model_4999.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --num-envs 64 --seeds "(42,43,44)" \
  --scenarios "('clean','local_friction_0p2','motor_scale_0p58','delay_35ms','payload_6kg','push_0p25','combo_patch_motor','combo_patch_delay','combo_motor_delay','combo_actuation_payload','combo_all')" \
  --output-dir evaluations/Unitree-G1-Tracking/robustness_combo_full_20260819
```

### T001：联合域随机化鲁棒 MLP

- 日期：2026-08-19
- 状态：全量训练中
- 对照原则：Actor/Critic、PPO 超参数、观测和奖励保持与原始 tracking baseline 一致，只改变训练时的动力学分布；因此改进可归因于联合域随机化，而不是更大的网络。
- 初始化：从原始 `model_4999.pt` 继续微调，不从零重新学习 LAFAN1 跑步动作。
- 新任务：`Unitree-G1-Tracking-Robust`。
- 每个 episode 重采样：全局足部摩擦、电机 effort limit、位置目标延迟、torso 点负载；推扰在 episode 内按独立计时触发。Actor 不直接观察这些随机化参数。
- 课程阶段（控制周期 20 ms，延迟按 5 ms 物理步量化）：

| 阶段 | 起始 iteration | 摩擦 | 电机强度 | 延迟 | 负载 | 平面推扰 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.60～1.20 | 80%～100% | 0～10 ms | 0～2 kg | ±0.25 m/s |
| 1 | 1000 | 0.45～1.20 | 70%～100% | 0～20 ms | 0～4 kg | ±0.35 m/s |
| 2 | 2500 | 0.30～1.20 | 60%～100% | 0～35 ms | 0～6 kg | ±0.50 m/s |
| 3 | 4000 | 0.20～1.20 | 55%～100% | 0～40 ms | 0～7 kg | ±0.50 m/s |

设计原因：E007 表明单项边界直接组合会使当前策略立即失效，因此不能在微调一开始就只采样最难组合。课程从已有策略大多能完成的范围开始，在 5000 iterations 内逐步覆盖 E007 的失效边界。最终仍使用完全相同的 E001～E007 矩阵评测，避免训练分布与测试口径混淆。

失败记录 1：首次 smoke 启动时外层已经设置 `CUDA_VISIBLE_DEVICES=6`，同时仍传入 `--gpu-ids '[6]'`。启动器将该参数解释为“当前可见列表中的索引”，因此发生 `IndexError`，环境和训练均未初始化。修正为 `--gpu-ids '[0]'`，它在单元素可见列表中正确映射回物理 GPU 6。

失败记录 2：首个有效仿真 smoke 成功加载 `model_4999.pt`，但恢复训练会同时延续全局训练计数；原课程阈值 1000/2500/4000 因此全部已被越过，日志显示直接进入 stage 3，16 个环境在最难联合随机化下迅速终止。执行器包装、随机化事件和课程指标均正常工作，但课程起点错误。已把后三阶段的绝对阈值改为 6000/7500/9000 iterations，对应相对本次微调的 1000/2500/4000 iterations；该 run `2026-08-19_04-44-05_t001_joint_dr_smoke_16_v2` 仅作为失败诊断，不作为训练结果。

有效 smoke：`2026-08-19_04-45-15_t001_joint_dr_smoke_16_v3` 完成 2 iterations，正确加载原始 checkpoint，课程日志稳定显示 stage 0、摩擦下限 0.6、电机下限 0.8、延迟上限 10 ms、负载上限 2 kg；PPO 更新和 checkpoint/ONNX 保存成功，无 NaN/Inf。16 环境的两轮统计受随机 episode 起点和样本量影响，不用于判断性能。下一步用 512 环境做 200-iteration 短训练，检查吞吐和初期稳定性。

短训练：`2026-08-19_04-46-07_t001_joint_dr_short_512_200`，512 环境 × 200 additional iterations，共 2,457,600 samples。初期 mean episode length 约 55，约 30 iterations 后恢复到 400 以上；最终 iteration 5198 的 mean episode length 为 478.36/500、mean reward 31.90，无 NaN/Inf。吞吐约 4.2k～7.4k steps/s，总用时 7分13秒。结果证明预训练策略会在新分布下短暂退化，但能迅速适应 stage 0。

短训 checkpoint 的 1 seed × 16 episodes E007 回归结果仅用于管线验证：clean 100%，延迟 35 ms 100%，电机+延迟 68.8%，电机+延迟+负载及五项组合仍为 0%。相比原 baseline 的正式结果，电机+延迟出现早期改善信号，但样本量很小且 clean root RMSE 上升到 0.233 m，不能作为最终结论。输出目录：`evaluations/Unitree-G1-Tracking-Robust/t001_short_combo_smoke_20260819`。

全量脚本：`scripts/run_lafan_tracking_t001.sh`；配置为 4096 环境、5000 additional iterations、每 500 iterations 保存，仍从原始 `model_4999.pt` 初始化。最终 checkpoint 预计为 `model_9998.pt`，随后运行完整 3 seeds × 64 episodes 的 E001～E007 对照。

全量训练于 2026-08-19 04:56 UTC 启动：

- tmux：`g1_tracking_t001`
- 运行目录：`logs/rsl_rl/g1_tracking/2026-08-19_04-56-15_t001_joint_dr_full_4096`
- 控制台日志：`/data/users/yanghao/logs/unitree_rl_mjlab/g1_tracking_t001_full_20260819.log`
- GPU：物理 GPU 6；启动后显存约 2.8 GiB。
- 已核对日志明确加载原始 `model_4999.pt`，learning iteration 从 4999 继续到目标 9999，课程为 stage 0。
- 早期 iteration 5021：mean episode length 281.84、mean reward 17.30，较刚切换训练分布时继续恢复；吞吐约 22.5k steps/s，动态 ETA 约 5 小时。早期统计不作为最终性能结论。

多卡加速检查（用户确认其余 GPU 空闲后执行）：

- 单卡主训练在 iteration 6149～6257 附近稳定约 22k steps/s、约 4.4 s/iteration。
- 第一次四卡测试使用 GPU 1/2/3/4，其中 GPU 4 虽在 `nvidia-smi` 中空闲，但 `Exclusive_Process` 模式返回 `cudaErrorDevicesUnavailable`，rank 3 初始化失败；测试未进入 PPO iteration。
- 第二次四卡测试避开 GPU 4、改用 GPU 1/2/3/5，但一个 `Exclusive_Process` rank 长时间停在环境初始化，其他 rank 等待同步；同样未产生训练结果。
- 双卡测试改用 `Default` 模式的 GPU 5/7，每卡 2048 环境、全局仍为 4096 环境。20 iterations 测得约 34k～47k steps/s、2.1～2.9 s/iteration，是单卡的约 1.5～2.1 倍，且训练成功结束。
- 决策：从单卡 run 已落盘的 `model_6000.pt` 切换到 GPU 5/7 双卡继续。单卡在停止前已运行到约 iteration 6257，因此会重跑约 257 iterations，但预计净节省 1.5～2 小时。旧 run 和 checkpoint 全部保留。
- 双卡续训脚本：`scripts/run_lafan_tracking_t001_2gpu.sh`；每卡 2048 环境、4000 additional iterations、每 500 iterations 保存。
- 单卡会话在约 iteration 6257 停止后，GPU 6 已释放；没有删除旧 run 或 checkpoint。
- 双卡续训于 2026-08-19 06:28 UTC 启动，tmux 为 `g1_tracking_t001_2gpu`，运行目录为 `logs/rsl_rl/g1_tracking/2026-08-19_06-28-23_t001_joint_dr_full_2gpu_resume6000`，日志为 `/data/users/yanghao/logs/unitree_rl_mjlab/g1_tracking_t001_2gpu_20260819.log`。
- 已确认两个 rank 均加载 `model_6000.pt`，课程处于 stage 1。环境重新初始化造成的短暂低 episode length 在约 20 iterations 内恢复；iteration 6025～6030 的 mean episode length 约 447～462、mean reward 约 29.8～30.8，吞吐约 27k～36k steps/s，动态 ETA 约 3～4 小时。
- 2026-08-19 最新检查：双卡会话仍正常运行，已到 iteration `9794/10000`（完成约 `97.94%`），最新落盘 checkpoint 为 `model_9500.pt`。当前处于最终 curriculum stage 3：摩擦下限 0.20、电机强度下限 55%、动作延迟上限 40 ms、torso 负载上限 7 kg。
- iteration 9794 的 mean reward 为 `26.60`，mean episode length 为 `488.71/500`，action noise std 为 `0.35`；训练未出现 NaN/Inf 或异常退出。当前吞吐约 `24.2k steps/s`、约 `4.05 s/iteration`，日志估计剩余约 13 分钟。
- GPU 5/7 分别占用约 2.6/2.2 GiB，训练进程保持活跃；GPU 4/6 当前空闲。GPU 0～3 已被其他任务大量占用，不纳入本实验后续调度。
- 全量训练已于 2026-08-19 10:55 UTC 正常结束。最终 checkpoint 为 `model_9999.pt`，两个 worker 均无错误退出；累计 `393,216,000` 环境步。最后一轮 mean reward `26.10`、mean episode length `470.42/500`，处于 curriculum stage 3。单轮统计受 episode 采样波动影响，最终泛化结论只依据固定矩阵评测。
- 2026-08-19 11:09 UTC 启动 T001 全量 E001～E007 复测。脚本为 `scripts/run_t001_benchmark_worker.sh`，使用 GPU 5/6/7 三个独立 worker 并行；每个测试场景仍保持 3 seeds × 64 episodes，拆分 GPU 不改变实验统计口径。会话分别为 `t001_eval_w1`、`t001_eval_w2`、`t001_eval_w3`，日志分别位于 `logs/benchmarks/t001_eval_w{1,2,3}_20260819.log`。
- 三个 worker 启动后均已加载 `model_9999.pt` 并进入仿真，GPU 5/6/7 显存约 0.6～0.7 GiB、利用率约 57%～69%。输出统一写入 `evaluations/Unitree-G1-Tracking-Robust/t001_e*_20260819`，完成后与 baseline 对应 summary 逐场景比较。
- 全部 74 个匹配场景评测已完成，无 worker 异常。统一差分表为 `evaluations/Unitree-G1-Tracking-Robust/t001_vs_baseline_20260819/comparison.csv`；跨矩阵平均成功率提升 `21.94` 个百分点，38 个场景提升超过 1 个百分点，没有场景下降超过 1 个百分点。少数 `-0.52` 个百分点来自 192 episodes 中少成功 1 次，应视为采样尺度内的小波动。

T001 关键正式结果（每项 3 seeds × 64 episodes）：

| 场景 | 原策略 | T001 | 提升 |
|---|---:|---:|---:|
| 全局摩擦 0.20 | 38.0% | 58.3% | +20.3 pp |
| 局部摩擦 0.05 | 21.9% | 49.5% | +27.6 pp |
| 整体质量 1.40× | 0.0% | 97.9% | +97.9 pp |
| torso 负载 9 kg | 13.0% | 100.0% | +87.0 pp |
| 电机强度 50% | 33.9% | 100.0% | +66.1 pp |
| 延迟 50 ms | 29.2% | 90.6% | +61.4 pp |
| 局部摩擦 + 延迟 | 59.4% | 97.4% | +38.0 pp |
| 电机 + 延迟 | 18.8% | 99.0% | +80.2 pp |
| 电机 + 延迟 + 负载 | 0.0% | 77.6% | +77.6 pp |
| 五项全组合 | 0.0% | 65.6% | +65.6 pp |

clean 成功率为 99.5%（基线 100%，差 1/192 episode），Body MPKPE 从 0.041 m 降到 0.035 m。T001 明确修复了原策略的组合动力学崩溃，但全局极低摩擦、60 ms 以上延迟和五项组合仍有剩余失败，因此下一阶段验证短历史是否能根据最近状态—动作响应进一步适应。

- 2026-08-20 已使用最终 `model_9999.pt` 在 GPU 5 上完成 clean 场景离屏回放：1280×720、500 帧（控制时长约 10 s），输出为 `logs/rsl_rl/g1_tracking/2026-08-19_06-28-23_t001_joint_dr_full_2gpu_resume6000/videos/play/rl-video-step-0.mp4`。该视频展示策略实际控制结果，不是原始动作数据直接播放。

### T002：短历史观测 MLP

- 唯一结构变量：保留相同联合域随机化、奖励、PPO、critic 和 MLP 隐层，只给 actor 的 base 线/角速度、关节位置/速度、上一步动作加入 4 帧历史。控制周期 20 ms，对应 `t, t-20, t-40, t-60 ms`；命令和参考 anchor 仍只使用当前帧，避免无意义复制参考输入。
- actor 输入由 `160` 维增至 `439` 维，网络仍是 MLP `439→512→256→128→29`，不是 Transformer；critic 保持 `286→512→256→128→1`。
- 为保证公平且避免从头训练，`scripts/expand_tracking_checkpoint_history.py` 把 T001 actor 首层权重映射到各 observation term 的最新帧，历史列以 0 初始化；normalizer 和 Adam 状态同步扩展。迁移前后在随机输入上的最大 policy 输出差为 `2.38e-6`，数值上等价。
- 派生初始化 checkpoint：`logs/rsl_rl/g1_tracking/t002_history4_init_from_t001/model_9999.pt`。
- 16 环境 × 2 iterations smoke 已完成：任务正确报告 actor 439 维，checkpoint 与 optimizer 严格加载，stage 3 curriculum、PPO 更新、checkpoint/ONNX 保存均正常。目录：`logs/rsl_rl/g1_tracking/2026-08-19_11-20-11_t002_history4_smoke_16`。
- 下一步先运行 2048 环境 × 200 iterations 短训练，再用 E007 小矩阵检查历史列是否产生正向信号；通过后进入全量训练。
- T002 短训练最终未恢复：iteration 10198 的 mean episode length 仅 `5.31` steps、mean reward `-0.75`。因此不启动 E007 和全量训练；该结果说明“函数输出等价初始化”不足以保证 PPO 在新增历史维度后的优化稳定性，后续需重置 optimizer、降低学习率或采用单独历史编码器。本 run 作为失败实验保留，不影响 T001 结论和 Sprint 分支。

### S001：Unitree G1 Sprint-v2 直线竞速

- 日期：2026-08-20
- 状态：2048 环境短训练准备启动
- 对照：保留 `Unitree-G1-Marathon` 和其 `model_499.pt`，新建独立任务 `Unitree-G1-Sprint-v2`，避免覆盖旧失败基线。
- Marathon-v1 漏洞：初始 yaw 在 ±π 随机、前进奖励使用机体坐标 x 速度、没有世界赛道和出界终止，因此高速旋转仍可持续获得“前进”奖励；短训曾测得约 `2.48 rad/s` 的平均 yaw 误差代理。
- Sprint-v2 修复：所有环境以世界 `+X` 为赛道方向，初始 yaw 仅 ±0.05 rad；核心进度奖励使用世界 x 速度，并同时乘直立、朝向与跑道位置门控。新增横向位置、世界横向速度、朝向误差、世界 yaw rate 惩罚；根节点偏离中心线 0.9 m 或后退超过 0.5 m 时提前终止。
- 平滑/物理约束：保留足滑、落地冲击、机械功率、关节加速度、关节限位与自碰撞，新增二阶动作变化 `action_acc_l2`，避免通过高频关节抖动换速度。
- 速度课程：`0.8～1.8 → 1.5～2.5 → 2.2～3.2 → 2.8～4.0 m/s`，分别在 absolute iteration `0/2000/4500/7000` 切换；先学干净直线速度，再扩展鲁棒随机化。
- 初始化：从 Marathon-v1 `model_499.pt` 热启动相同的 392 维、4 帧历史 MLP，仅复用站立和交替步态能力；新奖励和赛道定义从 Sprint-v2 开始。
- 失败 smoke：首次启动在环境创建前因配置从上游 `mjlab.tasks.velocity.mdp` 查找本地新增终止函数而报 `AttributeError`，没有产生仿真数据。修正为从 `src.tasks.velocity.mdp.terminations` 显式导入。
- 有效 smoke：`logs/rsl_rl/g1_velocity/2026-08-20_02-13-20_sprint_v2_smoke_16_v2` 完成 16 环境 × 2 iterations；Actor 392、Critic 113，22 项奖励和 4 项终止均正确注册，checkpoint/optimizer/PPO/ONNX 保存成功，无 NaN/Inf。第二轮即时指标：世界前进速度 `0.813 m/s`、横向偏移 `0.152 m`、朝向对齐 `0.931`；样本仅 0.48 s，不用于性能结论。
- 短训练脚本：`scripts/run_g1_sprint_v2_short.sh`，GPU 5、2048 环境、500 additional iterations、每 100 iterations 保存。短训后必须以固定命令速度回放并检查旋转、出界、摔倒和动作自然性，再决定全量训练。
- 短训练于 2026-08-20 02:14 UTC 启动：tmux `g1_sprint_v2_short`，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_02-14-50_sprint_v2_short_2048_500`，控制台日志 `logs/benchmarks/g1_sprint_v2_short_20260820.log`。早期 iteration 509 吞吐约 `38.1k steps/s`，世界前进速度约 `0.787 m/s`、横向偏移 `0.297 m`、朝向对齐 `0.914`；mean episode length `143.66/1000`，主要因旧策略出界，尚处于纠正阶段。
- 500-iteration 短训练已于 02:26 UTC 完成，最终 checkpoint `model_998.pt`，总用时 11分07秒、末轮吞吐约 35.7k steps/s。训练即时指标从早期世界速度约 0.79 提升到 1.15 m/s，朝向对齐从约 0.91 提升到 0.977，mean episode length 从约 144 提升到 308/1000；仍以出界终止为主。
- 固定 1.5 m/s、seed 42、64 episodes 严格评测已完成，并对旧 Marathon `model_499.pt` 使用相同 Sprint-v2 环境复测。旧→Sprint 短训：实际世界前进速度 `0.931→1.280 m/s`、yaw rate RMS `1.767→0.388 rad/s`、平均存活 `2.10→4.40 s`、平均前进距离 `1.97→5.65 m`、足滑 `0.345→0.301 m/s`；出界率仅由 100% 降到 98.4%，仍不合格。输出：`evaluations/Unitree-G1-Sprint-v2/{marathon_model499,short_model998}_speed1p5_seed42`。
- 决策：不提前进入更高速 curriculum。继续从 `model_998.pt` 在 stage 0 的 0.8～1.8 m/s 范围训练 1000 iterations，使 absolute iteration 到约 1998、仍低于 stage 1 阈值 2000。脚本 `scripts/run_g1_sprint_v2_stage0.sh`；完成后仍以 1.5 m/s 的同一评测口径作为是否解锁速度的门槛。
- stage 0 续训于 2026-08-20 02:42 UTC 启动：tmux `g1_sprint_v2_stage0`，GPU 5，运行目录 `logs/rsl_rl/g1_velocity/2026-08-20_02-42-10_sprint_v2_stage0_2048_1000`，日志 `logs/benchmarks/g1_sprint_v2_stage0_20260820.log`。iteration 1026 吞吐约 39.1k steps/s、ETA 约 21 分钟，课程仍为 0.8～1.8 m/s；进程无异常。
- 自动评测脚本 `scripts/evaluate_g1_sprint_v2_stage0_after.sh` 已配置：只在训练会话正常退出且预期最终 `model_1997.pt` 存在时，依次评测固定 1.0/1.5/1.8 m/s、seed 42、各 64 episodes；若训练提前失败则拒绝用中间 checkpoint 代替。
- stage 0 续训于 2026-08-20 03:05 UTC 正常完成，最终 checkpoint 为 `model_1997.pt`；累计约 49.15M environment steps，本段耗时 22 分 34 秒、平均约 36.2k steps/s，无 NaN/Inf。自动评测首次未启动的原因不是训练失败，而是 tmux 对 `g1_sprint_v2_stage0` 做前缀匹配时同时命中了 `g1_sprint_v2_stage0_eval`；脚本已用精确目标 `=g1_sprint_v2_stage0` 修复。
- 固定速度结果（seed 42，各 64 episodes）：1.0 m/s 命令下实际速度 0.946 m/s、存活 8.33 s、距离 7.89 m、yaw RMS 0.310 rad/s、足滑 0.217 m/s、出界率 100%；1.5 m/s 下实际速度 1.366 m/s、存活 5.89 s、距离 8.05 m、yaw RMS 0.343 rad/s、足滑 0.281 m/s、出界率 98.4%；1.8 m/s 下实际速度 1.634 m/s、存活 4.13 s、距离 6.75 m、yaw RMS 0.376 rad/s、足滑 0.337 m/s、出界率 100%。三档跌倒率均为 0%。输出位于 `evaluations/Unitree-G1-Sprint-v2/stage0_model1997_speed{1p0,1p5,1p8}_seed42`。
- 与短训 `model_998.pt` 的 1.5 m/s 对照相比，速度 1.280→1.366 m/s、存活 4.40→5.89 s、距离 5.65→8.05 m、yaw RMS 0.388→0.343、足滑 0.301→0.281，但出界率仍为 98.4%。因此 stage 0 训练有正向收益，却没有解除赛道约束失效，暂不进入 stage 1。
- 原因诊断：actor 的 392 维输入包含本体感觉、命令与历史动作，但没有世界横向位置或相对世界 `+X` 的航向；奖励能告诉 PPO “出界不好”，单步策略却无法判断自己在中心线左侧还是右侧，属于结构性的部分可观测问题。下一实验将加入 track-relative actor observation（横向位置、航向误差、世界前向/横向速度），扩展输入层并继承 `model_1997.pt` 的已有权重，再用相同 1.5 m/s benchmark 判断出界率是否真正下降。

### S002：Unitree G1 Sprint-v3 赛道状态观测

- 唯一核心变量：在 v2 的 Actor 末尾新增 5 维当前帧 `track_state`：`y/0.9`、heading cos、heading sin、`v_x/4`、`v_y/4`。旧的 7 项本体观测仍各保留 4 帧历史和原顺序，因此 Actor 为 `392+5=397` 维；Critic 仍为 113 维，奖励、终止和 PPO MLP 不变。
- 热启动：`scripts/expand_sprint_checkpoint_track_state.py` 扩展 `model_1997.pt`。旧 392 列完整复制，新 5 列权重、Adam 一阶/二阶动量置零；normalizer 新维均值为 0、方差/标准差为 1。已核对第一层 `(512,397)`、normalizer `(1,397)` 且新列最大绝对值为 0，故转换时与旧 Actor 严格等价。
- 课程控制：首个提速节点由 iteration 2000 延后至 2500，避免新输入尚未学习时马上扩大到 2.5 m/s。先进行 16 环境 smoke，再在 GPU 5 做 2048 环境 × 500 additional iterations；用固定 1.5 m/s、seed 42、64 episodes 与 v2 `model_1997.pt` 对照。通过标准为跌倒率接近 0、出界率显著下降且速度/足滑不倒退；通过后自动进入 1.5～2.5 m/s。
- 有效 smoke：`2026-08-20_06-14-22_sprint_v3_track_obs_smoke_16` 完成 2 iterations。ObservationManager 明确显示 Actor 397、Critic 113、`track_state` 仅当前帧 5 维；扩展 checkpoint 和 Adam 状态加载成功，课程仍为 0.8～1.8 m/s，PPO 更新及 checkpoint 保存无 NaN/Inf。
- 纠偏适应训练于 06:15 UTC 启动：GPU 5、2048 环境、500 additional iterations，tmux `g1_sprint_v3_track_adapt`，目录 `2026-08-20_06-15-32_sprint_v3_track_adapt_2048_500`。iteration 2016 吞吐约 34.0k steps/s、ETA 约 11.5 分钟，课程为 0.8～1.8 m/s；训练初期随机 episode 起点统计波动较大，最终以独立 deterministic benchmark 为准。
- 自动门控：训练完成后评测本轮全部落盘 checkpoint（固定 1.5 m/s、seed 42、各 64 episodes），优先选择出界率最低、存活最长者。进入 stage 1 必须同时满足跌倒率≤5%、出界率≤20%、实际速度≥1.25 m/s、足滑≤0.35 m/s；通过后自动在 GPU 5 启动 1.5～2.5 m/s 的 1000-iteration 训练，未通过则停止提速并保留诊断结果。
- 训练于 06:27 UTC 正常完成，最终 `model_2496.pt`，24.576M 本段 environment steps，耗时 11分29秒、末轮约 35.1k steps/s。自动评测比较 `model_2000/2100/2200/2300/2400/2496`；最佳为 `model_2300.pt`：固定 1.5 m/s、64 episodes 下跌倒 0%、出界 67.2%、time-out 32.8%、速度 1.430 m/s、存活 11.08 s、距离 16.07 m、yaw RMS 0.271 rad/s、足滑 0.335 m/s。相较 v2 `model_1997` 的出界 98.4%、存活 5.89 s、距离 8.05 m，赛道状态观测带来显著正向结果。
- stage-1 门控未通过，因此 06:29 后没有训练进程且 GPU 5 空闲，这是预期安全行为。后期 `model_2400/2496` 出界率回升到 84.4%/100%，说明高学习率下出现策略振荡，不能简单使用最后 checkpoint。

### S003：Sprint-v3-Lane 定向纠偏与稳定更新

- 从最佳 `model_2300.pt` 而非最终模型续训。新增两项 shaping：`lane_barrier=(|y|/0.9)^4`（权重 -2，靠近边界快速增大）和 `outward_lateral_velocity=relu(sign(y)*v_y)`（中心 ±0.1 m 外启用，权重 -1），让策略直接学习“向中心线方向修正”，而不只是知道偏离不好。
- PPO 学习率从 1e-3 降为 3e-4，entropy coefficient 从 0.01 降为 0.005，减少已经形成可用步态后的大幅策略漂移。提速节点设置为 iteration 2800；先完成 500 additional iterations 的低速纠偏，再复用相同门槛决定是否进入 1.5～2.5 m/s。
- 正式训练于 2026-08-20 07:48 UTC 启动：GPU 5，PID 2206578，tmux `g1_sprint_v3_lane_adapt`，2048 envs × 500 iterations，目录 `2026-08-20_07-48-17_sprint_v3_lane_adapt_2048_500`，日志 `logs/benchmarks/g1_sprint_v3_lane_adapt_20260820.log`。iteration 2322 吞吐约 35.0k steps/s、ETA 约 11 分钟；`lane_barrier` 与 `outward_lateral_velocity` 均有非零回报项，无初始化错误或 NaN/Inf。
- 完成后自动比较本轮全部 checkpoint 的固定 1.5 m/s 结果；通过既定四项门槛才会自动启动 `g1_sprint_v3_stage1`。等待脚本使用 tmux 精确会话匹配，避免再次发生前缀误判。
- iteration 2430 资源检查：2048 envs 约 35k steps/s，GPU 5 仅 1.58 GiB、瞬时利用率约 62%。用户要求提高并行度，因此选择最近的安全 checkpoint `model_2400.pt` 切换到 4096 envs、400 additional iterations，使最终仍为 `model_2799.pt`；会保留原运行目录，并重建与新会话绑定的评测门控。
- 4096-env 续训于 07:52 UTC 启动：PID 2209195，tmux `g1_sprint_v3_lane_4096`，目录 `2026-08-20_07-52-59_sprint_v3_lane_adapt_4096_resume2400`。iteration 2423～2425 实测约 61.6～61.7k steps/s、1.59～1.60 s/iteration，显存 2.95 GiB、GPU 利用率约 74%；相对 2048-env 的约 35k steps/s 提升约 76%。新门控会话只等待精确的 4096 训练会话并评测新目录。
- 4096-env 训练于 08:05 UTC 正常完成，最终 `model_2799.pt`，本段 39.322M environment steps，耗时 11分52秒；后段约 54k steps/s。固定 1.5 m/s、seed 42、各 64 episodes 的 checkpoint 对照：`model_2400/2500/2600/2700/2799` 出界率分别为 100%/100%/100%/50%/100%，均无跌倒；最佳 `model_2700` 的 time-out 率 50%、速度 1.358 m/s、存活 12.67 s、距离 17.54 m、yaw RMS 0.373 rad/s、横向位置 RMS 0.363 m、足滑 0.319 m/s。
- 与 S002 最佳 `model_2300` 比，S003 最佳出界率 67.2%→50%、完整率 32.8%→50%、距离 16.07→17.54 m，说明 barrier 和 outward-velocity shaping 有效；但 checkpoint 间波动仍明显，最终 `model_2799` 退回 100% 出界。stage-1 门槛未通过，因此提速训练按设计未启动，GPU 5 已释放。输出：`evaluations/Unitree-G1-Sprint-v3-Lane/lane_adapt_4096_checkpoints_speed1p5_seed42`。

### E0：model_2700 多 seed 左右偏置诊断

- `scripts/evaluate.py` 新增 clean/DR 切换；clean 会关闭 actor observation corruption 以及 startup 的摩擦、encoder bias、base COM 随机化，DR 保留训练时随机化。新增 episode 指标：heading error RMS/absolute mean（degree）、terminal y/vy/heading、`dy/dx`；summary 新增 +Y 和 -Y 出界率。
- 新指标的 4-env clean smoke 已通过。正式口径为固定 1.5 m/s、seeds `(11,23,42,67,89)`、每 seed 128 episodes；clean 和 DR 各 640 episodes，总计 1280。输出目录：`evaluations/Unitree-G1-Sprint-v3-Lane/e0_model2700_{clean,dr}_5seed_128`。
- E0 结果：clean 的 640/640 全部 time-out，无跌倒/出界；速度 1.332 m/s、航向 RMS 2.98°、航向绝对均值 2.51°、`dy/dx=-0.0167`、足滑 0.319 m/s。DR 的出界率 42.19%、time-out 57.81%、无跌倒；速度 1.360 m/s、航向 RMS 7.04°、绝对均值 6.12°、`dy/dx=-0.0755`、足滑 0.326 m/s。
- 方向统计：DR 共 270 次出界，其中 -Y 266 次、+Y 4 次，98.5% 同向；五个 seed 出界率为 38.3%～49.2%，均以 -Y 为主。clean 虽不出界也存在稳定负漂移。这不是 seed 42 偶然波动，E2 必须加入左右镜像数据增强或 mirror loss。

### E1：速度自适应 phase

- 新任务 `Unitree-G1-Sprint-v4-AdaptivePhase` 只改变 phase 观测：新增共享 `running_gait_period()`，Actor phase 与 `adaptive_running_gait` reward 使用完全相同的 `(speed_range=(0.5,4.0), period_range=(0.55,0.30))`。输入仍为 397 维，可直接加载 `model_2700.pt`。
- 训练配置：4096 envs × 300 additional iterations，每 50 保存；保持 0.8～1.8 m/s，不提前提速。PPO 使用 lr 1e-4、clip 0.1、desired KL 0.005、entropy 0.004，以避免此前 checkpoint 振荡。完成后复用 E0 的 clean/DR 多 seed 指标，要求 clean 不退化且 DR 航向/出界有改善。
- 16-env × 2-iteration smoke `2026-08-20_09-25-53_sprint_e1_adaptive_phase_smoke_16` 有效完成：397/113 维 actor/critic 与 `model_2700.pt`/optimizer 严格加载，课程仍为 0.8～1.8 m/s，两轮更新无错误或 NaN/Inf。
- 正式训练于 09:27 UTC 启动：GPU 5、4096 envs × 300 iterations，tmux `g1_sprint_e1_phase`，目录 `2026-08-20_09-27-50_sprint_e1_adaptive_phase_4096_300`。iteration 2725 约 58k steps/s、ETA 约 8 分钟。自动后处理将先用 DR seed42×64 筛选所有 50-iteration checkpoints，再对最佳 checkpoint 运行 clean/DR 各 5 seeds×128 episodes，确保和 E0 口径严格一致。

- E1 于 09:37 UTC 正常完成：最终 `model_2999.pt`，29.491M environment steps，耗时 8分58秒，约 58k steps/s；评测于 09:43 完成。seed42 screening 的 `model_2700/2750/2800/2850/2900/2950/2999` 出界率为 34.4%/100%/100%/98.4%/100%/100%/100%，故自动选择未更新的起点 `model_2700.pt`。
- 同一 `model_2700.pt` 的严格 5-seed 对照表明，自适应 phase 环境相对固定 0.6 s phase：clean 仍为 100% 完赛，速度 1.332→1.341 m/s、航向 RMS 2.98→2.40°、`dy/dx -0.0167→-0.0122`；DR 出界率 42.19→35.31%、完整率 57.81→64.53%、航向 RMS 7.04→6.59°、`dy/dx -0.0755→-0.0626`，仅出现 1/640 跌倒。说明相位一致化的推理时效果为正。
- 负结果：窄跑道上继续 PPO 后，从首个 50-iteration checkpoint 即发生航向崩坏，航向 RMS 上升到 14.8°～18.6°。optimizer lr 已核对为约 1e-4，并非 CLI 参数未生效；更可能是 phase 输入分布改变、窄跑道提前终止导致纠偏样本不足，以及 PPO 更新相互耦合。E1 不采用任何训练后模型，下一阶段把自适应 phase 放入宽跑道/无窄边界终止的 vx/vy/yaw 技能训练中重新适应。

### E2-A：三维命令跟踪基线（无 symmetry）

- 新任务 `Unitree-G1-Sprint-E2A-Command`，输入/网络保持 397/113 维。命令器 `MixedVelocityCommand` 每次重采样保留 50% 纯直线命令；其余由 curriculum 在 iteration 0/200/400 将 `vy` 从 ±0.10→±0.20→±0.30 m/s，yaw 从 ±0.15→±0.30→±0.50 rad/s，vx 固定范围 1.0～1.8 m/s。
- 移除所有世界 +X/lane shaping，避免与合法横移、转向命令冲突；只保留 body-frame 三维命令跟踪、步态和安全/平滑奖励。训练 episode 12 s，±2 m 外或世界 x 后退超过 2 m 才终止。
- `scripts/train.py` 新增 `--weights-only-resume`：RSL-RL 仅加载 actor/critic state（含 normalizer），不加载 optimizer/iteration，并在 load 后把环境 `common_step_counter` 清零。PPO 为 lr 1e-4、clip 0.1、desired KL 0.005、entropy 0.004、8 mini-batches；计划 4096 envs × 600 iterations，每 50 保存。
- 失败记录：首次 smoke 因 `MixedVelocityCommand` 类定义误插入 GUI `for` 循环内部导致 `IndentationError`，在任务导入阶段终止，未创建环境/模型。修正后 `2026-08-20_10-04-16_sprint_e2a_command_smoke_16_v2` 完成 2 iterations；日志明确显示 `Loaded actor/critic weights only; reset optimizer and counters`、iteration 从 0 开始、50% mixed command 类生效、第一档命令范围正确，无 NaN/Inf。
- 正式训练于 10:05 UTC 启动：GPU 5、4096 envs × 600 iterations，tmux `g1_sprint_e2a`，目录 `2026-08-20_10-05-45_sprint_e2a_command_4096_600`。iteration 17 吞吐 50～58k steps/s、ETA 约 20 分钟，显存约 2.58 GiB。
- 评测器新增 `velocity_x/y_rmse`、lateral/yaw direction correctness 和 body-frame vx/vy/yaw 均值。自动矩阵会对全部 checkpoint 测试固定 vx=1.5 下的 straight、vy=±0.3、yaw=±0.5，并以跌倒≤2%、直线速度保留≥95%、两方向正确率≥95%、vy RMSE≤0.15、yaw RMSE≤0.20 选优；输出 `evaluations/Unitree-G1-Sprint-E2A-Command/e2a_command_matrix_seed42`。
- 正式训练于 10:26 UTC 正常完成：最终 `model_599.pt`，58,982,400 environment steps，耗时 20分08秒，后段约 48.4k steps/s；训练与后处理会话均已退出，GPU 5 已释放。自动评测于 10:38 UTC 完成，测试了 `model_0/50/.../550/599` 共 13 个 checkpoint。
- 自动 selector 判定 `passed=false`，综合最佳为 `model_350.pt`，而非最终 checkpoint。`model_350` 的五类命令评测最大跌倒率 0%，直线实际速度 1.515 m/s，相对未训练基准 1.367 m/s 的保留率 110.8%；横向命令最低方向正确率 97.71%，最大 vy RMSE 0.133 m/s，已满足横移要求；yaw 命令最低方向正确率 50.45%，最大 yaw-rate RMSE 0.791 rad/s，未满足 95% 和 0.20 rad/s 门槛。
- `model_599.pt` 同样无跌倒，直线速度 1.412 m/s、横向最低方向正确率 95.43%、最大 vy RMSE 0.141 m/s；yaw 最低方向正确率 52.25%、最大 RMSE 0.895 rad/s。后半段继续训练没有解决 yaw，说明当前主要瓶颈是旋转命令的奖励/采样和左右非对称，而不是生存稳定性。E2-A 结论为“横移技能成功、转向技能失败”；下一实验 E2-B 保持同一起点和训练规模，引入镜像约束并增强 yaw 跟踪，避免把 E2-A checkpoint 的偏置带入对照。
- 任务定义复核推翻了“下一步直接 symmetry”的安排：E2-A 虽移除了世界坐标 reward shaping，却仍将 `outside_lane` 放宽后保留在 ±2 m，并保留 `running_backwards`。在 vx=1.5 m/s、yaw=0.5 rad/s 的正确圆周运动中，半径 3 m、3 秒世界横移约 2.79 m，必然因正确执行命令而被终止；第一阶段 yaw=0.15 持续数秒也存在同样冲突。`model_350` 在 yaw±0.5 下实际 yaw 近零正是“忽略命令以换取存活”的合理局部最优，而非单纯奖励权重不足。原 E2-A 结果仍保留，但其 yaw 指标不能用于否定策略转向能力。

### E2-B0：yaw 任务定义修复探针（无 symmetry）

- 新任务 `Unitree-G1-Sprint-E2B0-Yaw-Probe` 从 E2-A 配置派生但彻底移除 `outside_lane` 和 `running_backwards`，并新增非足部接地传感器；终止只剩 `time_out`、`fell_over`、`illegal_contact`。世界赛道约束不会再进入低层技能学习。
- 新增 `CategoricalVelocityCommand`，按 35% 纯直线、25% 纯横移、30% 纯转向、10% 组合命令采样，避免 E2-A 中 vy/yaw 连续随机耦合。vx 1.0～1.8、vy ±0.30；yaw 在 iteration 0/100/200 为 ±0.15/±0.30/±0.50。
- yaw 指数奖励从 weight 1.5/std 0.3 改为 3.0/0.5，并新增 weight -0.5 的非饱和平方 yaw-rate error；`angular_momentum_penalty` 新增兼容参数，B0 仅惩罚 x/y 角动量，不惩罚合法转向所需的 z 分量。E2-A 和既有任务仍保持原行为。
- 从 E2-A 综合最佳 `model_350.pt` 只加载 Actor/Critic/normalizer，重置优化器和计数器。计划 4096 envs × 200 iterations、每25保存；只用 yaw±0.15/±0.30 筛选，并同时回归 straight、vy±0.3。探针门槛：失败率≤2%、直线速度保留≥95%、yaw 两方向正确率≥90%、yaw RMSE≤0.30；横移宽松回归门槛为正确率≥90%、RMSE≤0.18。
- 配置静态检查及 16-env × 2-iteration 有效 smoke `2026-08-20_10-54-44_sprint_e2b0_yaw_probe_smoke_16_v2` 已通过：397/113 维网络权重加载成功，三项终止、18 项奖励、Categorical command、第一档 yaw±0.15 均被运行时确认，无 NaN/Inf。此前同名无 v2 的 smoke 因沙箱无 CUDA 权限在环境构造前失败，不计为算法实验。
- 正式启动器 `scripts/run_g1_sprint_e2b0_yaw_probe.sh`、自动评测器 `scripts/evaluate_g1_sprint_e2b0_after.sh` 和独立 selector 已完成并通过语法检查。10:57 UTC 尝试创建训练/评测后台会话时，平台自动权限审批连接失败并拒绝执行；复核确认没有后台会话、没有正式 checkpoint，故当前状态是“实现与 smoke 完成，正式探针待启动”，不能误记为训练中。

#### 2026-08-24 公开方案复核后的 B0 修订

- HUGWBC 的 task command 是 `(vx, vy, yaw-rate)`，Actor 使用命令与本体感觉，世界位置只属于外层任务；early termination 为躯干/连杆非法接触和大倾角。其角速度正则只作用于 roll/pitch，速度课程按跟踪表现推进。Unitree 官方 G1 velocity deployment 支持非零 `vy/yaw`，Booster Gym 也将 x/y/yaw 奖励拆分并只惩罚 xy 角速度。三者共同支持当前 B0 的无世界跑道低层定义。
- 为让实验结论可归因，正式 B0 改为两组同起点短对照：B0-A=`删除世界终止+分类采样`，保留旧 yaw 奖励；B0-B 在 A 上追加 `weight 3/std 0.5 + -0.5 yaw error² + 不惩罚 z 角动量`。两组均从 `model_350.pt` weights-only 启动，4096 envs×100 iterations、save interval 25、yaw 仅 ±0.15。先 clean seed42 筛选，再 clean/DR 3 seeds×64 复评；通过者继续100轮至 ±0.30。
- 新增审计项：Actor 仍含 5 维 `track_state` 世界赛道观测。B0 为保持 397 维 checkpoint 兼容而将其零掩码；B1 再收缩输入并映射首层权重。阶段升级以评测门槛触发，不再仅依赖固定 iteration。每个命令类别额外记录 yaw steady-state gain、响应时间、正负方向差值、滑移、倾角和功率，以区分“没转”“转得慢”和“转向不稳”。
- 代码落地：新增 `zero_track_state()`，B0-A/B Actor 的末5维保持存在但恒为零，杜绝世界赛道信息泄漏且兼容 `model_350` 的397维首层。新增独立任务 `Unitree-G1-Sprint-E2B0A-Task-Fix`（17项奖励）和 `Unitree-G1-Sprint-E2B0B-Reward-Fix`（18项奖励）；二者终止均严格为 timeout/fell_over/illegal_contact，命令与PPO完全相同。
- 2026-08-24 02:32 UTC 两组 16-env×2 smoke 分别在 GPU4/5 并行通过：`2026-08-24_02-32-33_sprint_e2b0a_task_fix_smoke_16` 与 `...e2b0b_reward_fix_smoke_16`。日志确认 weights-only reset、yaw±0.15、三项终止、无 NaN/Inf；两轮均无 fell/illegal contact。
- 02:33 UTC 正式 A/B 并行启动。A：GPU4、`2026-08-24_02-33-07_sprint_e2b0a_task_fix_4096_100`；B：GPU5、`2026-08-24_02-33-07_sprint_e2b0b_reward_fix_4096_100`。各4096×100、save interval25、seed42。训练日志分别为 `logs/benchmarks/g1_sprint_e2b0{a_task_fix,b_reward_fix}_20260824.log`；自动评测会话等待两组结束，在 `evaluations/Unitree-G1-Sprint-E2B0-AB/e2b0_ab_seed42` 输出逐组矩阵与总决策。
- 两组于02:37 UTC完成，各9,830,400 steps；A耗时3分23秒、末轮约51.2k steps/s，B耗时3分24秒、末轮约47.3k steps/s，均无 fell_over/illegal_contact。最终checkpoint均为`model_99.pt`。
- 评测工程失败记录：第一次固定命令评测把Categorical的`rel_straight_envs`设0，却未把其余三类重归一，环境构造时因概率和非1退出；已修复为固定评测强制100% combined，保证请求的三个命令轴不被类别掩码。第二次被第一次遗留输出目录的防覆盖机制拒绝；残留已移动为`e2b0_ab_seed42_failed_20260824_0237`保留。随后A在GPU5、B在GPU4并行补评；冗余串行B评测在完整并行结果生成后停止。
- B0-A决策：最佳`model_99`，failure0%，straight1.450 m/s，lateral方向98.52%、RMSE0.0837；yaw最低方向40.31%、最大RMSE0.6383，未产生可用转向信号。仅修任务定义不充分。
- B0-B决策：最佳`model_99`，failure0%，straight1.451 m/s，lateral方向98.40%、RMSE0.0966；yaw最低方向59.48%、最大RMSE0.2965。正向命令实际均值+0.074 rad/s、正确率59.48%；负向实际-0.106 rad/s、正确率62.18%。各checkpoint yaw RMSE持续下降0.669/0.456/0.371/0.344/0.297，证明奖励修复有效但100轮尚未收敛。
- 原selector的straight retention以零掩码后的未训练`model_350`速度1.658 m/s为分母；该模型对1.5命令存在明显过冲。B的1.451虽然retention仅87.53%，但绝对误差更小，不能解释为直线能力灾难性退化。下一版门槛改用`速度≥1.44且vx RMSE不劣化`。由于yaw方向正确率仍低于90%，不进入±0.30；下一探针应保持±0.15，增加纯yaw采样占比，并从B`model_99`继续学习。

### E2-B0-B2：yaw采样聚焦

- 唯一训练变量是命令类别比例：B0-B的straight/lateral/turn/combo=35/25/30/10调整为20/10/60/10，yaw-active占比从40%升到70%。yaw范围继续固定±0.15，所有奖励、终止、随机化、PPO和397维零掩码输入不变；从B0-B最佳`model_99.pt` weights-only重置优化器。
- 16-env×2 smoke `2026-08-24_02-56-47_sprint_e2b0b2_yaw_focus_smoke_16`完成，无fall/illegal/NaN。正式训练于02:57 UTC在GPU5启动：4096×100、每25保存，目录`2026-08-24_02-57-13_sprint_e2b0b2_yaw_focus_4096_100`，iteration1吞吐52.4k steps/s、ETA约4分钟。自动评测输出计划为`evaluations/Unitree-G1-Sprint-E2B0B2-Yaw-Focus/seed42`。
- 训练于03:00:55 UTC完成，最终`model_99.pt`。自动评测在完成baseline/straight/lat±后异常结束：`yaw_pos_015`仅创建空目录且无Python traceback，故未把该次退出解释成模型失败；空目录改名保留，正负yaw用GPU4/5并行补跑，selector随后手动完成。
- 最佳`model_99`：failure0%，straight1.4545 m/s、相对B`model_99`基准1.4508保留100.25%；lateral最低方向98.57%、RMSE0.0859；yaw最低瞬时方向65.96%、最大瞬时RMSE0.2501。checkpoint趋势仍向好：方向59.24/62.93/61.97/63.07/65.96%，RMSE0.295/0.277/0.267/0.262/0.250。
- 固定+0.15命令时episode平均yaw=+0.1284 rad/s、方向69.30%、RMSE0.2339；-0.15时平均=-0.1214、方向65.96%、RMSE0.2501。平均yaw已经正确且接近命令，低方向分数主要来自步态周期内瞬时躯干角速度换向。下一动作不是立刻扩大命令或继续堆奖励，而是补充低通/周期平均yaw和累计heading-rate评测；只有这些指标也显示振荡过大时，才训练filtered-yaw reward和yaw jerk正则。

### E2-B0-B2 yaw低通审计 / E2-B0-B3 ±0.30

- `scripts/evaluate.py` 新增 `yaw_filter_tau_s`（默认0.3秒）及低通yaw RMSE/方向、低通均值、累计heading yaw-rate、稳态增益、80%响应时间和超调比。4-env smoke显示瞬时方向68.46%，低通后99.17%，验证指标能滤除步态周期摆动。
- 正式DR seed42×64/方向结果写入 `evaluations/Unitree-G1-Sprint-E2B0B2-Yaw-Focus/seed42/yaw_audit_tau030`。+0.15：低通方向99.55%、RMSE0.0506、均值0.1252、heading-rate0.1309、增益0.8346、响应0.264s；-0.15：96.45%、0.0613、-0.1180、-0.1209、0.7869、0.905s。两向均无fall/illegal contact，前进速度1.451/1.452 m/s，全部通过预设门槛。
- 结论：B2已学会±0.15转向，瞬时yaw换向主要是跑步步态内振荡；暂不增加filtered reward或jerk惩罚，以免无依据地限制自然摆动。下一单变量实验只扩命令幅度。
- 新任务 `Unitree-G1-Sprint-E2B0B3-Yaw030` 将命令范围固定为±0.30，类别比例仍为20/10/60/10，其他配置不变，从B2 `model_99.pt` weights-only重置优化器。16-env×2 smoke `2026-08-24_03-15-35_sprint_e2b0b3_yaw030_smoke_16`通过；Actor/Critic397/113维、三项终止、18项奖励和±0.30课程均正确，无fall/illegal/NaN。正式规模4096×100、save interval25。
- 第一次正式后台启动创建 `2026-08-24_03-16-47_sprint_e2b0b3_yaw030_4096_100` 并保存 `model_0.pt` 后tmux会话异常退出，GPU随即释放；该目录标为不完整工程失败，不用于算法结论。改用受管运行会话后，有效正式run于03:17 UTC启动：`2026-08-24_03-17-31_sprint_e2b0b3_yaw030_4096_100`，GPU5、4096×100。iteration1约23.0k steps/s、GPU利用率95%、显存5.17 GiB，fall/illegal均0，ETA约7.5分钟。
- 有效run于03:24 UTC完成，9,830,400 environment steps，最终`model_99.pt`。末轮mean reward68.12、episode length600/600，fall/illegal均0；训练后段吞吐升至约47k steps/s。
- 最终模型DR seed42×64/命令的7项回归：straight vx=1.4479；lat+/-方向98.51%/99.82%、RMSE0.0804/0.0704；yaw+/-0.15滤波方向99.69%/97.74%、RMSE0.0425/0.0552、增益0.910/0.796；yaw+/-0.30滤波方向99.83%/99.60%、RMSE0.0635/0.0763、增益0.851/0.836。所有448 episodes无fall/illegal。
- B3通过。±0.30的超调比已从B2约1.57降至1.17～1.20，响应时间+0.637/-0.986s；因此无需新增jerk惩罚。保留`model_99.pt`并进入±0.50短探针，仍保持单变量扩命令范围。

### E2-B1：yaw ±0.50短探针

- 新任务 `Unitree-G1-Sprint-E2B1-Yaw050` 仅把B3的yaw训练范围从±0.30扩到±0.50；20/10/60/10类别比例、奖励、终止、DR、PPO均不变，从B3 `model_99.pt` weights-only重置优化器。
- 16-env×2 smoke `2026-08-24_03-29-36_sprint_e2b1_yaw050_smoke_16`通过，运行时确认±0.50课程、三项终止和18项奖励，无fall/illegal/NaN。
- 正式4096×100于03:30 UTC在GPU5启动，目录 `2026-08-24_03-30-01_sprint_e2b1_yaw050_4096_100`。iteration5约47.4k steps/s，fall/illegal均0，ETA约3.4分钟；完成后先评yaw±0.50，再回归±0.30/±0.15、straight和lat±0.30。
- 训练于03:33 UTC正常完成，9,830,400 steps，最终`model_99.pt`；末轮reward71.14、episode length600、action std0.37、fall/illegal均0。
- DR seed42×64/命令的9类回归全部完成。straight vx1.442；lat+/-方向98.80%/99.79%、RMSE0.0731/0.0743；yaw+/-0.50滤波方向99.83%/99.81%、RMSE0.0813/0.1130、均值+0.447/-0.414、heading-rate+0.462/-0.425、增益0.894/0.828，所有576 episodes零失败。
- 旧技能未发生生存或方向遗忘，但左右幅值仍不一致：+/-0.15增益1.037/0.705，+/-0.30为0.940/0.794；负向响应时间约0.92～1.01s，正向0.24～0.65s。故B1通过“能执行±0.50”技能门槛，但未通过严格对称门槛。保留`model_99.pt`作为E2-C起点，下一实验只引入mirror/symmetry约束。

### E2-C：G1左右镜像数据增强

- 决策边界：这是最后一次yaw专项。只跑4096×100；之后无论结果是否完美都进入高层赛道闭环。E2-C的目标是降低正负增益差，而不是继续提高yaw上限或刷单一RMSE。
- 新增 `src/tasks/velocity/mdp/symmetry.py`，覆盖Actor397维历史、Critic113维特权观测和29维动作。关节映射按MuJoCo actuator顺序显式定义；矢状面反射下pitch保号，roll/yaw取反，左右肢体交换；angular velocity使用伪向量符号，phase加半周期，足部/接触力交换。
- `scripts/test_g1_symmetry.py` 使用随机张量验证Actor/Critic/action镜像是involution，并验证RSL-RL TensorDict扩增后批次从7变14、前半原始后半镜像，全部逐元素通过。
- PPO使用RSL-RL原生`use_data_augmentation=True`、`use_mirror_loss=False`，避免同时引入两个对称机制。新增独立task `Unitree-G1-Sprint-E2C-Symmetry`，环境、命令、奖励与B1完全相同。
- 首次smoke `03-55-26`在runner构造后、learning前因symmetry运行时`_env`对象无法写入YAML退出。修复`train.py`：runner构造前deepcopy静态agent配置用于记录，运行时dict允许extension注入对象。v2 smoke `03-56-11`完成2轮，symmetry metric出现且无fall/illegal/NaN。
- 正式run于03:56 UTC在GPU5启动：`2026-08-24_03-56-37_sprint_e2c_symmetry_aug_4096_100`，4096×100、save interval25，从B1`model_99.pt` weights-only。iteration1约45.4k steps/s，fall/illegal均0，ETA约4分钟。
- 正式训练于04:00 UTC完成，9,830,400 steps，最终`model_99.pt`；末轮reward71.98、episode length600、fall/illegal均0。
- DR seed42×64/命令的9类回归共576 episodes，全部无fall/illegal。straight vx1.4418；lat+/-方向98.96%/99.80%、RMSE0.0695/0.0715；yaw+/-0.15增益1.000/0.733、+/-0.30为0.921/0.808、+/-0.50为0.876/0.841。
- 与B1相比，正负yaw增益差：±0.15由0.3320降到0.2673，±0.30由0.1459降到0.1121，±0.50由0.0661降到0.0356。直线速度1.4420→1.4418基本不变；横移RMSE最大值0.0743→0.0715。镜像增强有效但低速残差仍在。
- 决策：采用E2-C`model_99.pt`作为最终低层策略；严格执行停止规则，不再继续yaw专项。下一步接入世界赛道状态高层控制器，必要时使用正负不同的控制增益补偿低速残差。

### S1：第一档奔跑提速（1.5～2.2 m/s）

- 起点固定为 E2-C `2026-08-24_03-56-37_sprint_e2c_symmetry_aug_4096_100/model_99.pt`，只加载 Actor/Critic/normalizer 并重置优化器与计数器。目标是在保留低层三维命令能力的前提下，把直线奔跑从约1.44 m/s推进到2.0～2.2 m/s。
- 新增 `CategoricalVelocityCommand` 速度回放：主分布 vx=1.5～2.2 m/s，20%环境改采样旧分布1.0～1.8 m/s；类别比例straight/lateral/turn/combined=60/10/20/10，vy±0.30、yaw±0.50。该设计同时增加高速样本和直线样本，但不完全删除旧速度及纠偏技能。
- 新增 `forward_velocity_tracking_error_l2`（weight -0.5），为速度误差提供不饱和梯度；新增 `speed_dependent_torso_lean_l2`，命令1.5→2.2 m/s时目标前倾2°→8°，替代固定竖直躯干目标但沿用weight -0.8。脚滑、落地冲击、机械功率、动作变化、关节限位和非法接触约束保持不变。
- 新任务为 `Unitree-G1-Sprint-S1-Speed220`，启动器 `scripts/run_g1_sprint_s1_speed220.sh`。PPO：lr1e-4、clip0.1、desired KL0.005、entropy0.004、8 mini-batches；4096 envs×300 iterations、save interval25、seed42，继续启用 symmetry data augmentation。
- 16-env×2 smoke `2026-08-24_06-26-55_sprint_s1_speed220_smoke_16` 完成。配置运行时验证通过；target lean约5.1°，新增速度误差奖励非零，无fall/illegal/NaN。
- 工程失败记录：06:27 UTC在GPU5以及06:28 UTC改到GPU4的两次普通受限启动均报`No CUDA GPUs are available`，分别只创建空/配置日志目录 `06-27-51`、`06-28-18`，均在环境构造阶段终止且未进行PPO更新。原因是当前执行沙箱屏蔽CUDA计算，不是GPU占用或代码错误。
- 有效正式 run 于06:28 UTC通过服务器GPU权限在物理GPU4启动：`logs/rsl_rl/g1_velocity/2026-08-24_06-28-40_sprint_s1_speed220_4096_300`。iteration5约46.3k steps/s、ETA约11分钟；课程范围与全部新增项已在日志确认。训练最初几轮出现少量fall/illegal探索事件，暂不据此提前停止，最终以固定命令独立评测筛选checkpoint。
- 计划评测所有25轮间隔checkpoint：straight vx=1.5/2.0/2.2，回归lat±0.3及yaw±0.5；统计失败率、vx RMSE/增益、横移与低通yaw、躯干倾角、足滑、落地冲击、机械功率和动作平滑。只有实际速度达到命令的90%、失败≤2%且安全指标未明显退化，才进入下一档2.0～2.8 m/s。

- S1有效run于06:39 UTC正常完成：300 iterations、29,491,200 steps，13个checkpoint及`policy.onnx`完整；末轮mean reward70.6004、episode length600/600、fall/illegal均0。TensorBoard趋势显示reward从0.1065持续升高，训练速度综合误差下降，目标/实际躯干前倾在末轮约4.97°/4.62°，没有后期策略崩坏。
- 先对13个checkpoint执行DR seed42×64、固定vx=2.2筛选。`model_0`仍有25% fall和18.75% illegal，说明未更新的E2-C起点不能直接稳定执行2.2；从`model_25`起所有模型均64/64 timeout。最终`model_299`取得最高return79.52、最低vx RMSE0.3586，实际vx2.1262 m/s、foot slip0.4167 m/s，因此入选。
- `model_299`六类回归共384 episodes全部零失败。straight vx1.5→实际1.4663、vx2.0→1.9320；lat+/-0.3方向98.90%/99.83%、RMSE0.0697/0.0646；yaw+/-0.5低通方向99.83%/99.78%、RMSE0.0806/0.0837、稳态增益0.935/0.949。与E2-C相比，横移保持，yaw幅值对称性进一步改善。
- 再用DR seeds11/23/67/89、每seed64 episodes复评model299的vx2.2：256/256 timeout，fall/illegal均0；实际vx=2.1314±0.0277 m/s，相对2.2命令跟踪率96.88%，vx RMSE=0.3584±0.0070，slip=0.4194 m/s。连同seed42筛选，总计320个2.2 m/s回合零失败。
- S1判定`passed`，正式选择`model_299.pt`。结构化结论位于`evaluations/Unitree-G1-Sprint-S1-Speed220/decision.json`；原始筛选、回归和多seed结果分别位于`seed42/screen_vx220`、`seed42/model299_regression`与`multiseed/model299_vx220`。下一阶段S2只扩展速度到2.0～2.8 m/s，继续保留旧速度回放、symmetry和三维命令回归门槛。

### S2：第二档奔跑提速（2.0～2.8 m/s）

- 起点为S1验收模型`2026-08-24_06-28-40_sprint_s1_speed220_4096_300/model_299.pt`。新增任务`Unitree-G1-Sprint-S2-Speed280`，只加载Actor/Critic/normalizer，重置optimizer与iteration。
- 主vx分布2.0～2.8 m/s，25%环境回放1.5～2.2 m/s；命令类别仍为straight/lateral/turn/combined=60/10/20/10，vy±0.30、yaw±0.50。保持S1全部奖励权重和symmetry data augmentation，避免同时加大足滑惩罚导致无法区分速度课程与正则项影响。
- 躯干前倾采用连续目标：speed range1.5～2.8 m/s、lean2°～12°。因此2.0/2.2 m/s目标约5.85°/7.38°，与S1对应约6.29°/8°接近，同时为2.8 m/s留出更合理的前倾空间。
- 16-env×2 smoke `2026-08-25_02-59-20_sprint_s2_speed280_smoke_16`正常结束。运行时确认主范围2.0～2.8、平均target lean约9°、weights-only和symmetry生效；两轮无fall/illegal/NaN。
- 有效正式run于02:59 UTC在物理GPU1启动：`logs/rsl_rl/g1_velocity/2026-08-25_02-59-46_sprint_s2_speed280_4096_400`。规模4096×400、save interval25、seed42；PPO lr1e-4、clip0.1、desired KL0.005、entropy0.004、8 mini-batches。iteration5约46.0k steps/s、ETA约14.6分钟。初始任务切换阶段有少量fall/illegal，iteration5已降到各0.0417（训练日志episode统计，不等同最终失败率）。
- 评测计划：全部25轮checkpoint先做DR seed42×64的vx2.8筛选，最佳候选再做vx2.0/2.4/2.8、回放速度1.5、lat±0.3、yaw±0.5及多seed复评。通过条件为实际vx/命令≥90%、fall和illegal各≤2%、旧技能回归不明显退化；2.8 m/s足滑目标≤0.55 m/s，若速度通过但足滑超标，下一步单独做S2-slip正则消融，而不是直接进入3.4 m/s。

- S2正式run于03:14 UTC正常完成：400 iterations、39,321,600 environment steps，17个checkpoint和`policy.onnx`完整落盘。第50轮后mean episode length持续600，训练fall/illegal持续0；末轮reward65.9254、slip0.5343 m/s、mechanical power约656 W。actual/target torso lean从iteration25的5.46°/8.94°收敛至末轮8.12°/8.93°。
- 全checkpoint固定vx2.8的DR seed42×64筛选共1088 episodes，所有模型均零fall/illegal。`model_0`实际2.7067但RMSE0.4766、slip0.5406、return66.61；训练逐步降低误差和滑移。`model_399`实际2.6774、RMSE0.4619、slip0.5007、return75.11，是最低RMSE/最低滑移/最高回报的综合最佳，故不按单一平均速度选择model0。
- `model_399`回归矩阵：straight1.5/2.0/2.4实际1.5038/1.9498/2.3115；lat+/-0.3方向98.87%/99.64%、RMSE0.0770/0.0778；yaw+/-0.5低通方向99.83%/99.71%、RMSE0.0798/0.1177、增益1.002/1.127。共448 episodes零失败。负yaw相较S1出现约12.7%过冲，但仍在既定RMSE0.20门槛内，记录为限制而非继续yaw专项。
- 多seed复评使用11/23/67/89各64 episodes：2.8 m/s下256/256 timeout、fall/illegal均0；实际vx2.6839±0.0318 m/s、命令跟踪率95.85%、vx RMSE0.4612±0.0079、foot slip0.5079±0.0246。加seed42后，入选模型共320个2.8 m/s回合零失败。
- S2判定`passed`，采用`model_399.pt`。结构化决策在`evaluations/Unitree-G1-Sprint-S2-Speed280/decision.json`，原始筛选/回归/多seed目录分别为`seed42/screen_vx280`、`seed42/model399_regression`、`multiseed/model399_vx280`。下一阶段若继续提速，应采用2.5～3.4 m/s重叠区间，并保持2.8 m/s足滑≤0.55与失败≤2%的硬门控。

### S3：第三档奔跑提速（2.5～3.4 m/s）

- 起点为S2验收模型`2026-08-25_02-59-46_sprint_s2_speed280_4096_400/model_399.pt`。新增任务`Unitree-G1-Sprint-S3-Speed340`，weights-only加载并重置optimizer/iteration。
- 主vx2.5～3.4 m/s，30%环境回放2.0～2.8；类别比例60/10/20/10、vy±0.30、yaw±0.50、symmetry与19项奖励权重完全保持S2设置。姿态目标改为speed2.0～3.4对应lean6°～16°，在2.8 m/s约11.71°，与S2的12°近似连续。
- 16-env×2 smoke `2026-08-25_03-31-15_sprint_s3_speed340_smoke_16`正常完成。Actor/Critic397/113维、S2 model399加载、3.4范围、回放和symmetry均正常；两轮无fall/illegal/NaN。第二轮slip0.825、power约1073 W只是极短初始化窗口，但提示S3可能触及接触/功率瓶颈。
- 正式run于03:31 UTC在物理GPU1启动：`logs/rsl_rl/g1_velocity/2026-08-25_03-31-41_sprint_s3_speed340_4096_400`，4096×400、save25、seed42，PPO参数沿用S2。iteration4～7任务切换期fall/illegal窗口计数较高、episode length约90～129；到iteration38 episode length恢复579、fall0.375、illegal0.0833，说明生存能力正在恢复。同期slip仍0.6655 m/s、power约792 W。
- S3采用硬门控：固定3.4 m/s实际速度/命令≥90%，DR独立评测fall与illegal各≤2%，foot slip≤0.55 m/s；旧速度、lat和yaw需继续回归。若速度/生存通过但slip超标，保留最佳checkpoint并新增单变量S3-slip实验（提高滑移接触期惩罚或加入速度相关滑移门控），不直接进入3.4～4.0 m/s。
- 正式run完成400 iterations、39,321,600 environment steps，生成`model_0/25/.../375/399.pt`共17个checkpoint及ONNX。训练reward从-1.28升至iteration350的58.11，末轮57.48；episode length从16.9恢复到后半程基本600/600，末轮fall0、illegal窗口0.0417。训练足滑在iteration25约0.664，后期最低约0.641，显示稳定性恢复但滑移没有降回S2水平。
- DR seed42×64、固定vx=3.4的全checkpoint筛选位于`evaluations/Unitree-G1-Sprint-S3-Speed340/seed42/screen_vx340`。`model_0`失败率90.6%、实际1.836 m/s；到`model_50`失败率已降至1.56%、实际3.130 m/s；`model_75`后基本稳定。综合最佳为`model_375`：实际3.222、跟踪率94.77%、vx RMSE0.562、return70.06、slip0.607、64/64无失败。`model_399`实际3.201、RMSE0.589且1/64非法接触，存在轻微末轮退化。
- `model_375`多seed复核位于`evaluations/Unitree-G1-Sprint-S3-Speed340/multiseed/model375_vx340`：seeds 11/23/42/67/89共320 episodes，fall0、illegal1/320=0.3125%，实际vx3.21165 m/s、跟踪率94.46%、vx RMSE0.56724、xy RMSE0.57147、foot slip0.61001 m/s、heading RMS6.74°、return69.78。
- 判定：速度门槛`3.212≥3.06 m/s`通过，生存门槛`0.31%≤2%`通过，足滑门槛`0.610>0.55 m/s`失败。S3标记`needs_slip_optimization`，保留`model_375.pt`为候选起点，不进入S4。下一实验S3-slip只调整足滑相关约束，weights-only重置优化器；速度≥3.06、总失败≤2%、DR slip≤0.55三项同时通过后，才做旧速度与三维命令回归并升级。

### S3-slip A：foot-slip 权重 -0.4 → -0.8

- 为保证视频命令与评测完全一致，`scripts/play.py`新增固定前进/横移/yaw-rate参数，并在分类命令任务中强制100% combined以绕过轴掩码。用S3`model_375.pt`录制固定3.4 m/s、500帧/50 FPS、1280×720视频，输出`logs/rsl_rl/g1_velocity/2026-08-25_03-31-41_sprint_s3_speed340_4096_400/videos/play/rl-video-step-0.mp4`；文件2.43 MB、时长10秒，抽查第50/150/250/350/450帧均保持奔跑。
- 代码审计确认原`feet_slip`已经使用`contact_sensor.found>0`门控，并对接触足端世界系xy速度平方求和；训练日志记录的均值与独立评测采用同类定义。A组因此只改变一个变量：weight -0.4→-0.8，不修改函数形状。
- 新任务`Unitree-G1-Sprint-S3-Slip-W080`及启动器`scripts/run_g1_sprint_s3_slip_w080.sh`已实现。注册、Python编译和配置断言通过；16-env×2 smoke目录`2026-08-25_06-18-42_sprint_s3_slip_w080_smoke_16`完成，确认397/113维、foot-slip -0.8、weights-only、symmetry和3.4 m/s课程生效，无fall/illegal/NaN。
- 正式run于06:19 UTC在物理GPU1启动：`logs/rsl_rl/g1_velocity/2026-08-25_06-19-08_sprint_s3_slip_w080_4096_150`。规模4096×150、save25、seed42，从S3`model_375.pt`权重级热启动并重置optimizer/counter；PPO与S3相同。先用短实验回答加倍惩罚是否能把DR slip从0.610降至≤0.55，同时要求实际vx≥3.06、总失败≤2%。

### S3-natural-v1：LAFAN1相位动作风格先验

- 人工视频验收否决原S3动作：高速与生存虽通过，但存在膝关节偏直、脚向前探、支撑期滑步、上肢摆动僵硬，不能作为“自然奔跑”结果。按用户指令中止正在运行的S3-slip W080；中止时约iteration60，落盘`model_0/25/50.pt`，训练slip约0.628 m/s。该run标记`stopped_by_design_review`，不从其checkpoint继续。
- 数据审计：`lafan1_run1_subject2_112s_115s.npz`共149帧/50 Hz，root平均平移速度约2.066 m/s。全序列关节自相关与首尾姿态/速度距离均表明40帧为稳定周期；选择69:109时首尾联合误差最低之一。phase0左脚接近落地、phase0.5右脚接近落地，与现有`offset=[0,0.5]`一致。
- 新增stateful reward `phase_motion_joint_style`：按速度自适应period将episode phase映射到40帧参考并线性插值，返回`exp(-weighted_joint_mse/std²)`；腿/腰/臂权重1.0/0.5/0.7，std0.45 rad、reward weight1.5。它不跟踪世界root轨迹，不强制参考速度，只提供屈膝、对侧摆臂和周期姿态风格。
- `Unitree-G1-Sprint-S3-Natural-v1`继承S3全部速度、三维命令、DR、终止、symmetry和PPO，同时继承foot-slip -0.8。16-env×2 smoke `2026-08-25_06-30-02_sprint_s3_natural_v1_smoke_16`通过；初始style RMSE0.658 rad并在第二轮降至0.541，奖励非饱和，无fall/illegal/NaN。
- 正式run于06:30 UTC在GPU1启动：`logs/rsl_rl/g1_velocity/2026-08-25_06-30-40_sprint_s3_natural_v1_4096_200`，4096×200、save25、seed42，从原S3`model_375.pt`weights-only开始。iteration11 style RMSE约0.493，说明策略已收到明确风格梯度；早期episode/failure仍处于任务切换恢复期，不作最终判断。验收增加同摄像机10秒视频盲审，定量门槛仍为vx≥3.06、总失败≤2%、slip≤0.55。

### Natural-v1视频否决与3.4 m/s tracking expert启动

- Natural-v1正式训练完成200 iterations、19,660,800 steps，最终`model_199.pt`；末轮reward62.49、episode600、fall/illegal0，style joint RMSE由0.658降至0.408 rad，实际躯干前倾13.50°，训练slip约0.596。数值表明风格先验有效但没有达到足滑0.55。
- 同视角固定3.4 m/s视频已生成：`logs/rsl_rl/g1_velocity/2026-08-25_06-30-40_sprint_s3_natural_v1_4096_200/videos/play/rl-video-step-0.mp4`。抽帧显示膝部较S3更弯、前倾更明确，但上臂仍后收且腿部仍以蹬滑为主。按新增的人眼硬门槛判定`visual_style_failed`，不再从速度策略局部最优继续加大弱style reward。
- 新增通用工具`scripts/time_scale_motion.py`。对原LAFAN G1 NPZ进行连续线性/四元数最短弧归一化插值，位置轨迹不变，joint/body线角速度乘时间倍率。倍率3.4/2.066=1.6457后得到91帧、1.8秒、root平面速度3.397 m/s的`src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz`，所有数组finite。
- 旧T001 robust tracking`model_9999.pt`直接跟踪加速motion的1.8秒视频位于`artifacts/sprint_tracking_speed340/videos/play/rl-video-step-0.mp4`。实际机器人仍呈自然屈膝和对侧摆臂，但ghost参考逐渐领先；结论是动作流形正确、速度适应不足，适合weights-only微调。
- 新增`scripts/run_g1_tracking_sprint340_probe.sh`：`Unitree-G1-Tracking-Robust`、4096×500、save50、lr1e-4、clip0.1、KL0.005、8 mini-batches，从T001`model_9999.pt`重置优化器。16-env×2 smoke `2026-08-25_06-50-14_tracking_sprint340_probe_smoke_16`通过。
- 正式run于06:51 UTC在GPU1启动：`logs/rsl_rl/g1_tracking/2026-08-25_06-51-21_tracking_sprint340_probe_4096_500`。iteration0因速度突变mean episode14.64 steps、主要为ee-body-pos终止，吞吐约19.4k steps/s、ETA约42分钟；保留adaptive phase sampling与joint DR，后续以episode length和body/joint tracking误差是否持续改善判断探针有效性。

### Tracking sprint 3.4 m/s 探针完成与选优

- 4096×500正式run正常完成，目录`logs/rsl_rl/g1_tracking/2026-08-25_06-51-21_tracking_sprint340_probe_4096_500`；生成`model_0/50/.../450/499.pt`和`policy.onnx`。iteration0→499：mean reward 0.0297→24.2788，episode length 14.64→497.84/500，body position error 0.1032→0.1746 m（早期数值受大量短回合影响），joint position error 1.1045→0.7650 rad，joint velocity error 16.12→14.02 rad/s。model350的episode length 499.90/500最高，model499的最终reward、anchor velocity和joint position更优，因此二者都进入视频/clean比较。
- 同一91帧、1.8秒加速参考录制了两条720p视频：`artifacts/sprint_tracking_speed340/model350/videos/play/rl-video-step-0.mp4`和`artifacts/sprint_tracking_speed340/model499/videos/play/rl-video-step-0.mp4`。抽查第10/30/50/70/85帧：两者均出现自然屈膝、腾空和交替摆臂；model499整体更靠近ghost，未出现原S3的直腿滑行局部最优。
- clean seed42×64一次完整参考片段评测：model350成功率100%、body MPKPE 0.05009 m、joint position RMSE 0.13220 rad、foot slip 0.34657 m/s、root displacement 5.60446 m；model499成功率100%、body MPKPE 0.04776 m、joint position RMSE 0.12861 rad、foot slip 0.33367 m/s、root displacement 5.60531 m。model499相对model350的MPKPE降低约4.65%、foot slip降低约3.72%，选为当前3.4 m/s tracking expert。
- 结果目录：`evaluations/Unitree-G1-Tracking/sprint340_model350_clean_seed42`与`evaluations/Unitree-G1-Tracking/sprint340_model499_clean_seed42`。用户提醒所有GPU均已被占用后，不再启动多seed/DR矩阵；检查确认本项目无残留训练或评测进程。
- 当前结论边界：这证明了3.4 m/s自然动作可以由强tracking策略稳定复现，但它仍是固定动作expert，不等同于可接收任意`vx/vy/yaw`的通用奔跑策略。下一实验应在GPU空闲后先做多seed、摩擦/推扰动评测；通过后再以expert轨迹进行蒸馏或teacher-student微调。

## 7. 文档维护规则

- 每个实验使用唯一编号；开始、完成或失败时都更新状态。
- 记录假设、唯一变量、seed、并行数、checkpoint、输出目录和关键结果。
- 失败实验不删除，记录错误原因和修正方法。
- 大型 checkpoint 和视频不提交 GitHub，只提交脚本、配置、汇总结果、图表和文档。
- `PROGRESS.md` 保存里程碑总览；本文件保存详细实验过程和可复现信息。
