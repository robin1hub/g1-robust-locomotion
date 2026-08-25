# Unitree RL MjLab：RTX 4090 部署教程与项目理解

本文根据一次真实的 A100 服务器部署、训练、评测和可视化记录整理，目标是在一台配备 NVIDIA RTX 4090 的 Linux 服务器上复现 Unitree G1 强化学习运动控制项目。

适用项目：

- 上游：`https://github.com/unitreerobotics/unitree_rl_mjlab`
- 本项目：`https://github.com/robin1hub/g1-robust-locomotion`
- 已验证代码基线：上游提交 `1425b15`

本文重点不是“一条命令安装”，而是解释目录隔离、依赖兼容、验证顺序、训练方法以及各模块在做什么。不同服务器的驱动和系统环境可能不同，安装时应以实际报错为准。

## 1. 项目在做什么

该项目在 MuJoCo/MjLab 中并行创建大量 Unitree 机器人环境，通过 RSL-RL 的 PPO 算法学习运动控制策略。目前主要包含两类任务：

- `Unitree-G1-Flat`：平地速度控制；
- `Unitree-G1-Rough`：带高度扫描的楼梯、斜坡、随机高度场和波浪地形控制。

训练时，策略接收机器人状态和速度指令，输出 29 个关节位置目标。底层 PD 控制器再根据关节目标产生实际控制力矩。因此策略不是直接输出电机力矩。

完整数据流为：

```text
速度指令 + IMU + 关节状态 + 上一动作 (+ Rough 高度扫描)
                         ↓
                     Actor MLP
                         ↓
                  29维关节位置目标
                         ↓
                     PD 控制器
                         ↓
                   MuJoCo 机器人
                         ↓
          reward / termination / next observation
```

## 2. 策略模型和 PPO

当前策略不是 Transformer，而是非循环 MLP Actor–Critic。

### Flat Actor

输入共 98 维：

| 观测 | 维度 |
|---|---:|
| 机身角速度 | 3 |
| 投影重力方向 | 3 |
| 速度指令 | 3 |
| 步态相位 | 2 |
| 关节位置 | 29 |
| 关节速度 | 29 |
| 上一时刻动作 | 29 |
| 合计 | 98 |

网络结构：

```text
98 → Linear(512) → ELU
   → Linear(256) → ELU
   → Linear(128) → ELU
   → Linear(29) → Gaussian action distribution
```

### Rough Actor

Rough 任务增加 187 维 `height_scan`：

```text
285 → 512 → 256 → 128 → 29
```

高度扫描覆盖机器人附近约 `1.6m × 1.0m` 的区域，分辨率为 `0.1m`。它提供地形几何信息，但不能直接辨认摩擦系数或地面材质。

### Critic 与特权信息

Critic 比 Actor 多读取机身线速度、足端高度、接触状态、腾空时间和接触力：

- Flat Critic：113 维；
- Rough Critic：300 维。

这是 asymmetric Actor–Critic：训练时 Critic 可使用仿真特权信息，部署时只保留 Actor。

PPO 的主要配置：

```text
clip ratio            0.2
learning rate         1e-3，自适应调整
gamma                 0.99
GAE lambda            0.95
epochs per iteration  5
mini batches          4
rollout length        24 steps / env / iteration
```

## 3. 环境、奖励与随机化

基础训练环境包含：

- 摩擦随机化：脚部摩擦约 `0.3～1.6`；
- 编码器偏置和观测噪声；
- 躯干质心偏移；
- 每 5～6 秒一次随机推扰；
- 速度指令课程；
- Rough 任务的地形难度 curriculum。

主要奖励包括：

- 线速度、角速度跟踪；
- 直立姿态；
- 合理的关节姿态；
- 动作变化平滑；
- 交替步态和足端抬高；
- 抑制滑脚、自碰撞、硬着地和关节越界；
- 摔倒的大额惩罚。

训练 reward 只用于优化，不能单独证明策略好。最终仍应统计跌倒率、速度 RMSE、滑移和动作平滑度，并观看回放。

## 4. 推荐的 4090 服务器目录

不要把环境、缓存和模型混在一个目录。以下假设用户名为 `$USER`，数据盘为 `/data`：

```text
/data/users/$USER/projects/g1-robust-locomotion   项目源码
/data/users/$USER/envs/humanoid/unitree-mjlab    Python环境
/data/users/$USER/envs/toolchains/miniforge3     Miniforge
/data/users/$USER/cache/warp                      Warp编译缓存
/data/users/$USER/logs/unitree-mjlab              外部控制台日志
```

如果服务器没有 `/data/users`，可统一替换为 `/workspace/$USER` 或管理员分配的数据目录。关键是不要使用系统 Python，也不要在共享服务器上修改全局 CUDA。

本文命令使用变量简化路径：

```bash
export G1_ROOT="/data/users/$USER"
export G1_REPO="$G1_ROOT/projects/g1-robust-locomotion"
export G1_ENV="$G1_ROOT/envs/humanoid/unitree-mjlab-py311"
export G1_CONDA="$G1_ROOT/envs/toolchains/miniforge3"
```

这些变量只对当前终端有效。

## 5. 部署前检查

### 系统建议

- Ubuntu 22.04 或兼容 Linux；
- RTX 4090，24GB 显存；
- 推荐至少 64GB 内存；
- 建议预留 30GB 以上磁盘空间；
- 能正常访问 GitHub 和 Python 包源；
- 已安装 `git`、`curl`/`wget`、编译工具和 `tmux`。

查看 GPU：

```bash
nvidia-smi
```

确认输出包含 RTX 4090，并记录：

- Driver Version；
- CUDA Version（它表示驱动最高支持版本，不等于当前 Python 环境的 CUDA runtime）；
- 当前显存占用；
- 是否有其他用户进程。

4090 是 Ada 架构，可以使用本文已验证的 PyTorch CUDA 13 环境；如果安装源没有完全相同的 wheel，也可以使用项目依赖支持的 CUDA 12.x PyTorch。不要仅凭 `nvidia-smi` 的 CUDA Version 决定安装命令。

## 6. 安装 Miniforge

创建目录：

```bash
mkdir -p "$G1_ROOT"/{projects,envs/humanoid,envs/toolchains,cache/warp,logs/unitree-mjlab,tmp}
```

下载并安装：

```bash
cd "$G1_ROOT/tmp"
wget -O Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh -b -p "$G1_CONDA"
```

不需要执行 `conda init`，避免影响其他项目的 shell。

## 7. 克隆项目并创建环境

推荐克隆本项目：

```bash
git clone https://github.com/robin1hub/g1-robust-locomotion.git "$G1_REPO"
cd "$G1_REPO"
```

创建 Python 3.11 环境：

```bash
"$G1_CONDA/bin/conda" create -y -p "$G1_ENV" python=3.11 pip
```

激活方式：

```bash
source "$G1_CONDA/etc/profile.d/conda.sh"
conda activate "$G1_ENV"
```

确认解释器：

```bash
which python
python --version
```

应指向 `$G1_ENV/bin/python`，Python 应为 3.11。

## 8. 安装依赖

先升级基础安装工具：

```bash
python -m pip install -U pip setuptools wheel
```

然后安装项目和经过验证的兼容约束：

```bash
cd "$G1_REPO"
python -m pip install -e .

python -m pip install \
  'mujoco==3.5.0' \
  'warp-lang==1.12.1' \
  'scipy==1.17.1'
```

这几个固定版本非常重要：

1. 项目使用 `mujoco-warp==3.5.0`，较新的 MuJoCo 可能修改或删除它依赖的枚举；
2. MjLab 1.2.0 会导入私有模块 `warp.context`，Warp 1.13 起删除该模块，因此使用 Warp 1.12.1；
3. 当前 MjLab terrain 模块会导入 SciPy，但依赖声明不完整，因此显式安装 SciPy。

这次实际验证的核心版本：

```text
Python       3.11.15
PyTorch      2.13.0+cu130
mjlab        1.2.0
mujoco-warp  3.5.0
mujoco       3.5.0
warp-lang    1.12.1
scipy        1.17.1
```

`pip install -e .` 会根据 MjLab 依赖安装 PyTorch。如果最终 PyTorch wheel 与服务器驱动不兼容，应从 PyTorch 官方源选择与驱动、4090 和 MjLab 兼容的 CUDA wheel，再重新运行项目安装。不要盲目升级 MuJoCo/Warp 来解决 PyTorch 问题。

检查依赖：

```bash
python -m pip check
```

## 9. 创建项目环境脚本

为了避免每次依赖 Conda shell，可以在项目根目录创建本机专用脚本 `env.local.sh`：

```bash
cat > env.local.sh <<EOF
#!/usr/bin/env bash
export PATH="$G1_ENV/bin:\${PATH}"
export LD_LIBRARY_PATH="$G1_ENV/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1
export WARP_CACHE_PATH="$G1_ROOT/cache/warp"
EOF
```

使用：

```bash
cd "$G1_REPO"
source env.local.sh
which python
```

`LD_LIBRARY_PATH` 优先使用环境中的 C++ runtime，可避免 Conda ICU 与 Ubuntu 系统 `libstdc++` ABI 不匹配。`PYTHONNOUSERSITE=1` 防止用户级 pip 包污染环境，Warp 缓存则放到数据盘。

不要把包含本机绝对路径的 `env.local.sh` 上传到公共仓库。

## 10. 分级验证

安装后不要直接启动全量训练。按下面顺序排错最省时间。

### 10.1 检查 PyTorch 和 GPU

```bash
source env.local.sh

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### 10.2 检查关键包

```bash
python - <<'PY'
import mujoco
import warp
import mjlab
print("mujoco:", mujoco.__version__)
print("warp:", warp.__version__)
print("mjlab imported")
PY
```

### 10.3 列出任务

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/list_envs.py
```

应能看到 `Unitree-G1-Flat` 和 `Unitree-G1-Rough`。

### 10.4 Flat smoke test

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs=16 \
  --env.scene.terrain.num-envs=16 \
  --agent.max-iterations=1 \
  --agent.save-interval=1 \
  --agent.run-name=flat_smoke \
  --agent.logger=tensorboard \
  --video=False
```

验收：完成 iteration 0，无 NaN/Inf，生成 `model_0.pt` 和 `policy.onnx`。

### 10.5 Rough smoke test

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python scripts/train.py Unitree-G1-Rough \
  --env.scene.num-envs=16 \
  --env.scene.terrain.num-envs=16 \
  --agent.max-iterations=1 \
  --agent.save-interval=1 \
  --agent.run-name=rough_smoke \
  --agent.logger=tensorboard \
  --video=False
```

首次 Rough 运行可能编译高度场碰撞和 raycast Warp 内核，等待几十秒到数分钟属于正常现象；后续会复用缓存。

## 11. 4090 环境数选择

RTX 4090 只有 24GB 显存，不能照搬 A100 80GB 的所有参数。显存不是唯一指标，环境数增加后可能出现吞吐边际收益下降。

建议分级测试：

### Flat

```text
先测 512 → 1024 → 2048 → 4096
```

### Rough

```text
先测 256 → 512 → 1024 → 2048
```

每个规模只运行 10 iterations，记录：

- `Steps per second`；
- `Collection time`；
- `Learning time`；
- `nvidia-smi` 显存和 GPU 利用率；
- 是否 OOM；
- 单 iteration 时间。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py Unitree-G1-Rough \
  --env.scene.num-envs=1024 \
  --env.scene.terrain.num-envs=1024 \
  --agent.max-iterations=10 \
  --agent.save-interval=10 \
  --agent.run-name=rough_scale_1024 \
  --video=False
```

选择“steps/s 较高且保留至少 2～4GB 显存余量”的规模，不要单纯追求显存占满。A100 实测 Rough 从 4096 增加到 8192 环境，显存约翻倍，但吞吐只提升约 17%，说明更多环境并不总是更快。

还要注意：项目按 PPO iteration 数训练。环境数翻倍会让每轮样本数翻倍；保持相同 iterations 会让总采样量翻倍，直接比较时并不公平。

## 12. 长时间训练

使用 `tmux`，避免 SSH 中断导致训练退出：

```bash
tmux new -s g1_flat
```

在 tmux 内：

```bash
cd "$G1_REPO"
source env.local.sh

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs=2048 \
  --env.scene.terrain.num-envs=2048 \
  --agent.max-iterations=10001 \
  --agent.save-interval=100 \
  --agent.run-name=g1_flat_4090 \
  --agent.logger=tensorboard \
  --video=False 2>&1 | tee "$G1_ROOT/logs/unitree-mjlab/g1_flat_4090.log"
```

这里的 2048 只是保守起点，应替换成前一步实测最优规模。

退出但保持运行：`Ctrl-b` 后按 `d`。重新进入：

```bash
tmux attach -t g1_flat
```

监控：

```bash
watch -n 2 nvidia-smi
tail -f "$G1_ROOT/logs/unitree-mjlab/g1_flat_4090.log"
```

## 13. TensorBoard

服务器启动：

```bash
cd "$G1_REPO"
source env.local.sh
tensorboard --logdir logs/rsl_rl/g1_velocity --host 127.0.0.1 --port 6006
```

本地电脑建立 SSH 隧道：

```bash
ssh -L 6006:127.0.0.1:6006 用户名@服务器地址
```

浏览器访问：

```text
http://127.0.0.1:6006
```

TensorBoard 用于查看 reward、episode length、跌倒、速度误差和 terrain level 曲线，不显示机器人实时画面。

## 14. 浏览器可视化

服务器没有桌面也可以使用 Viser。训练结束后加载 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py Unitree-G1-Flat \
  --checkpoint-file=/绝对路径/model_10000.pt \
  --num-envs=1 \
  --device=cuda:0 \
  --viewer=viser
```

Viser 默认监听服务器 8080 端口。本地建立隧道：

```bash
ssh -L 18080:127.0.0.1:8080 用户名@服务器地址
```

浏览器打开：

```text
http://127.0.0.1:18080
```

训练和 viewer 不建议同时使用同一张 4090。回放只需要 1～4 个环境，无需数千个环境。

## 15. 自动评测

本项目新增了无头评测脚本，它使用有限 episode 并保留训练时随机化：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py Unitree-G1-Flat \
  --checkpoint=/绝对路径/model_10000.pt \
  --num-envs=64 \
  --seeds '(42,43,44)' \
  --device=cuda:0
```

输出：

- `episodes.csv`：每个 episode 原始结果；
- `summary.csv`：checkpoint 汇总；
- `results.json`：配置与机器可读结果。

绘图：

```bash
python scripts/plot_evaluation.py \
  --summary-csv=evaluations/Unitree-G1-Flat/<timestamp>/summary.csv
```

关键指标：跌倒率、完整回合率、XY/yaw 速度 RMSE、接触滑移、动作变化 RMS、机身倾斜 RMS。

## 16. 恢复中断训练

先确认旧进程已经不存在：

```bash
tmux ls
pgrep -af 'scripts/train.py'
nvidia-smi
```

然后恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs=2048 \
  --env.scene.terrain.num-envs=2048 \
  --agent.max-iterations=10001 \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  --agent.resume=True \
  --agent.load-run=<原run目录名> \
  --agent.load-checkpoint='model_.*.pt' \
  --video=False
```

启动后检查日志是否明确打印了加载的 checkpoint。不要在旧训练仍运行时启动第二份恢复任务。

## 17. 常见问题

### `ModuleNotFoundError: warp.context`

原因：Warp 版本过新。处理：

```bash
python -m pip install --force-reinstall 'warp-lang==1.12.1'
```

### MuJoCo enum/API 报错

原因：`mujoco-warp==3.5.0` 搭配了过新的 MuJoCo。处理：

```bash
python -m pip install --force-reinstall 'mujoco==3.5.0'
```

### Terrain 模块找不到 SciPy

```bash
python -m pip install 'scipy==1.17.1'
```

### `libstdc++.so`、ICU 或 `GLIBCXX` 报错

先确认使用了环境 lib：

```bash
export LD_LIBRARY_PATH="$G1_ENV/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

### Rough 首次启动很慢

通常是在编译 Warp 高度场碰撞和 raycast 内核。检查 `WARP_CACHE_PATH` 可写并等待编译完成，不要立即判断为卡死。

### 显存占用不高是否异常

不一定。MuJoCo Warp 并行仿真可能受接触求解、raycast 或 kernel 调度限制，而不是显存容量限制。判断训练是否健康应看：

- GPU utilization；
- steps/s；
- collection time；
- 日志是否持续推进；
- reward 和 episode length 是否改善。

### 为什么 Rough 比 Flat 慢

Rough 多了 187 维高度扫描、更复杂的碰撞、更多接触容量和 terrain curriculum。真实 A100 测试中 Flat 4096 环境约 56k steps/s，Rough 4096 环境约 25.5k steps/s，差异是合理的。

## 18. Docker 说明

该 Unitree 仓库本身没有提供可直接使用的项目 Dockerfile。上游 MjLab 可能有容器方案，但共享服务器通常需要管理员授予 Docker 权限。对单台 4090 服务器，项目级 Conda 环境更容易控制版本和排错。

如果必须使用 Docker，应至少做到：

- 安装 NVIDIA Container Toolkit；
- 映射项目、日志和 Warp 缓存目录；
- 固定 MuJoCo、Warp、MjLab 和 PyTorch 版本；
- 使用 `--gpus` 明确限制显卡；
- 不把 checkpoint 写入容器临时层。

## 19. 推荐复现顺序

```text
1. 完成安装和 pip check
2. 验证 torch.cuda.is_available()
3. list_envs
4. Flat 16 env × 1 iteration
5. Rough 16 env × 1 iteration
6. 环境数扩展测试
7. Flat 短训练
8. Flat 全量训练
9. 自动评测和 Viser 回放
10. Rough 短训练与全量训练
```

不要跳过 smoke test，也不要用可视化模式进行全量训练。

## 20. 对项目的进一步理解

这个项目最有价值的部分并不是“成功跑出一个会走的机器人”，而是完整掌握以下链路：

- 将机器人控制问题定义为 MDP；
- 设计可部署 Actor 与特权 Critic；
- 用大规模并行仿真提高 on-policy PPO 采样效率；
- 用 reward、termination 和 curriculum 引导运动学习；
- 用 domain randomization 为 sim-to-real 做准备；
- 区分训练 reward、定量评测和主观视频效果；
- 理解高度感知只能提供几何信息，局部摩擦仍需要接触后的在线适应；
- 建立训练、checkpoint、TensorBoard、回放和批量评测的完整工程流程。

后续适合扩展的方向包括：

1. Rough 分地形评测；
2. 摩擦、质量、质心、推扰和控制延迟 benchmark；
3. 历史帧拼接、TCN、GRU 时序策略；
4. 机身速度、滑移和外力的显式状态重建；
5. 单帧 MLP 与历史策略的消融实验；
6. 真机条件允许时进行 sim-to-real 对齐。

在简历中，应只描述已经完成且有数据支持的内容，不应仅因使用某个名词就宣称实现了 Transformer、世界模型或真机迁移。
