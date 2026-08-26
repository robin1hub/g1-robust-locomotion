# G1 自然奔跑项目：交接与跨服务器部署手册

最后更新：2026-08-25

本文件用于把当前项目迁移到另一台 Linux GPU 服务器，并让接手者能够完成安装验证、模型回放、定量评测和后续训练。它记录的是本项目的实际状态，不等同于上游仓库的通用安装说明。

## 1. 项目与当前结论

- GitHub：`https://github.com/robin1hub/g1-robust-locomotion`
- 主分支：`main`
- 自然跑步实验代码基线：`db49bb2`
- 上游：`https://github.com/unitreerobotics/unitree_rl_mjlab`
- 仿真后端：MuJoCo + MuJoCo Warp
- 强化学习：RSL-RL PPO
- 机器人：Unitree G1，29 个关节动作
- 策略网络：MLP Actor–Critic，不是 Transformer

当前完成了两条跑步路线：

1. **通用速度策略**：可接收 `vx/vy/yaw_rate` 命令，并通过分阶段课程达到约 3.21 m/s；但 3.4 m/s 时足滑约 0.610 m/s，动作存在直腿滑行，因此不是最终自然跑姿。
2. **自然跑步 tracking expert**：用加速后的 LAFAN1 重定向动作训练固定 3.397 m/s 专家。当前选择 `model_499.pt`，clean seed42×64 成功率 100%，Body MPKPE 0.0478 m，足滑 0.3337 m/s，保留屈膝、腾空和交替摆臂。

重要边界：`model_499.pt` 是固定动作跟踪专家，还不是能自由接收任意速度和转向命令的通用自然跑步策略。下一阶段应先做多 seed/扰动鲁棒性评测，再蒸馏到通用 velocity policy。

完整实验历史见：

- `PROGRESS.md`：里程碑和当前结论；
- `EXPERIMENT_LOG.md`：实验参数、失败分支与可复现记录；
- `DEPLOYMENT_4090_ZH.md`：Flat/Rough 基线及 4090 环境数建议；
- `EVALUATION.md`：通用评测说明。

## 2. GitHub 中没有哪些文件

以下目录或数据被 `.gitignore` 排除，不会随 `git clone` 下载：

- `logs/`：checkpoint、TensorBoard、ONNX 和训练视频；
- `artifacts/`：整理后的候选模型与视频；
- `evaluations/`：评测 CSV/JSON；
- `src/assets/motions/g1/lafan1_*.npz`：原始与加速动作文件；
- 本机 Conda 环境和 Warp/Pip 缓存。

因此，只有源码时可以重新训练，但**不能直接回放当前最佳模型**。若要无损接续当前进度，必须从旧服务器额外复制第 5 节列出的文件。

## 3. 新服务器资源和使用规范

### 3.1 建议资源

- Ubuntu 22.04 或兼容 Linux；
- NVIDIA 驱动 550 或更高；
- Python 3.11；
- 单卡最低建议 24 GB 显存，A100 80GB 已验证 4096 并行环境；
- 内存建议至少 64 GB；
- 只复现代码建议预留 20 GB，保留环境、日志和多轮训练建议 50 GB 以上。

RTX 4090 不要直接假设能使用与 A100 相同的环境数。建议 tracking 从 512 或 1024 环境开始，按 `1024 → 2048 → 4096` 测吞吐和显存，保留 2～4 GB 余量。

### 3.2 共享服务器规则

1. 运行前用 `nvidia-smi` 检查空闲卡和所属进程，不占用他人 GPU；
2. 始终用 `CUDA_VISIBLE_DEVICES` 只暴露获准使用的卡；
3. 大文件放数据盘，不放 `/home`；
4. 每个项目使用独立环境，不修改系统 Python、系统 CUDA 或共享环境；
5. 长任务放入 `tmux` 或管理员提供的 Slurm；
6. 只结束自己的进程，不操作其他用户任务；
7. 密码、Token 和 SSH 私钥不得写入仓库、日志或共享目录；
8. 若服务器启用了 Slurm，以管理员的 `srun/sbatch` 规范为准，不直接抢卡。

## 4. 推荐目录布局

以下命令均假设新服务器的数据根目录为 `/data/users/$USER`。如果管理员分配的是 `/workspace/$USER`，整体替换 `G1_ROOT` 即可。

```bash
export G1_ROOT="/data/users/$USER"
export G1_REPO="$G1_ROOT/projects/g1-robust-locomotion"
export G1_ENV="$G1_ROOT/envs/humanoid/unitree_rl_mjlab-py311"
export G1_CONDA="$G1_ROOT/envs/toolchains/miniforge3"
export G1_CACHE="$G1_ROOT/cache"
export G1_TMP="$G1_ROOT/tmp"

mkdir -p \
  "$G1_ROOT/projects" \
  "$G1_ROOT/envs/humanoid" \
  "$G1_ROOT/envs/toolchains" \
  "$G1_CACHE/warp" \
  "$G1_TMP"
```

不要复用 `$HOME`、系统 Conda 或其他项目的 `.venv`。

## 5. 从旧服务器复制最小资产

### 5.1 必需资产及校验值

| 文件 | 用途 | 大小 | SHA256 |
|---|---|---:|---|
| `src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz` | 原始约 2.066 m/s 重定向动作 | 268802 B | `58f9d729ceba657248750c620230a91edfa5c98468923795ac0f1a9f404d584f` |
| `src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz` | 3.397 m/s 加速动作 | 164866 B | `f5c2097293889135dae27050400e8e4f1b8f6597ef746732933f12f56531bc52` |
| `artifacts/sprint_tracking_speed340/model499/model_499.pt` | 当前自然跑步 expert | 6769187 B | `bca589b65f2d9972dc288142d77d43861f738178f9221d5dbdb65bcb5bbe91f5` |

推荐一并迁移：

- `logs/rsl_rl/g1_tracking/2026-08-25_06-51-21_tracking_sprint340_probe_4096_500/`：完整500轮 tracking run；
- `evaluations/Unitree-G1-Tracking/sprint340_model499_clean_seed42/`：当前 clean 评测；
- `artifacts/sprint_tracking_speed340/model499/videos/`：人工验收视频；
- `logs/rsl_rl/g1_velocity/2026-08-25_03-31-41_sprint_s3_speed340_4096_400/model_375.pt`：通用3.4 m/s速度策略候选，SHA256 `ff4b01680714e7f428304838b9c715ae0d3b7f53b6e25232af8bbbd70b3299ae`。

### 5.2 使用 rsync 迁移

先在新服务器完成第 6 节的代码克隆。然后在旧服务器执行，替换目标登录名、主机和路径：

```bash
export OLD_REPO="/data/users/yanghao/projects/unitree_rl_mjlab"
export NEW_HOST="username@new-server"
export NEW_REPO="/data/users/username/projects/g1-robust-locomotion"

ssh "$NEW_HOST" "mkdir -p \
  '$NEW_REPO/src/assets/motions/g1' \
  '$NEW_REPO/artifacts/sprint_tracking_speed340/model499' \
  '$NEW_REPO/logs/rsl_rl/g1_tracking' \
  '$NEW_REPO/evaluations/Unitree-G1-Tracking'"

rsync -avP \
  "$OLD_REPO/src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz" \
  "$OLD_REPO/src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz" \
  "$NEW_HOST:$NEW_REPO/src/assets/motions/g1/"

rsync -avP \
  "$OLD_REPO/artifacts/sprint_tracking_speed340/model499/" \
  "$NEW_HOST:$NEW_REPO/artifacts/sprint_tracking_speed340/model499/"

rsync -avP \
  "$OLD_REPO/logs/rsl_rl/g1_tracking/2026-08-25_06-51-21_tracking_sprint340_probe_4096_500/" \
  "$NEW_HOST:$NEW_REPO/logs/rsl_rl/g1_tracking/2026-08-25_06-51-21_tracking_sprint340_probe_4096_500/"

rsync -avP \
  "$OLD_REPO/evaluations/Unitree-G1-Tracking/sprint340_model499_clean_seed42/" \
  "$NEW_HOST:$NEW_REPO/evaluations/Unitree-G1-Tracking/sprint340_model499_clean_seed42/"
```

迁移后在新服务器校验：

```bash
cd "$G1_REPO"
sha256sum \
  src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  artifacts/sprint_tracking_speed340/model499/model_499.pt
```

如果无法访问旧服务器，只要有原始动作 NPZ，可以重新生成加速动作：

```bash
python scripts/time_scale_motion.py \
  --input-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz \
  --output-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --speed-scale 1.645721
```

但 `model_499.pt` 无法由 GitHub 恢复，只能复制或重新训练。

## 6. 获取源码与 GitHub 登录

公开仓库可直接使用 HTTPS：

```bash
git clone https://github.com/robin1hub/g1-robust-locomotion.git "$G1_REPO"
cd "$G1_REPO"
git switch main
git pull --ff-only origin main
git log -1 --oneline
```

需要推送时建议为新服务器用户创建独立 SSH key，不要复制其他服务器的私钥：

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
cat "$HOME/.ssh/id_ed25519.pub"
```

把公钥添加到 GitHub 后，将远端切换为 SSH 并测试：

```bash
git remote set-url origin git@github.com:robin1hub/g1-robust-locomotion.git
ssh -T git@github.com
```

## 7. 安装独立 Python 环境

### 7.1 安装 Miniforge

若服务器已有个人 Conda，可跳过下载；不要要求 root 权限。

```bash
cd "$G1_TMP"
wget -O Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$G1_CONDA"
```

### 7.2 创建并激活环境

```bash
"$G1_CONDA/bin/conda" create -y -p "$G1_ENV" python=3.11 pip
source "$G1_CONDA/etc/profile.d/conda.sh"
conda activate "$G1_ENV"

python --version
which python
```

### 7.3 安装项目

```bash
cd "$G1_REPO"
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install \
  'mujoco==3.5.0' \
  'warp-lang==1.12.1' \
  'scipy==1.17.1' \
  tensorboard
python -m pip check
```

当前旧服务器已验证的核心环境：

```text
Python          3.11.15
torch           2.13.0
mjlab           1.2.0
mujoco          3.5.0
mujoco-warp     3.5.0
warp-lang       1.12.1
numpy           2.4.6
scipy           1.17.1
rsl-rl-lib      5.0.1
tensorboard     2.21.0
tyro            1.0.15
```

其中 MuJoCo、MuJoCo Warp、Warp 和 MjLab 的组合已经验证，不建议随意升级。PyTorch 必须选择与新服务器 NVIDIA 驱动兼容的 CUDA wheel；`nvidia-smi` 显示的是驱动最高支持 CUDA 版本，不是 Python 环境实际 CUDA runtime。

### 7.4 每次登录后的环境变量

```bash
source "$G1_CONDA/etc/profile.d/conda.sh"
conda activate "$G1_ENV"
export LD_LIBRARY_PATH="$G1_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export WARP_CACHE_PATH="$G1_CACHE/warp"
export MPLCONFIGDIR="$G1_TMP/matplotlib"
export MUJOCO_GL=egl
mkdir -p "$WARP_CACHE_PATH" "$MPLCONFIGDIR"
cd "$G1_REPO"
```

可以把这些内容保存为新服务器本机脚本，但不要把带绝对路径的本机配置提交到 GitHub。

## 8. 分级安装验收

### 8.1 GPU 和关键包

```bash
nvidia-smi

CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch, mujoco, warp, mjlab
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mujoco:", mujoco.__version__)
print("warp:", warp.__version__)
PY
```

### 8.2 任务注册

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/list_envs.py | grep 'Unitree-G1'
```

至少应能找到：

- `Unitree-G1-Flat`
- `Unitree-G1-Rough`
- `Unitree-G1-Tracking`
- `Unitree-G1-Tracking-Robust`
- `Unitree-G1-Sprint-S3-Speed340`
- `Unitree-G1-Sprint-S3-Natural-v1`

### 8.3 16 环境 smoke test

先确认 GPU 0 确实空闲，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --env.scene.num-envs 16 \
  --env.scene.terrain.num-envs 16 \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.run-name migration_smoke_16 \
  --agent.logger tensorboard \
  --gpu-ids '[0]' \
  --video False
```

验收条件：能完成2轮、没有 NaN/Inf/CUDA error，并生成 checkpoint 和 ONNX。首次启动会编译 Warp 内核，耗时较长属于正常现象。

## 9. 回放当前最佳自然跑步模型

### 9.1 有桌面的服务器

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py Unitree-G1-Tracking \
  --checkpoint-file artifacts/sprint_tracking_speed340/model499/model_499.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --num-envs 1 \
  --device cuda:0 \
  --viewer native
```

### 9.2 无桌面服务器：浏览器查看

服务器运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py Unitree-G1-Tracking \
  --checkpoint-file artifacts/sprint_tracking_speed340/model499/model_499.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --num-envs 1 \
  --device cuda:0 \
  --viewer viser
```

本地电脑建立 SSH 隧道：

```bash
ssh -L 18080:127.0.0.1:8080 username@new-server
```

浏览器打开 `http://127.0.0.1:18080`。TensorBoard 只能看曲线；Viser/录制视频才显示机器人动作。

### 9.3 无桌面服务器：录制 MP4

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/play.py Unitree-G1-Tracking \
  --checkpoint-file artifacts/sprint_tracking_speed340/model499/model_499.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --num-envs 1 \
  --device cuda:0 \
  --video True \
  --video-length 90 \
  --video-height 720 \
  --video-width 1280 \
  --no-terminations True
```

视频默认写入 checkpoint 附近的 `videos/play/`。录制只需要1个环境，不要为视频占用多张卡。

## 10. 复现 clean 评测

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_tracking_robustness.py \
  Unitree-G1-Tracking \
  --checkpoint artifacts/sprint_tracking_speed340/model499/model_499.pt \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --num-envs 64 \
  --seeds '(42,)' \
  --scenarios "('clean',)" \
  --device cuda:0 \
  --output-dir evaluations/Unitree-G1-Tracking/migration_model499_clean_seed42
```

目标复现值允许因硬件/数值实现略有波动：

| 指标 | 当前旧服务器结果 |
|---|---:|
| 成功率 | 1.000 |
| Body MPKPE | 0.04776 m |
| Joint position RMSE | 0.12861 rad |
| Foot slip | 0.33367 m/s |
| Root displacement | 5.60531 m |

若成功率不是100%，先检查动作文件和模型 SHA256、任务名、依赖版本及是否错误启用了随机扰动，不要直接继续训练。

## 11. 继续训练

仓库中的部分 `scripts/run_g1_*.sh` 是原服务器实验记录，包含 `/data/users/yanghao/...`、固定物理 GPU 编号和旧 run 名。迁移后不要不加检查地直接执行它们；优先使用本节的参数化命令，或先把脚本中的项目路径、环境路径、GPU 和 checkpoint 全部替换成新服务器值。

### 11.1 同一 tracking 任务继续优化

把 `model_499.pt` 放在下面的兼容目录中：

```text
logs/rsl_rl/g1_tracking/migrated_sprint340/model_499.pt
```

然后使用权重级恢复，重置旧优化器和 iteration：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --env.scene.num-envs 1024 \
  --env.scene.terrain.num-envs 1024 \
  --agent.max-iterations 200 \
  --agent.save-interval 25 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.algorithm.clip-param 0.1 \
  --agent.algorithm.desired-kl 0.005 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.run-name migrated_sprint340_continue \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run migrated_sprint340 \
  --agent.load-checkpoint model_499.pt \
  --weights-only-resume True \
  --gpu-ids '[0]' \
  --video False
```

`1024` 是新服务器保守起点，不是最终最优值。smoke 正常后再增加环境数。

### 11.2 多 GPU

只有获得多卡使用权限且确实能提升吞吐时才使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py Unitree-G1-Tracking-Robust \
  --motion-file src/assets/motions/g1/lafan1_run1_subject2_112s_115s_speed340.npz \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.num-envs 4096 \
  --agent.max-iterations 200 \
  --gpu-ids '[0,1,2,3]' \
  --video False
```

项目会启动每卡一个 worker，并把同一份环境配置传给每个 worker，因此示例中的4096是**每个 worker**的环境数，4卡合计16384个环境。迁移初期应先降低单卡环境数，并确认日志中的 rank/device、总采样量和吞吐符合预期。不要仅凭显存空余就盲目增加环境。

### 11.3 长任务与监控

```bash
tmux new -s g1_sprint
```

训练启动后用以下方式检查：

```bash
watch -n 2 nvidia-smi
ps -u "$USER" -f | grep 'scripts/train.py'
tensorboard --logdir "$G1_REPO/logs/rsl_rl" --host 127.0.0.1 --port 6006
```

本地建立 TensorBoard 隧道：

```bash
ssh -L 6006:127.0.0.1:6006 username@new-server
```

浏览器访问 `http://127.0.0.1:6006`。

## 12. 当前关键实验资产

| 分支 | checkpoint | 结论 |
|---|---|---|
| E2-C 三维命令 + symmetry | `2026-08-24_03-56-37_sprint_e2c_symmetry_aug_4096_100/model_99.pt` | 低层 `vx/vy/yaw` 技能基线 |
| S1 2.2 m/s | `2026-08-24_06-28-40_sprint_s1_speed220_4096_300/model_299.pt` | 2.131 m/s，多seed零失败 |
| S2 2.8 m/s | `2026-08-25_02-59-46_sprint_s2_speed280_4096_400/model_399.pt` | 2.684 m/s，多seed零失败 |
| S3 3.4 m/s velocity | `2026-08-25_03-31-41_sprint_s3_speed340_4096_400/model_375.pt` | 3.212 m/s，足滑0.610，跑姿不自然 |
| S3 Natural-v1 | `2026-08-25_06-30-40_sprint_s3_natural_v1_4096_200/model_199.pt` | 弱风格奖励改善不足，视觉否决 |
| Tracking sprint expert | `2026-08-25_06-51-21_tracking_sprint340_probe_4096_500/model_499.pt` | 当前自然跑姿最佳 |

这些 checkpoint 默认都不在 GitHub。若下一阶段要做 expert-to-policy 蒸馏，至少迁移 tracking `model_499`、velocity S3 `model_375`、原始动作和3.4 m/s动作。

### 12.1 自然冲刺参考（2026-08-26）

`speed400` 是普通 `run1` 片段的时间缩放版本，不是原始冲刺动作。下一阶段应优先使用 LAFAN1 官方 `Sprint` 类别的 `sprint1`。官方 BVH 已缓存到：

```text
/data/users/root/cache/g1-robust/lafan1-official/sprint/sprint1_subject2.bvh
/data/users/root/cache/g1-robust/lafan1-official/sprint/sprint1_subject4.bvh
```

已下载的 G1 29-DoF 重定向 CSV 位于：

```text
/data/users/root/cache/g1-robust/lafan1-retargeted-g1/sprint1_subject2.csv
/data/users/root/cache/g1-robust/lafan1-retargeted-g1/sprint1_subject4.csv
```

候选窗口转换后均为50 Hz、99帧：

| 候选 | 原始帧范围（零基、右开） | NPZ | 参考速度 | 结论 |
|---|---:|---|---:|---|
| S4-A | `3115:3175` | `lafan1_sprint1_subject4_103p83s_105p83s.npz` | 3.860 m/s | 最快，波动和肩部不对称略大 |
| S4-B | `3775:3835` | `lafan1_sprint1_subject4_125p83s_127p83s.npz` | 3.800 m/s | **拒绝：末段减速并后倾制动** |
| S2-A | `3570:3630` | `lafan1_sprint1_subject2_119s_121s.npz` | 3.536 m/s | 髋部侧摆和根部偏航偏大 |

复核发现 S4-B 最后约0.4秒从约4.23 m/s减速到3.18 m/s，躯干由前倾转为平均后倾8度，末帧约后倾13度，不能作为主训练参考。完整动作扫描后改用：

| 用途 | 原始帧范围（零基、右开） | NPZ | 平均速度 | 姿态结论 |
|---|---:|---|---:|---|
| 已拒绝 | `5019:5065` | `lafan1_sprint1_subject4_167p30s_168p80s_steady.npz` | 3.990 m/s | 约1.04秒后开始转为直立减速姿态 |
| 课程备选 | `3866:3913` | `lafan1_sprint1_subject4_128p87s_130p40s_accel.npz` | 3.948 m/s | 加速冲刺，前倾更强，速度约3.03到4.28 m/s |

复核表明，根速度统计无法可靠识别人体已经进入减速姿态，因此目前没有任何 LAFAN1 `sprint1` 窗口被批准为恒速冲刺主参考。加速段只可用于以后带阶段标签的课程，不能单独长期训练。

新的恒速候选来自 MotionDecode 的 `BG_Sprint_run_00687`：从高速平台提取原始帧 `506:578` 的0.60秒完整步态周期，进行小幅首尾闭合后重复5次，并把根速度固定为4.3 m/s：

```text
src/assets/motions/g1/motiondecode_BG_Sprint_run_00687_cycle_periodic_v430.npz
```

该候选共150帧、2.98秒，净速度4.301 m/s，躯干前倾始终为21.7到27.1度，每0.60秒关节姿态闭合。它已消除停止/制动阶段，但重定向参考仍有支撑脚横向滑动，因此状态是“待抗滑训练验证”，尚未批准为最终参考。生成脚本为 `scripts/build_periodic_sprint_reference.py`。数据源：`https://huggingface.co/datasets/CMRobot/MotionDecode`。

2026-08-26 已在6张共享 RTX 4090 上完成短适配验证：每卡512环境，共3072环境；先跑5轮探针，再从探针 `model_4.pt` 跑20轮。共享训练峰值额外占用约为 GPU0 2.9 GiB、其余卡各0.9 GiB，训练期间采样利用率约13%到21%，退出后显存全部回到原进程基线。20轮末 episode length 仅24.12步（约0.48秒），anchor linear velocity error 1.5031 m/s，foot slip 0.8924，说明直接从 `speed400` 权重切换到该4.3 m/s参考没有成功。不要继续堆叠该实验轮数；下一步应采用逐级速度课程，并在早期阶段关闭或放宽高频终止与强随机化，待存活时间稳定后再恢复抗滑约束。

对应运行目录：

```text
logs/rsl_rl/g1_tracking/2026-08-26_07-03-39_tracking_sprint430_periodic_shared_6gpu_512_each_probe5
logs/rsl_rl/g1_tracking/2026-08-26_07-07-15_tracking_sprint430_periodic_shared_6gpu_512_each_adapt20
```

数据来源：Ubisoft 官方 LAFAN1 仓库 `https://github.com/ubisoft/ubisoft-laforge-animation-dataset`；G1 重定向数据 `https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1`。LAFAN1 官方许可为 CC BY-NC-ND 4.0，外发或重新发布重定向资产前必须单独确认许可条件；原始 CSV/BVH 不提交到本仓库。

### 12.2 MotionDecode 00435 速度课程结果（2026-08-26）

`BG_Sprint_run_00435` 的1.48秒高速窗口没有停止或制动尾段，原始参考净速度为
4.371 m/s。保持空间姿态不变并逐级缩短动作时长，比直接切换到00687周期动作稳定得多。

| 参考速度 | 选取 checkpoint | clean seed42 实际速度 | 成功率 | Body MPKPE | Foot slip |
|---:|---|---:|---:|---:|---:|
| 4.371 m/s | `07-39-14.../model_99.pt` | 3.819 m/s | 98.4% | 0.0589 m | 0.475 m/s |
| 4.828 m/s | `07-51-54.../model_99.pt` | 4.266 m/s | 100% | 0.0548 m | 0.545 m/s |
| 5.218 m/s | `07-57-48.../model_99.pt` | 4.630 m/s | 100% | 0.0574 m | 0.630 m/s |
| 5.577 m/s | `08-06-06.../model_80.pt` | 4.865 m/s | 100% | 0.0657 m | 0.732 m/s |
| 5.577 m/s + 防滑 | `08-14-49.../model_20.pt` | 4.859 m/s | 100% | 0.0657 m | 0.652 m/s |

最终选择最后一行作为当前高速均衡模型。轻量 `-0.05` 支撑脚滑移惩罚相对未防滑
`model_80` 将滑移降低约10.9%，速度只降低约0.1%。不要用防滑 run 的最后一轮：
`model_59` 虽达到4.887 m/s，但滑移回升到0.719 m/s。

关键资产及校验值：

```text
src/assets/motions/g1/motiondecode_BG_Sprint_run_00435_fast1p5s_speed560.npz
SHA256 88d2877760d0f6ce02b934d7b877eff610ed7d7ac0c97c443f94303b023335e7

logs/rsl_rl/g1_tracking/2026-08-26_08-14-49_tracking_motiondecode00435_v558_antislip005_from_model80_shared_6gpu_384_60/model_20.pt
SHA256 d6cdb3f51937f370950b4b281143b56fcb05b04fafbb6823fd1fe41652c5227c
```

三随机种子、每种子64环境的最终复评：

| 场景 | episodes | 成功率 | 实际速度 | Body MPKPE | Foot slip |
|---|---:|---:|---:|---:|---:|
| clean | 192 | 100% | 4.846 m/s | 0.065 m | 0.674 m/s |
| friction 0.6 | 192 | 99.5% | 4.823 m/s | 0.067 m | 0.719 m/s |
| friction 0.4 | 192 | 99.5% | 4.768 m/s | 0.071 m | 0.805 m/s |
| 0.25 m/s侧向推力 | 192 | 100% | 4.723 m/s | 0.065 m | 0.677 m/s |
| 0.50 m/s侧向推力 | 192 | 100% | 4.619 m/s | 0.066 m | 0.656 m/s |

推力在0.6秒施加；默认1.5秒晚于本动作结束，不能用于这个短动作的有效测试。结果位于：

```text
evaluations/Unitree-G1-Tracking/motiondecode00435_v558_antislip005_model20_robust_3seed64
```

最终离屏视频和接触表：

```text
logs/rsl_rl/g1_tracking/2026-08-26_08-14-49_tracking_motiondecode00435_v558_antislip005_from_model80_shared_6gpu_384_60/videos/play/rl-video-step-0.mp4
artifacts/sprint_tracking_motiondecode00435/model20_v558_antislip005_contact_sheet.jpg
```

录像中策略保持连续抬膝冲刺，没有停止尾段、后仰制动、倒地或异常扭转。策略相对绿色参考仍有
位移滞后，因此参考5.577 m/s不能当作策略真实速度；报告速度必须用独立评估位移除以时长。

课程训练入口为 `scripts/run_g1_tracking_sprint_curriculum_shared.sh`，需显式设置
`ALLOW_SHARED_GPUS=1`。本轮使用6卡、每卡384环境，训练退出后六卡都回到原进程显存基线。
新增任务 `Unitree-G1-Tracking-AntiSlip` 只在普通平地追踪任务上增加轻量滑移惩罚，不包含
`Unitree-G1-Tracking-Robust-AntiSlip` 的 actuator delay、payload 和强动力学随机化。

## 13. 下一阶段计划

建议严格按顺序推进：

1. 在新服务器复现 model499 clean seed42×64；
2. 对 model499 做5个 seed 的 clean 复评；
3. 加入摩擦、局部低摩擦、推扰动、控制延迟和电机强度场景；
4. 确认动作自然性在扰动下没有崩坏；
5. 采集 expert 的观测—动作轨迹；
6. 将 tracking expert 蒸馏到可接收 `vx/vy/yaw_rate` 的 velocity actor；
7. 用 PPO 小学习率微调，保留命令跟踪、足滑和动作风格三组指标；
8. 通过仿真部署验证后，再讨论真机 Sim2Real。

不要直接从 `model_499` 宣称已经得到通用跑步策略，也不要在未完成仿真部署和安全测试前上真机。

## 14. 常见故障

### `No CUDA GPUs are available`

- 检查 `nvidia-smi`；
- 检查 `CUDA_VISIBLE_DEVICES` 是否映射了存在且获准使用的卡；
- 检查是否在不允许 GPU 的容器/沙箱中；
- 不要用“显存看起来空”替代 CUDA 可用性测试。

### `ModuleNotFoundError: warp.context`

Warp 过新：

```bash
python -m pip install --force-reinstall 'warp-lang==1.12.1'
```

### MuJoCo Warp 枚举或 API 错误

```bash
python -m pip install --force-reinstall \
  'mujoco==3.5.0' \
  'mujoco-warp==3.5.0'
```

### `GLIBCXX`、ICU 或 `libstdc++.so` 错误

```bash
export LD_LIBRARY_PATH="$G1_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

并确认 `which python` 指向项目环境。

### 找不到 motion file

GitHub 不包含 LAFAN1 NPZ。按第5节复制或重新生成，不要把动作数据硬编码进源码。

### 找不到 checkpoint

GitHub 不包含 `.pt`。检查 rsync 目标、SHA256 和 `--checkpoint-file` 绝对/相对路径。恢复训练时 checkpoint 还必须位于 `logs/rsl_rl/<experiment>/<load-run>/` 结构下。

### 显存很低是否异常

不一定。判断训练是否正常应看 GPU utilization、steps/s、iteration 是否推进、episode length 和 reward，而不是只看显存。环境数过少时可以逐级增加，但需要比较吞吐，更多环境不一定更快。

### 服务器没有图形界面

全量训练使用 `MUJOCO_GL=egl`。查看动作使用 Viser + SSH 隧道，或离屏录制 MP4；不要为训练开启 native viewer。

## 15. 最终迁移验收清单

- [ ] `git log -1` 位于本项目 `main` 最新版本；
- [ ] Python 3.11 独立环境安装完成，`pip check` 通过；
- [ ] `torch.cuda.is_available()` 为 True，GPU 名称正确；
- [ ] `mujoco==3.5.0`、`mujoco-warp==3.5.0`、`warp-lang==1.12.1`；
- [ ] 两个 motion NPZ 与 `model_499.pt` 的 SHA256 正确；
- [ ] 任务列表包含 tracking 和 sprint 自定义任务；
- [ ] 16环境×2轮 smoke test 通过；
- [ ] model499 能录制或浏览器回放完整1.8秒动作；
- [ ] clean seed42×64 成功率100%，指标接近第10节；
- [ ] TensorBoard 和 SSH 隧道可访问；
- [ ] 长任务使用 tmux/Slurm，GPU 使用符合新服务器规范；
- [ ] 日志、模型和缓存写入数据盘；
- [ ] 本项目无残留重复训练进程后再启动下一实验。

完成以上项目后，才算迁移成功；仅仅 `git clone` 和安装依赖不代表当前实验状态已经恢复。
