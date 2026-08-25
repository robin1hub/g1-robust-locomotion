"""RL configuration for Unitree G1 velocity task."""

from dataclasses import asdict, dataclass
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class G1SymmetryPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config with the RSL-RL symmetry extension exposed to serialization."""

  symmetry_cfg: dict[str, Any] | None = None


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """G1 速度控制使用的 Actor-Critic 与 PPO 超参数。"""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Actor: observation -> 29维关节动作分布的均值；GaussianDistribution
      # 额外维护标准差，以便训练时探索。
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      # Critic 只估计状态价值。它可读取 critic observation（含特权信息），
      # 而部署到真机的 Actor 只依赖可获得的 actor observation。
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      # clip_param 限制新旧策略变化；entropy_coef 鼓励探索；gamma/lam 分别用于
      # 回报折扣与 GAE。adaptive 会依据 desired_kl 自动调节学习率。
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=100,
    num_steps_per_env=24,
    # 每轮先让每个并行环境采样24步。因此4096环境每轮约得到98304个样本。
    max_iterations=10001,
  )


def unitree_g1_symmetry_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """G1 PPO config using mirrored transition augmentation only."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.algorithm = G1SymmetryPpoAlgorithmCfg(
    **asdict(cfg.algorithm),
    symmetry_cfg={
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": (
        "src.tasks.velocity.mdp.symmetry:g1_lateral_symmetry"
      ),
    },
  )
  return cfg
