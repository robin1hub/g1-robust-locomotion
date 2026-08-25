"""Unitree G1 velocity environment configurations."""

from copy import deepcopy

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.mdp.rewards import (
  adaptive_running_gait,
  forward_progress,
  forward_velocity_tracking_error_l2,
  normalized_mechanical_power,
  outward_lateral_velocity,
  phase_motion_joint_style,
  straight_track_lane_barrier_l4,
  straight_track_heading_error_l2,
  straight_track_lateral_position_l2,
  straight_track_progress,
  speed_dependent_torso_lean_l2,
  world_lateral_velocity_l2,
  world_yaw_rate_l2,
  yaw_rate_tracking_error_l2,
)
from src.tasks.velocity.mdp.observations import (
  adaptive_running_phase,
  straight_track_state,
  zero_track_state,
)
from src.tasks.velocity.mdp.velocity_command import (
  CategoricalVelocityCommandCfg,
  MixedVelocityCommandCfg,
)
from src.tasks.velocity.mdp.terminations import (
  outside_straight_lane,
  running_backwards,
)
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构建 G1 崎岖地形速度任务；平地任务会在此配置基础上做删减。"""
  # make_velocity_env_cfg 提供所有双足/腿式机器人共用的基础配置：仿真步长、
  # observation、reward、command、domain randomization 等；本函数再填入 G1 特有项。
  cfg = make_velocity_env_cfg()

  # 接触相关容量。崎岖地形接触更复杂，因此 CCD 和接触槽位设置得较大。
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}
  # get_g1_robot_cfg() 内含 MJCF、29个关节、默认姿态、执行器和 PD 参数。

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE
  # 策略输出不是直接力矩，而是归一化的关节位置增量/目标；scale 将每一维
  # 网络动作换算到各关节合适的角度范围，底层执行器再用 PD 跟踪。

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  # variable_posture 使用 std 控制“允许偏离默认姿态多少”：std 越小约束越严。
  # 站立时所有关节严格，走/跑时腿和摆臂关节允许更大幅度运动。
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # play 只用于展示策略：关闭观测噪声、推力扰动和 curriculum，避免演示过程
    # 因训练用随机化而频繁重置；训练配置本身不会走入这个分支。
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构建 G1 平地速度任务（当前建议首先阅读和运行的环境）。"""
  # 复用 Rough 的机器人、动作、奖励和随机化，只替换与地形有关的部分。
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  # 平面高度恒定，保留 height_scan 只会增加无信息的输入维度和计算量。

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_marathon_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Long-horizon, speed-and-efficiency focused running task for Unitree G1."""
  cfg = unitree_g1_flat_env_cfg(play=False)

  # Four proprioceptive frames let the actor infer translational motion without
  # exposing simulator-only base linear velocity at deployment time.
  cfg.observations["actor"].history_length = 4
  cfg.episode_length_s = 60.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (8.0, 12.0)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (0.5, 1.5)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  # Establish the running gait first. Robust pushes and rough tracks are added
  # only after this baseline can finish long episodes consistently.
  cfg.events.pop("push_robot", None)
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.4)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.75
  cfg.rewards["track_angular_velocity"].weight = 0.5
  cfg.rewards["body_orientation_l2"].weight = -0.5
  cfg.rewards["pose"].weight = 0.3
  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["angular_momentum"].weight = -0.005
  cfg.rewards["joint_acc_l2"].weight = -1.0e-7
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["foot_slip"].weight = -0.35
  cfg.rewards["soft_landing"].weight = -5.0e-4
  cfg.rewards.pop("stand_still", None)

  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=adaptive_running_gait,
    weight=0.8,
    params={
      "offset": [0.0, 0.5],
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
      "speed_range": (0.5, 4.0),
      "period_range": (0.55, 0.30),
      "stance_range": (0.55, 0.38),
    },
  )
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.14
  cfg.rewards["forward_progress"] = RewardTermCfg(
    func=forward_progress,
    weight=0.5,
    params={"max_speed": 4.5, "upright_power": 2.0},
  )
  cfg.rewards["mechanical_power"] = RewardTermCfg(
    func=normalized_mechanical_power,
    weight=-0.005,
    params={"power_scale": 1000.0},
  )

  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (0.5, 1.5)},
          {"step": 2500 * 24, "lin_vel_x": (0.8, 2.2)},
          {"step": 5000 * 24, "lin_vel_x": (1.2, 3.0)},
          {"step": 7500 * 24, "lin_vel_x": (1.5, 4.0)},
        ],
      },
    ),
    "power_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "mechanical_power",
        "weight_stages": [
          {"step": 0, "weight": -0.005},
          {"step": 5000 * 24, "weight": -0.01},
          {"step": 7500 * 24, "weight": -0.02},
        ],
      },
    ),
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Keep playback inside the speed range covered by the current M1 checkpoint.
    # Raise this after later curriculum stages have actually been trained.
    twist_cmd.ranges.lin_vel_x = (1.0, 1.5)

  return cfg


def unitree_g1_sprint_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Straight-track sprint task with explicit anti-spin and anti-drift terms."""
  cfg = unitree_g1_marathon_env_cfg(play=False)
  cfg.episode_length_s = 20.0

  # Every environment starts on its own world +X lane. A small yaw range keeps
  # reset diversity without recreating Marathon-v1's arbitrary-heading loophole.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "yaw": (-0.05, 0.05),
  }
  cfg.events["foot_friction"].params["ranges"] = (0.8, 1.2)
  cfg.events.pop("push_robot", None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (5.0, 8.0)
  twist_cmd.ranges.lin_vel_x = (0.8, 1.8)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  # Speed tracking remains the primary task, while world-frame progress and
  # lane terms make the meaning of "fast" unambiguous.
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.6
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.35
  cfg.rewards["body_orientation_l2"].weight = -0.8
  cfg.rewards["pose"].weight = 0.2
  cfg.rewards["body_ang_vel"].weight = -0.03
  cfg.rewards["angular_momentum"].weight = -0.003
  cfg.rewards["joint_acc_l2"].weight = -1.0e-7
  cfg.rewards["action_rate_l2"].weight = -0.03
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.01,
  )
  cfg.rewards["foot_slip"].weight = -0.4
  cfg.rewards["soft_landing"].weight = -7.5e-4
  cfg.rewards["foot_gait"].weight = 0.7
  cfg.rewards["foot_clearance"].weight = -0.4
  cfg.rewards["foot_clearance"].params["target_height"] = 0.14

  cfg.rewards["forward_progress"] = RewardTermCfg(
    func=straight_track_progress,
    weight=1.0,
    params={
      "max_speed": 4.5,
      "lane_half_width": 0.9,
      "upright_power": 2.0,
      "heading_power": 2.0,
    },
  )
  cfg.rewards["lane_position"] = RewardTermCfg(
    func=straight_track_lateral_position_l2,
    weight=-2.0,
  )
  cfg.rewards["lateral_velocity"] = RewardTermCfg(
    func=world_lateral_velocity_l2,
    weight=-0.5,
  )
  cfg.rewards["heading_error"] = RewardTermCfg(
    func=straight_track_heading_error_l2,
    weight=-1.5,
  )
  cfg.rewards["yaw_rate"] = RewardTermCfg(
    func=world_yaw_rate_l2,
    weight=-0.5,
  )
  cfg.rewards["mechanical_power"].weight = -0.002

  cfg.terminations["outside_lane"] = TerminationTermCfg(
    func=outside_straight_lane,
    params={"lane_half_width": 0.9},
  )
  cfg.terminations["running_backwards"] = TerminationTermCfg(
    func=running_backwards,
    params={"max_backward_distance": 0.5},
  )

  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (0.8, 1.8)},
          {"step": 2000 * 24, "lin_vel_x": (1.5, 2.5)},
          {"step": 4500 * 24, "lin_vel_x": (2.2, 3.2)},
          {"step": 7000 * 24, "lin_vel_x": (2.8, 4.0)},
        ],
      },
    ),
    "power_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "mechanical_power",
        "weight_stages": [
          {"step": 0, "weight": -0.002},
          {"step": 4500 * 24, "weight": -0.006},
          {"step": 7000 * 24, "weight": -0.01},
        ],
      },
    ),
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (1.5, 2.0)

  return cfg


def unitree_g1_sprint_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Sprint-v2 with explicit track-relative actor observations.

  Sprint-v2 rewards lane keeping in world coordinates, but its actor only sees
  body-relative proprioception and cannot know which way to steer back.  V3
  appends five current-frame track signals while preserving the old 392-feature
  ordering, enabling an exactly equivalent zero-column warm start.
  """
  cfg = unitree_g1_sprint_v2_env_cfg(play=False)

  actor_group = cfg.observations["actor"]
  actor_group.terms = {
    name: deepcopy(term) for name, term in actor_group.terms.items()
  }
  actor_group.history_length = None
  for term in actor_group.terms.values():
    term.history_length = 4
  actor_group.terms["track_state"] = ObservationTermCfg(
    func=straight_track_state,
    params={"lane_half_width": 0.9, "speed_scale": 4.0},
    clip=(-2.0, 2.0),
    history_length=0,
  )

  # Reserve roughly 500 additional iterations for learning the new steering
  # feedback before exposing the policy to faster commands.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.8, 1.8)},
    {"step": 2500 * 24, "lin_vel_x": (1.5, 2.5)},
    {"step": 5000 * 24, "lin_vel_x": (2.2, 3.2)},
    {"step": 7500 * 24, "lin_vel_x": (2.8, 4.0)},
  ]
  cfg.curriculum["power_weight"].params["weight_stages"] = [
    {"step": 0, "weight": -0.002},
    {"step": 5000 * 24, "weight": -0.006},
    {"step": 7500 * 24, "weight": -0.01},
  ]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (1.0, 1.8)

  return cfg


def unitree_g1_sprint_v3_lane_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Sprint-v3 with stronger directional lane-recovery shaping."""
  cfg = unitree_g1_sprint_v3_env_cfg(play=False)
  cfg.rewards["lane_barrier"] = RewardTermCfg(
    func=straight_track_lane_barrier_l4,
    weight=-2.0,
    params={"lane_half_width": 0.9},
  )
  cfg.rewards["outward_lateral_velocity"] = RewardTermCfg(
    func=outward_lateral_velocity,
    weight=-1.0,
    params={"center_deadzone": 0.1},
  )
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.8, 1.8)},
    {"step": 2800 * 24, "lin_vel_x": (1.5, 2.5)},
    {"step": 5300 * 24, "lin_vel_x": (2.2, 3.2)},
    {"step": 7800 * 24, "lin_vel_x": (2.8, 4.0)},
  ]
  cfg.curriculum["power_weight"].params["weight_stages"] = [
    {"step": 0, "weight": -0.002},
    {"step": 5300 * 24, "weight": -0.006},
    {"step": 7800 * 24, "weight": -0.01},
  ]
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (1.0, 1.8)
  return cfg


def unitree_g1_sprint_v4_adaptive_phase_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Sprint-v3-Lane with observation and gait reward on one adaptive clock."""
  cfg = unitree_g1_sprint_v3_lane_env_cfg(play=False)
  cfg.observations["actor"].terms["phase"].func = adaptive_running_phase
  cfg.observations["actor"].terms["phase"].params = {
    "command_name": "twist",
    "speed_range": (0.5, 4.0),
    "period_range": (0.55, 0.30),
  }
  # Isolate the phase change at the current 0.8-1.8 m/s range.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.8, 1.8)},
    {"step": 3200 * 24, "lin_vel_x": (1.5, 2.2)},
    {"step": 5000 * 24, "lin_vel_x": (2.0, 2.8)},
    {"step": 7000 * 24, "lin_vel_x": (2.5, 3.4)},
    {"step": 9000 * 24, "lin_vel_x": (3.0, 4.0)},
  ]
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (1.0, 1.8)
  return cfg


def unitree_g1_sprint_e2a_command_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Wide-lane low-level vx/vy/yaw command-tracking skill task."""
  cfg = unitree_g1_sprint_v4_adaptive_phase_env_cfg(play=False)
  cfg.episode_length_s = 12.0

  old_cmd = cfg.commands["twist"]
  assert isinstance(old_cmd, UniformVelocityCommandCfg)
  cfg.commands["twist"] = MixedVelocityCommandCfg(
    entity_name=old_cmd.entity_name,
    resampling_time_range=(3.0, 5.0),
    debug_vis=old_cmd.debug_vis,
    heading_command=False,
    rel_standing_envs=0.0,
    rel_straight_envs=0.5,
    init_velocity_prob=old_cmd.init_velocity_prob,
    ranges=MixedVelocityCommandCfg.Ranges(
      lin_vel_x=(1.0, 1.8),
      lin_vel_y=(-0.10, 0.10),
      ang_vel_z=(-0.15, 0.15),
      heading=None,
    ),
    viz=MixedVelocityCommandCfg.VizCfg(
      z_offset=old_cmd.viz.z_offset,
      scale=old_cmd.viz.scale,
    ),
  )

  # The low-level skill tracks body-frame commands. World-track objectives
  # would directly conflict with commanded strafing and turning.
  for reward_name in (
    "forward_progress",
    "lane_position",
    "lateral_velocity",
    "heading_error",
    "yaw_rate",
    "lane_barrier",
    "outward_lateral_velocity",
  ):
    cfg.rewards.pop(reward_name, None)
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_angular_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.3

  cfg.terminations["outside_lane"].params["lane_half_width"] = 2.0
  cfg.terminations["running_backwards"].params["max_backward_distance"] = 2.0
  cfg.observations["critic"].terms["phase"].func = adaptive_running_phase
  cfg.observations["critic"].terms["phase"].params = {
    "command_name": "twist",
    "speed_range": (0.5, 4.0),
    "period_range": (0.55, 0.30),
  }

  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (1.0, 1.8),
            "lin_vel_y": (-0.10, 0.10),
            "ang_vel_z": (-0.15, 0.15),
          },
          {
            "step": 200 * 24,
            "lin_vel_x": (1.0, 1.8),
            "lin_vel_y": (-0.20, 0.20),
            "ang_vel_z": (-0.30, 0.30),
          },
          {
            "step": 400 * 24,
            "lin_vel_x": (1.0, 1.8),
            "lin_vel_y": (-0.30, 0.30),
            "ang_vel_z": (-0.50, 0.50),
          },
        ],
      },
    ),
  }
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


def unitree_g1_sprint_e2b0_yaw_probe_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Yaw-task repair probe without world-frame lane constraints or symmetry."""
  cfg = unitree_g1_sprint_e2a_command_env_cfg(play=False)

  # Keep the 397-D model_350 input layout while preventing the low-level
  # controller from using world lane position/heading as a shortcut.
  cfg.observations["actor"].terms["track_state"].func = zero_track_state
  cfg.observations["actor"].terms["track_state"].params = {"size": 5}

  old_cmd = cfg.commands["twist"]
  assert isinstance(old_cmd, MixedVelocityCommandCfg)
  cfg.commands["twist"] = CategoricalVelocityCommandCfg(
    entity_name=old_cmd.entity_name,
    resampling_time_range=old_cmd.resampling_time_range,
    debug_vis=old_cmd.debug_vis,
    heading_command=False,
    rel_standing_envs=0.0,
    rel_straight_envs=0.35,
    rel_lateral_envs=0.25,
    rel_turn_envs=0.30,
    rel_combined_envs=0.10,
    init_velocity_prob=old_cmd.init_velocity_prob,
    ranges=CategoricalVelocityCommandCfg.Ranges(
      lin_vel_x=(1.0, 1.8),
      lin_vel_y=(-0.30, 0.30),
      ang_vel_z=(-0.15, 0.15),
      heading=None,
    ),
    viz=CategoricalVelocityCommandCfg.VizCfg(
      z_offset=old_cmd.viz.z_offset,
      scale=old_cmd.viz.scale,
    ),
  )

  # A correct body-frame turn necessarily leaves a straight world-frame lane.
  cfg.terminations.pop("outside_lane", None)
  cfg.terminations.pop("running_backwards", None)
  foot_geoms = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in range(1, 8)
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision$",
      exclude=foot_geoms,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (nonfoot_ground_cfg,)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.5
  cfg.rewards["yaw_rate_tracking_error"] = RewardTermCfg(
    func=yaw_rate_tracking_error_l2,
    weight=-0.5,
    params={"command_name": "twist"},
  )
  cfg.rewards["angular_momentum"].params["penalize_yaw"] = False

  # The probe trains only the ±0.15 and ±0.30 stages. The ±0.50 stage begins
  # after iteration 200 and is reserved for the follow-up full E2-B1 run.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (1.0, 1.8),
      "lin_vel_y": (-0.30, 0.30),
      "ang_vel_z": (-0.15, 0.15),
    },
    {
      "step": 100 * 24,
      "lin_vel_x": (1.0, 1.8),
      "lin_vel_y": (-0.30, 0.30),
      "ang_vel_z": (-0.30, 0.30),
    },
    {
      "step": 200 * 24,
      "lin_vel_x": (1.0, 1.8),
      "lin_vel_y": (-0.30, 0.30),
      "ang_vel_z": (-0.50, 0.50),
    },
  ]
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


def unitree_g1_sprint_e2b0a_task_fix_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """B0-A ablation: repair task termination/sampling but keep E2-A yaw rewards."""
  cfg = unitree_g1_sprint_e2b0_yaw_probe_env_cfg(play=play)
  cfg.rewards["track_angular_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.3
  cfg.rewards.pop("yaw_rate_tracking_error", None)
  cfg.rewards["angular_momentum"].params.pop("penalize_yaw", None)
  return cfg


def unitree_g1_sprint_e2b0b_reward_fix_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """B0-B ablation: task repair plus wider/non-saturating yaw rewards."""
  return unitree_g1_sprint_e2b0_yaw_probe_env_cfg(play=play)


def unitree_g1_sprint_e2b0b2_yaw_focus_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """B0-B2: hold ±0.15 yaw and increase yaw-active command sampling."""
  cfg = unitree_g1_sprint_e2b0b_reward_fix_env_cfg(play=play)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CategoricalVelocityCommandCfg)
  command_cfg.rel_straight_envs = 0.20
  command_cfg.rel_lateral_envs = 0.10
  command_cfg.rel_turn_envs = 0.60
  command_cfg.rel_combined_envs = 0.10
  if not play:
    # With weights-only restart, the 100-iteration probe remains at ±0.15.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (1.0, 1.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.15, 0.15),
      },
      {
        "step": 100 * 24,
        "lin_vel_x": (1.0, 1.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.15, 0.15),
      },
    ]
  return cfg


def unitree_g1_sprint_e2b0b3_yaw030_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """B0-B3: expand the verified yaw skill from ±0.15 to ±0.30 rad/s."""
  cfg = unitree_g1_sprint_e2b0b2_yaw_focus_env_cfg(play=play)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CategoricalVelocityCommandCfg)
  command_cfg.ranges.ang_vel_z = (-0.30, 0.30)
  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (1.0, 1.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.30, 0.30),
      },
      {
        "step": 100 * 24,
        "lin_vel_x": (1.0, 1.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.30, 0.30),
      },
    ]
  return cfg


def unitree_g1_sprint_e2b1_yaw050_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """E2-B1 probe: expand the verified low-level yaw skill to ±0.50 rad/s."""
  cfg = unitree_g1_sprint_e2b0b3_yaw030_env_cfg(play=play)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CategoricalVelocityCommandCfg)
  command_cfg.ranges.ang_vel_z = (-0.50, 0.50)
  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (1.0, 1.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
      {
        "step": 100 * 24,
        "lin_vel_x": (1.0, 1.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
    ]
  return cfg


def unitree_g1_sprint_s1_speed220_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-1 sprint task: preserve 3-D control while raising vx to 2.2 m/s."""
  cfg = unitree_g1_sprint_e2b1_yaw050_env_cfg(play=play)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CategoricalVelocityCommandCfg)
  command_cfg.rel_straight_envs = 0.60
  command_cfg.rel_lateral_envs = 0.10
  command_cfg.rel_turn_envs = 0.20
  command_cfg.rel_combined_envs = 0.10
  command_cfg.rel_speed_replay_envs = 0.20
  command_cfg.replay_lin_vel_x = (1.0, 1.8)
  command_cfg.ranges.lin_vel_x = (1.5, 2.2)

  cfg.rewards["forward_velocity_tracking_error"] = RewardTermCfg(
    func=forward_velocity_tracking_error_l2,
    weight=-0.5,
    params={"command_name": "twist"},
  )
  orientation_cfg = cfg.rewards["body_orientation_l2"]
  orientation_cfg.func = speed_dependent_torso_lean_l2
  orientation_cfg.params.update(
    {
      "command_name": "twist",
      "speed_range": (1.5, 2.2),
      "lean_range_deg": (2.0, 8.0),
    }
  )

  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (1.5, 2.2),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
      {
        "step": 300 * 24,
        "lin_vel_x": (1.5, 2.2),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
    ]
  return cfg


def unitree_g1_sprint_s2_speed280_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-2 sprint task: extend the verified speed envelope to 2.8 m/s."""
  cfg = unitree_g1_sprint_s1_speed220_env_cfg(play=play)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CategoricalVelocityCommandCfg)
  command_cfg.rel_speed_replay_envs = 0.25
  command_cfg.replay_lin_vel_x = (1.5, 2.2)
  command_cfg.ranges.lin_vel_x = (2.0, 2.8)

  # Preserve the Stage-1 lean target at low speeds and extend it smoothly for
  # faster running. Replay commands therefore retain their familiar posture.
  orientation_cfg = cfg.rewards["body_orientation_l2"]
  orientation_cfg.params.update(
    {
      "speed_range": (1.5, 2.8),
      "lean_range_deg": (2.0, 12.0),
    }
  )

  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (2.0, 2.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
      {
        "step": 400 * 24,
        "lin_vel_x": (2.0, 2.8),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
    ]
  return cfg


def unitree_g1_sprint_s3_speed340_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-3 sprint task: extend the verified speed envelope to 3.4 m/s."""
  cfg = unitree_g1_sprint_s2_speed280_env_cfg(play=play)
  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, CategoricalVelocityCommandCfg)
  command_cfg.rel_speed_replay_envs = 0.30
  command_cfg.replay_lin_vel_x = (2.0, 2.8)
  command_cfg.ranges.lin_vel_x = (2.5, 3.4)

  # Match the Stage-2 posture around the overlap and allow a larger forward
  # lean only near the new upper speed bound.
  orientation_cfg = cfg.rewards["body_orientation_l2"]
  orientation_cfg.params.update(
    {
      "speed_range": (2.0, 3.4),
      "lean_range_deg": (6.0, 16.0),
    }
  )

  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (2.5, 3.4),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
      {
        "step": 400 * 24,
        "lin_vel_x": (2.5, 3.4),
        "lin_vel_y": (-0.30, 0.30),
        "ang_vel_z": (-0.50, 0.50),
      },
    ]
  return cfg


def unitree_g1_sprint_s3_slip_w080_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """S3 slip ablation A: double only the contact-slip penalty."""
  cfg = unitree_g1_sprint_s3_speed340_env_cfg(play=play)
  cfg.rewards["foot_slip"].weight = -0.8
  return cfg


def unitree_g1_sprint_s3_natural_v1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """S3 natural-running fine-tuning with a phase-aligned motion style prior."""
  cfg = unitree_g1_sprint_s3_slip_w080_env_cfg(play=play)
  cfg.rewards["motion_joint_style"] = RewardTermCfg(
    func=phase_motion_joint_style,
    weight=1.5,
    params={
      "motion_file": "src/assets/motions/g1/lafan1_run1_subject2_112s_115s.npz",
      "frame_start": 69,
      "frame_end": 109,
      "command_name": "twist",
      "std": 0.45,
      "speed_range": (0.5, 4.0),
      "period_range": (0.55, 0.30),
      "minimum_speed": 1.5,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  return cfg
