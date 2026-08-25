"""Unitree G1 flat tracking environment configurations."""

from dataclasses import replace

from mjlab.actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
import src.tasks.tracking.mdp as tracking_mdp


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_robust_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """G1 tracking with staged joint dynamics randomization."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  if play:
    return cfg

  robot_cfg = cfg.scene.entities["robot"]
  if robot_cfg.articulation is None:
    raise RuntimeError("Robot has no articulation configuration")
  robot_cfg.articulation = replace(
    robot_cfg.articulation,
    actuators=tuple(
      DelayedActuatorCfg(
        base_cfg=actuator_cfg,
        delay_target="position",
        delay_min_lag=0,
        delay_max_lag=8,
      )
      for actuator_cfg in robot_cfg.articulation.actuators
    ),
  )

  # Re-sample dynamics on every episode so each policy update sees joint
  # combinations instead of assigning one fixed model to each environment.
  cfg.events["foot_friction"].mode = "reset"
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)
  cfg.events["motor_strength"] = EventTermCfg(
    mode="reset",
    func=tracking_mdp.delayed_actuator_effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": (0.8, 1.0),
    },
  )
  cfg.events["action_delay"] = EventTermCfg(
    mode="reset",
    func=dr.sync_actuator_delays,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "lag_range": (0, 2),
    },
  )
  cfg.events["torso_payload"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "operation": "add",
      "ranges": (0.0, 2.0),
    },
  )
  cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.25, 0.25),
  }

  cfg.curriculum = {
    "joint_randomization": CurriculumTermCfg(
      func=tracking_mdp.joint_randomization_stages,
      params={
        "stages": [
          {
            "step": 0,
            "friction": (0.6, 1.2),
            "motor_strength": (0.8, 1.0),
            "delay_lag": (0, 2),
            "payload_kg": (0.0, 2.0),
            "push_xy_mps": (-0.25, 0.25),
            "push_yaw_radps": (-0.25, 0.25),
          },
          {
            "step": 6000 * 24,
            "friction": (0.45, 1.2),
            "motor_strength": (0.7, 1.0),
            "delay_lag": (0, 4),
            "payload_kg": (0.0, 4.0),
            "push_xy_mps": (-0.35, 0.35),
            "push_yaw_radps": (-0.4, 0.4),
          },
          {
            "step": 7500 * 24,
            "friction": (0.3, 1.2),
            "motor_strength": (0.6, 1.0),
            "delay_lag": (0, 7),
            "payload_kg": (0.0, 6.0),
            "push_xy_mps": (-0.5, 0.5),
            "push_yaw_radps": (-0.6, 0.6),
          },
          {
            "step": 9000 * 24,
            "friction": (0.2, 1.2),
            "motor_strength": (0.55, 1.0),
            "delay_lag": (0, 8),
            "payload_kg": (0.0, 7.0),
            "push_xy_mps": (-0.5, 0.5),
            "push_yaw_radps": (-0.78, 0.78),
          },
        ]
      },
    )
  }
  return cfg


def unitree_g1_robust_history_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Robust G1 tracking with a short proprioceptive/action history.

  Reference commands remain current-frame observations. Only signals that can
  reveal the realized dynamics are stacked, so the actor can infer actuator
  lag, effective motor strength, and slip from recent state-action response.
  Four samples at the 20 ms control period cover t, t-20, t-40, and t-60 ms.
  """
  cfg = unitree_g1_robust_tracking_env_cfg(play=play)
  for term_name in (
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  ):
    cfg.observations["actor"].terms[term_name].history_length = 4
  return cfg
