"""Scripted-human handover environment — V2 reward and success redesign.

Identical to RobotHumanHandoverCart except:
  - V2 reward: dense guidance (approach + transport) replacing the -1/step baseline
  - V2 success: RETREAT or COMPLETE (human has gripped object)
  - No SIR, no RSI required — same pure-RL setup as ReactiveHumanHandoverCart

Used as the SCRIPTED ablation partner for ReactiveHumanHandoverCart.  The only
difference between the two conditions is the human model:

  ScriptedHandoverCartV2  — sinusoidal looping arm (unchanged baseline behaviour)
  ReactiveHumanHandoverCart — grip-gated reactive arm

Author:
    Emanuel (Kunsan National University) — V2 redesign 2025-06-17
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np

from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (
    RobotHumanHandoverCart,
    RobotHumanHandoverPhase,
)
from robosuite.utils.placement_samplers import ObjectPositionSampler


class ScriptedHandoverCartV2(RobotHumanHandoverCart):
    """RobotHumanHandoverCart with the V2 reward and success redesign.

    The human model is unchanged from the baseline (sinusoidal looping in
    REACH_OUT phase).  Only the reward and success condition are updated to
    match ReactiveHumanHandoverCart, creating a clean controlled ablation:

        Variable       ScriptedV2          ReactiveV2
        -----------    -----------------   -----------------------
        Human model    scripted loop       grip-gated reactive
        Reward         V2 dense            V2 dense  (identical)
        Success        RETREAT|COMPLETE    RETREAT|COMPLETE (identical)
        Shield         PFL                 PFL       (identical)
        Network        [256, 256]          [256, 256] (identical)
        No SIR/RSI     yes                 yes        (identical)

    The comparison isolates the effect of human reactivity on policy learning.

    Args:
        All args identical to RobotHumanHandoverCart.  Extra reward parameters:
        task_reward (float): Sparse terminal reward on RETREAT / COMPLETE. Default 10.0.
        object_gripped_reward (float): Reward when gripped (milestone). Default 0.0.
        object_in_human_hand_reward (float): Same as task_reward for RETREAT. Default 10.0.
    """

    def __init__(
        self,
        robots: Union[str, List[str]],
        robot_base_offset=None,
        env_configuration: str = "default",
        controller_configs=None,
        gripper_types: Union[str, List[str]] = "default",
        initialization_noise="default",
        table_full_size=(1.5, 2.0, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        object_full_size=(0.04, 0.04, 0.04),
        use_camera_obs: bool = True,
        use_object_obs: bool = True,
        reward_scale: Optional[float] = 1.0,
        reward_shaping: bool = True,
        goal_dist: float = 0.06,
        n_object_placements_sampled_per_100_steps: int = 3,
        collision_reward: float = -1.0,
        task_reward: float = 10.0,
        object_in_human_hand_reward: float = 10.0,
        object_gripped_reward: float = 0.0,
        object_placement_initializer: Optional[ObjectPositionSampler] = None,
        obstacle_placement_initializer: Optional[ObjectPositionSampler] = None,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        render_camera: str = "frontview",
        render_collision_mesh: bool = False,
        render_visual_mesh: bool = True,
        render_gpu_device_id: int = -1,
        control_freq: float = 10,
        horizon: int = 1000,
        ignore_done: bool = False,
        hard_reset: bool = True,
        camera_names="frontview",
        camera_heights: Union[int, List[int]] = 256,
        camera_widths: Union[int, List[int]] = 256,
        camera_depths: Union[bool, List[bool]] = False,
        camera_segmentations=None,
        renderer: str = "mujoco",
        renderer_config: Dict[str, Any] = None,
        shield_type: str = "PFL",
        visualize_failsafe_controller: bool = False,
        visualize_pinocchio: bool = False,
        control_sample_time: float = 0.004,
        human_animation_names: List[str] = [
            "RobotHumanHandover/0",
            "RobotHumanHandover/1",
            "RobotHumanHandover/2",
            "RobotHumanHandover/3",
            "RobotHumanHandover/4",
            "RobotHumanHandover/5",
            "RobotHumanHandover/6",
            "RobotHumanHandover/7",
            "RobotHumanHandover/8",
        ],
        base_human_pos_offset: List[float] = [0.0, 0.0, 0.0],
        human_animation_freq: float = 90,
        human_rand: List[float] = [0.0, 0.0, 0.0],
        n_animations_sampled_per_100_steps: int = 2,
        safe_vel: float = 0.001,
        self_collision_safety: float = 0.01,
        collision_debounce_delay: float = 0.01,
        seed: int = 0,
        verbose: bool = False,
        done_at_collision: bool = False,
        done_at_success: bool = True,
    ):
        super().__init__(
            robots=robots,
            robot_base_offset=robot_base_offset,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            table_full_size=table_full_size,
            table_friction=table_friction,
            object_full_size=object_full_size,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            goal_dist=goal_dist,
            n_object_placements_sampled_per_100_steps=n_object_placements_sampled_per_100_steps,
            collision_reward=collision_reward,
            task_reward=task_reward,
            object_in_human_hand_reward=object_in_human_hand_reward,
            object_gripped_reward=object_gripped_reward,
            object_placement_initializer=object_placement_initializer,
            obstacle_placement_initializer=obstacle_placement_initializer,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            shield_type=shield_type,
            visualize_failsafe_controller=visualize_failsafe_controller,
            visualize_pinocchio=visualize_pinocchio,
            control_sample_time=control_sample_time,
            human_animation_names=human_animation_names,
            base_human_pos_offset=base_human_pos_offset,
            human_animation_freq=human_animation_freq,
            human_rand=human_rand,
            n_animations_sampled_per_100_steps=n_animations_sampled_per_100_steps,
            safe_vel=safe_vel,
            self_collision_safety=self_collision_safety,
            collision_debounce_delay=collision_debounce_delay,
            seed=seed,
            verbose=verbose,
            done_at_collision=done_at_collision,
            done_at_success=done_at_success,
        )

    # ------------------------------------------------------------------
    # Grip guard — prevents spurious RETREAT from oscillating arm contact
    # ------------------------------------------------------------------

    def _is_robot_gripping_object(self) -> bool:
        """Return True if the robot gripper is grasping the manipulation object."""
        try:
            return self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.manipulation_object,
            )
        except Exception:
            return False

    def step(self, action: np.ndarray):
        """Override to require robot grip before accepting a RETREAT transition.

        The parent RobotHumanHandoverCart.step() triggers RETREAT on any MuJoCo
        contact between the palm geom (inactive, gap=0.2 m) and the hammer, with
        no grip requirement.  The scripted arm oscillates throughout REACH_OUT;
        if it swings within 0.20 m of the hammer sitting on the table, the parent
        would call _human_pickup_object() and set RETREAT — a false success.

        This override detects that case and reverts the transition.
        """
        phase_before = self.task_phase
        obs, rew, done, info = super().step(action)

        if (
            phase_before == RobotHumanHandoverPhase.REACH_OUT
            and self.task_phase == RobotHumanHandoverPhase.RETREAT
            and not self._is_robot_gripping_object()
        ):
            self._human_drop_object()
            self.task_phase = RobotHumanHandoverPhase.REACH_OUT

        return obs, rew, done, info

    # ------------------------------------------------------------------
    # Success condition — matches ReactiveHumanHandoverCart
    # ------------------------------------------------------------------

    def _check_success(self, achieved_goal: List[float], desired_goal: List[float]) -> bool:
        """Success when human has grasped (RETREAT) or animation complete (COMPLETE).

        Baseline requires COMPLETE (full retreat animation must finish).
        V2 changes this to RETREAT so the episode ends as soon as the human
        grips the object — identical to ReactiveHumanHandoverCart._check_success.
        """
        return self.task_phase in (
            RobotHumanHandoverPhase.RETREAT,
            RobotHumanHandoverPhase.COMPLETE,
        )

    # ------------------------------------------------------------------
    # Reward — matches ReactiveHumanHandoverCart
    # ------------------------------------------------------------------

    def _sparse_reward(
        self,
        achieved_goal: List[float],
        desired_goal: List[float],
        info: Dict[str, Any],
    ) -> float:
        """V2 sparse milestones — all non-negative.

        RETREAT / COMPLETE  →  task_reward (10.0)
        object gripped      →  0.0  (no per-step penalty for gripping)
        otherwise           →  0.0  (no per-step penalty)

        Identical to ReactiveHumanHandoverCart._sparse_reward.
        """
        object_gripped = bool(achieved_goal[6])

        if self._check_success(achieved_goal=achieved_goal, desired_goal=desired_goal):
            return self.task_reward
        elif object_gripped:
            return self.object_gripped_reward  # 0.0
        else:
            return 0.0

    def _dense_reward(
        self,
        achieved_goal: List[float],
        desired_goal: List[float],
        info: Dict[str, Any],
    ) -> float:
        """V2 phase-split dense guidance.

        Phase 1 — approach & grasp:
            r = 0.5 × exp(−3 × ‖eef − obj‖)

        Phase 2 — transport (object gripped):
            r = 1.0 × exp(−2 × ‖obj − target‖)

        Identical to ReactiveHumanHandoverCart._dense_reward except the
        robot_progress term (0.5×ρ) is omitted because the scripted human arm
        does not respond to where the object is — progress cannot be measured
        against a moving target.

        Identical to ReactiveHumanHandoverCart._dense_reward.
        """
        eef_pos = np.array(achieved_goal[:3])
        obj_pos = np.array(achieved_goal[3:6])
        object_gripped = bool(achieved_goal[6])

        if not object_gripped:
            eef_2_obj = np.linalg.norm(obj_pos - eef_pos)
            return 0.5 * float(np.exp(-3.0 * eef_2_obj))
        else:
            obj_2_target = np.linalg.norm(np.array(desired_goal) - obj_pos)
            return 1.0 * float(np.exp(-2.0 * obj_2_target))
