"""Reactive robot-to-human handover environment — V2 Clean Redesign.

Extends RobotHumanHandoverCart by replacing fixed sinusoidal looping in the
REACH_OUT phase with a grip-gated reactive human model:

    robot_progress = grip_flag × clip(1 - dist(object, human_hand) / max_dist, 0, 1)
    human_speed    = speed_min + speed_scale × robot_progress

Key V2 design choices vs prior attempts (v1–v6):
  • Human arm frozen until robot grips the object (grip-gated progress)
  • Dramatic reactive range: speed_min=0.001 (frozen), speed_scale=2.0 (fast)
    → 2000× speed difference vs V1's 1.2× (0.05 → 0.060)
  • Dense reward built into environment (not a wrapper), phase-split:
      approach:  0.5 × exp(-3 × ‖eef − obj‖)
      transport: 1.0 × exp(-2 × ‖obj − target‖)
      (ρ drives human arm speed only — NOT included in reward to preserve single-variable comparison)
  • Sparse milestones: 0 (base) / 0 (gripped) / 10 (RETREAT/COMPLETE) — no negatives
  • Success defined as task_phase ∈ {RETREAT, COMPLETE} (human has object)
  • reward_shaping=True — dense reward active every step
  • No SIR required — pure RL with dense guidance

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


class ReactiveHumanHandoverCart(RobotHumanHandoverCart):
    """RobotHumanHandoverCart with a V2 grip-gated reactive human model.

    The REACH_OUT animation speed is gated on whether the robot is grasping the
    manipulation object and proportional to how close the object is to the
    human's hand:

        robot_progress = grip_flag × clip(1 − dist(obj, target) / max_dist, 0, 1)
        human_speed    = speed_min + speed_scale × robot_progress

    Human arm is effectively frozen (speed_min = 0.001 frame/step) until the
    robot grips the object.  Once gripped and delivered (robot_progress ≈ 1),
    the arm extends quickly (up to speed_min + speed_scale = 2.001 frame/step).

    Success is redefined as task_phase ∈ {RETREAT, COMPLETE} — the episode ends
    as soon as the human grasps the object, rather than waiting for the full
    retreat animation to finish.

    Reward shaping (reward_shaping=True, recommended):
        phase 1: 0.5 × exp(-3 × ‖eef − obj‖)
        phase 2: 1.0 × exp(-2 × ‖obj − target‖)
        Identical formula to ScriptedHandoverCartV2 — robot_progress (ρ) is NOT
        in the reward so that the A/B comparison isolates the human model only.

    Args:
        max_transport_dist (float): Distance at which robot_progress = 0.
            Defaults to 0.5 m.
        speed_min (float): Human arm speed when robot has not gripped / is far.
            Defaults to 0.001 frame/step — effectively frozen.
        speed_scale (float): Additional speed multiplied by robot_progress.
            Defaults to 2.0 — arm completes extension in ~45 steps at full progress.
        All other args: identical to RobotHumanHandoverCart.
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
        max_transport_dist: float = 0.5,
        speed_min: float = 0.001,
        speed_scale: float = 2.0,
    ):
        self._reach_out_progress = 0.0
        self._max_transport_dist = max_transport_dist
        self._speed_min = speed_min
        self._speed_scale = speed_scale

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
    # Grip check — also guards the RETREAT trigger in step()
    # ------------------------------------------------------------------

    def _is_robot_gripping_object(self) -> bool:
        """Return True if the robot gripper is in contact with the manipulation object."""
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
        contact between the palm geom (gap=0.2 m) and the hammer, with no grip
        requirement.  In the reactive model the arm barely moves without grip so
        spurious contacts are very unlikely, but the check is explicit here for
        symmetry with ScriptedHandoverCartV2 and for correctness.

        If the parent transitions to RETREAT without the robot gripping, the
        weld is released and the phase is reverted to REACH_OUT.
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
    # Reactive human model
    # ------------------------------------------------------------------

    def _get_robot_progress(self) -> float:
        """Transport progress in [0, 1].

        Returns 0 when the robot has not gripped the object or when the
        object is >= max_transport_dist away from the human hand.
        Returns 1 when the object is at the target (human hand).
        """
        if not self._is_robot_gripping_object():
            return 0.0
        try:
            obj_pos = self.sim.data.body_xpos[self.manipulation_object_body_id]
            dist = np.linalg.norm(np.array(obj_pos) - self.target_pos)
            return float(np.clip(1.0 - dist / self._max_transport_dist, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_animation_time(self, control_time: int) -> int:
        """Override: REACH_OUT advances proportionally to grip-gated robot progress.

        In REACH_OUT the animation advances at:
            human_speed = speed_min + speed_scale × robot_progress
        where robot_progress = 0 when the robot has not gripped the object.

        This effectively freezes the human arm until the robot picks up the object,
        then extends it quickly as the robot approaches.

        All other phases (APPROACH, RETREAT, COMPLETE) are identical to the
        baseline RobotHumanHandoverCart.

        Args:
            control_time: Current low-level controller time.

        Returns:
            Current animation frame index.
        """
        # Linear base time from HumanEnv (bypasses baseline's sinusoidal loop)
        animation_time = super(RobotHumanHandoverCart, self)._compute_animation_time(control_time)
        classic_animation_time = animation_time

        keyframes = self.human_animation_data[self.human_animation_id][1]["keyframes"]

        # APPROACH → REACH_OUT: natural animation trigger (unchanged from baseline)
        if animation_time > keyframes[0] and self.task_phase == RobotHumanHandoverPhase.APPROACH:
            self.task_phase = RobotHumanHandoverPhase.REACH_OUT
            # Initialise from animation position so there is no arm-position jump
            self._reach_out_progress = float(
                min(animation_time - keyframes[0], keyframes[1] - keyframes[0])
            )

        # Reactive REACH_OUT: advance arm at grip-gated progress-proportional speed
        if self.task_phase == RobotHumanHandoverPhase.REACH_OUT:
            robot_progress = self._get_robot_progress()
            human_speed = self._speed_min + self._speed_scale * robot_progress
            self._reach_out_progress = min(
                self._reach_out_progress + human_speed,
                float(keyframes[1] - keyframes[0]),
            )
            animation_time = int(keyframes[0] + self._reach_out_progress)
            self._n_delayed_timesteps = classic_animation_time - animation_time

        # RETREAT: run animation linearly from where REACH_OUT left off
        if self.task_phase == RobotHumanHandoverPhase.RETREAT:
            animation_time -= self._n_delayed_timesteps

        # COMPLETE: freeze at last frame
        if animation_time >= self.human_animation_length - 1:
            self.task_phase = RobotHumanHandoverPhase.COMPLETE
            animation_time = self.human_animation_length - 1

        return animation_time

    def _reset_animation(self):
        """Reset reactive tracking in addition to base reset."""
        super()._reset_animation()
        self._reach_out_progress = 0.0

    # ------------------------------------------------------------------
    # Success condition — human has taken the object
    # ------------------------------------------------------------------

    def _check_success(self, achieved_goal: List[float], desired_goal: List[float]) -> bool:
        """Success when the human has grasped the object (RETREAT or COMPLETE phase).

        Changing success from COMPLETE to RETREAT removes the dependency on the
        full retreat animation finishing within the episode horizon.  The handover
        is physically complete the moment the human grips the object.
        """
        return self.task_phase in (
            RobotHumanHandoverPhase.RETREAT,
            RobotHumanHandoverPhase.COMPLETE,
        )

    # ------------------------------------------------------------------
    # Reward — redesigned for dense learning signal
    # ------------------------------------------------------------------

    def _sparse_reward(
        self,
        achieved_goal: List[float],
        desired_goal: List[float],
        info: Dict[str, Any],
    ) -> float:
        """Phase-gated sparse milestones — all non-negative.

        Milestone schedule:
            RETREAT / COMPLETE  →  task_reward   (default 10.0)
            object gripped      →  0.0            (no penalty for gripping)
            otherwise           →  0.0            (no per-step penalty)

        The absence of per-step penalties lets SAC explore freely without
        collapsing to the minimum-action policy under constant negative pressure.
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
        """Phase-split dense guidance reward — identical formula to ScriptedHandoverCartV2.

        Phase 1 — approach & grasp (object not yet gripped):
            r = 0.5 × exp(−3 × ‖eef − obj‖)

        Phase 2 — transport (object gripped, robot moving to human hand):
            r = 1.0 × exp(−2 × ‖obj − target‖)

        Note: robot_progress (ρ) is intentionally NOT included here even though
        it exists and drives the human arm speed in _compute_animation_time.
        Including 0.5×ρ would add up to 0.5 extra reward per step exclusively for
        the reactive condition, confounding the human-model comparison.
        The reactive model's advantage comes only from human arm dynamics.
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
