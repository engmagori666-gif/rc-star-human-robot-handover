"""Transparent robot-to-human handover environment.

Extends ReactiveHumanHandoverCart by adding two transparency mechanisms:

    1. Legibility reward: the robot earns a bonus for moving its end-effector
       directly toward the current goal (cosine similarity between motion direction
       and goal direction), weighted by ``legibility_weight``.

    2. Intent-enhanced reactive human: the human's reach-out animation speed is
       driven by a blended signal combining distance-based progress and the
       robot's directional commitment toward the target::

           combined = (1 - intent_weight) * robot_progress
                    + intent_weight * max(0, intent_signal)
           human_speed = speed_min + speed_scale * combined

       This creates a genuine bidirectional loop:
           robot moves toward human  →  intent_signal high  →  human extends faster
           robot hesitates/moves away →  intent_signal low  →  human slows/waits

Author:
    Emanuel (Kunsan National University)
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np

from human_robot_gym.environments.manipulation.reactive_handover_env import (
    ReactiveHumanHandoverCart,
)
from robosuite.utils.placement_samplers import ObjectPositionSampler


class TransparentHandoverCart(ReactiveHumanHandoverCart):
    """RobotHumanHandoverCart with reactive human, legibility reward, and intent signalling.

    Transparency is implemented at two levels:

    **Legibility reward** — added to the base reward each step:

        legibility_reward = legibility_weight * cos(Δeef, eef→target)

    where ``Δeef`` is the EEF displacement since the last step and
    ``eef→target`` is the direction from the EEF to the human's hand.

    **Intent-enhanced human reactivity** — overrides ``_get_robot_progress``
    so ``_compute_animation_time`` (inherited from ``ReactiveHumanHandoverCart``)
    drives the human with the blended signal described above.

    Args:
        legibility_weight (float): Reward weight for the legibility bonus.
            Set to 0.0 to disable (reduces to ReactiveHumanHandoverCart).
            Defaults to 0.1.
        intent_weight (float): Blend weight [0, 1] for directional intent vs
            distance in the human speed formula.
            0.0 → pure distance (identical to ReactiveHumanHandoverCart).
            1.0 → pure directional intent.
            Defaults to 0.5.
        All other args: identical to ReactiveHumanHandoverCart.
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
        reward_shaping: bool = False,
        goal_dist: float = 0.1,
        n_object_placements_sampled_per_100_steps: int = 3,
        collision_reward: float = -10,
        task_reward: float = 1,
        object_in_human_hand_reward: float = -1,
        object_gripped_reward: float = -1,
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
        human_animation_freq: float = 190,
        human_rand: List[float] = [0.0, 0.0, 0.0],
        n_animations_sampled_per_100_steps: int = 2,
        safe_vel: float = 0.001,
        self_collision_safety: float = 0.01,
        collision_debounce_delay: float = 0.01,
        seed: int = 0,
        verbose: bool = False,
        done_at_collision: bool = False,
        done_at_success: bool = False,
        max_transport_dist: float = 0.5,
        speed_min: float = 0.05,
        speed_scale: float = 0.010,
        legibility_weight: float = 0.1,
        intent_weight: float = 0.5,
    ):
        # Must be set before super().__init__() because it triggers reset
        self._legibility_weight = float(legibility_weight)
        self._intent_weight = float(np.clip(intent_weight, 0.0, 1.0))
        self._prev_eef_pos = None
        self._cached_intent = 0.0

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
            max_transport_dist=max_transport_dist,
            speed_min=speed_min,
            speed_scale=speed_scale,
        )

    # ------------------------------------------------------------------
    # Intent signal helpers
    # ------------------------------------------------------------------

    def _get_eef_pos(self) -> np.ndarray:
        """Return the current end-effector position from the MuJoCo sim."""
        try:
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
        except Exception:
            return np.zeros(3)

    def _compute_intent_signal(self) -> float:
        """Compute the robot's directional commitment toward the target.

        Measures the cosine similarity between:
          - Δeef: EEF displacement since the last control step
          - eef → target: direction from EEF to the human's hand

        Returns:
            float in [-1, 1].  1 = moving straight toward target.
                               0 = stationary or perpendicular.
                              -1 = moving directly away.
        """
        if self._prev_eef_pos is None:
            return 0.0

        current_eef = self._get_eef_pos()
        delta = current_eef - self._prev_eef_pos
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-5:
            return 0.0

        try:
            goal_dir = self.target_pos - current_eef
        except Exception:
            return 0.0

        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm < 1e-5:
            return 1.0  # Already at target

        return float(np.dot(delta / delta_norm, goal_dir / goal_norm))

    # ------------------------------------------------------------------
    # Intent-enhanced human reactivity
    # ------------------------------------------------------------------

    def _get_robot_progress(self) -> float:
        """Override to blend distance-based progress with directional intent.

        The combined signal drives the reactive human's animation speed::

            combined = (1 - intent_weight) * distance_progress
                     + intent_weight * max(0, intent_signal)

        ``intent_signal`` is clamped to [0, 1] so that moving away from the
        target does not further slow the human below the distance-only baseline.
        """
        distance_progress = super()._get_robot_progress()
        # Only positive intent speeds up the human; negative intent clips to 0
        intent_signal = max(0.0, self._cached_intent)
        return (
            (1.0 - self._intent_weight) * distance_progress
            + self._intent_weight * intent_signal
        )

    # ------------------------------------------------------------------
    # Legibility reward
    # ------------------------------------------------------------------

    def reward(self, achieved_goal, desired_goal, info) -> float:
        """Add the legibility bonus on top of the base handover reward.

        The legibility bonus rewards the robot for goal-directed motion::

            legibility_reward = legibility_weight * intent_signal

        where ``intent_signal`` is in [-1, 1] (negative = penalised for
        moving away from the target).

        Args:
            achieved_goal: Achieved goal from the current observation.
            desired_goal: Desired goal for the current episode.
            info: Info dict from the current step.

        Returns:
            float: Base reward + legibility bonus.
        """
        base_reward = super().reward(achieved_goal, desired_goal, info)
        legibility_bonus = self._legibility_weight * self._cached_intent
        return base_reward + legibility_bonus

    # ------------------------------------------------------------------
    # Step / reset overrides for position tracking
    # ------------------------------------------------------------------

    def step(self, action):
        """Compute intent signal before physics advances, then run base step.

        Order of operations:
            1. Compute intent from EEF motion since last step (uses _prev_eef_pos).
            2. Cache it in _cached_intent so both reward() and _get_robot_progress()
               (called inside super().step()) see the same value.
            3. Run super().step() — this calls reward() and _compute_animation_time().
            4. Update _prev_eef_pos for the next step.
            5. Log legibility to info for TensorBoard.

        Args:
            action: Robot action array.

        Returns:
            Tuple of (obs, reward, done, info).
        """
        # Compute intent from motion since the last step
        self._cached_intent = self._compute_intent_signal()

        obs, rew, done, info = super().step(action)

        # Update history after physics has advanced
        self._prev_eef_pos = self._get_eef_pos()
        info["legibility"] = self._cached_intent

        return obs, rew, done, info

    def reset(self):
        """Reset environment and initialise EEF position tracking.

        Returns:
            np.ndarray: Initial observation.
        """
        obs = super().reset()
        self._prev_eef_pos = self._get_eef_pos()
        self._cached_intent = 0.0
        return obs
