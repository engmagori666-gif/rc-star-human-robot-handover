"""Expert policy for RobotHumanHandoverCart environment.

This expert is phase-aware — it reads the human animation phase
and times the delivery to coincide with the REACH_OUT phase
when the human hand is extended and ready to receive the object.

Phases:
    APPROACH: Human walking to table — robot picks up object and waits
    REACH_OUT: Human extending arm — robot delivers object to human hand
    RETREAT: Human walking back — episode should be done
    COMPLETE: Success
"""
from typing import Any, Dict, Optional
from dataclasses import dataclass
import numpy as np
from gym.spaces import Box
from human_robot_gym.demonstrations.experts.pick_place_human_cart_expert import (
    PickPlaceHumanCartExpert,
    PickPlaceHumanCartExpertObservation,
)


class RobotHumanHandoverCartExpert(PickPlaceHumanCartExpert):
    """Phase-aware expert for robot-to-human handover.

    Behavior:
        1. Pick up object (same as PickPlace)
        2. If human in APPROACH phase: move to waiting position above goal
        3. If human in REACH_OUT phase: move to human hand and release
        4. If delivered: done
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        signal_to_noise_ratio: float = 1.0,
        hover_dist: float = 0.15,
        tan_theta: float = 0.5,
        horizontal_epsilon: float = 0.06,
        vertical_epsilon: float = 0.06,
        goal_dist: float = 0.1,
        gripper_fully_opened_threshold: float = 0.02,
        release_when_delivered: bool = True,
        delta_time: float = 0.01,
        seed: Optional[int] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            signal_to_noise_ratio=signal_to_noise_ratio,
            hover_dist=hover_dist,
            tan_theta=tan_theta,
            horizontal_epsilon=horizontal_epsilon,
            vertical_epsilon=vertical_epsilon,
            goal_dist=goal_dist,
            gripper_fully_opened_threshold=gripper_fully_opened_threshold,
            release_when_delivered=release_when_delivered,
            delta_time=delta_time,
            seed=seed,
        )
        self._task_phase = None
        self._environment = None

    def set_task_phase(self, task_phase):
        """Called by environment to update current phase."""
        self._task_phase = task_phase

    def set_environment(self, environment):
        """Bind the expert to the wrapped environment for live phase updates."""
        self._environment = environment

    def __call__(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        if self._environment is not None:
            self.set_task_phase(self._environment.unwrapped.task_phase)
        return super().__call__(obs_dict=obs_dict)

    def _select_motion(self, obs: PickPlaceHumanCartExpertObservation) -> np.ndarray:
        """Phase-aware motion selection.

        - Not gripped: pick up object (same as parent)
        - Gripped + APPROACH phase: move to hover above target and wait
        - Gripped + REACH_OUT phase: move directly to target (human hand)
        - Delivered: move away
        """
        from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (
            RobotHumanHandoverPhase
        )

        if self._object_delivered(obs) and self._gripper_fully_opened(obs):
            return self._move_to_above_object(obs=obs)

        if not obs.object_gripped:
            # Pick up phase — same as parent
            if self._above_object(obs) and self._gripper_fully_opened(obs):
                return self._move_to_object(obs)
            else:
                return self._move_to_above_object(obs)

        # Object is gripped — now decide based on human phase
        if self._task_phase == RobotHumanHandoverPhase.APPROACH:
            # The human hand is still moving toward the workspace. Chasing that
            # target creates avoidable human contacts, so hold the grasped
            # object until REACH_OUT begins.
            return np.zeros(3)

        elif self._task_phase == RobotHumanHandoverPhase.REACH_OUT:
            # Human extending arm — move directly to human hand
            if self._above_target(obs):
                return self._move_to_target(obs)
            else:
                return self._move_to_above_target(obs)

        else:
            # RETREAT or COMPLETE — move to target if not delivered
            if self._above_target(obs):
                return self._move_to_target(obs)
            else:
                return self._move_to_above_target(obs)

    def _select_gripper_action(self, obs: PickPlaceHumanCartExpertObservation) -> np.ndarray:
        """Release only when in REACH_OUT phase and at target."""
        from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (
            RobotHumanHandoverPhase
        )

        if self._object_delivered(obs):
            return self._open_gripper()

        if obs.object_gripped:
            # Only release when human is ready AND we are at target
            if (self._task_phase == RobotHumanHandoverPhase.REACH_OUT and
                    self._at_target(obs)):
                return self._open_gripper()
            return self._close_gripper()

        if self._at_object(obs):
            return self._close_gripper()

        return self._open_gripper()

    def _at_target(self, obs: PickPlaceHumanCartExpertObservation) -> bool:
        """Check if end-effector is close enough to target."""
        return (
            np.linalg.norm(obs.vec_eef_to_target[:2]) < self._horizontal_epsilon and
            abs(obs.vec_eef_to_target[2]) < self._vertical_epsilon
        )
