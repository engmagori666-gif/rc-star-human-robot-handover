"""Finite-state recovery for grasp and handover transport deadlocks."""
from collections import deque
from typing import Any, Dict, Tuple

import gym
import numpy as np


class PregraspRecoveryWrapper(gym.Wrapper):
    """Recover from stalled grasp acquisition or handover transport.

    The wrapper operates in Cartesian action space outside the predictive
    governor. Recovery actions therefore still pass through both the human
    governor and the lower-level safety shield.
    """

    IDLE = 0
    LIFT = 1
    DESCEND = 2
    CLOSE = 3
    TRANSPORT_LIFT = 4
    TRANSPORT_MOVE = 5

    def __init__(
        self,
        env: gym.Env,
        trigger_distance: float = 0.03,
        trigger_steps: int = 25,
        progress_window: int = 20,
        minimum_progress: float = 0.002,
        hover_height: float = 0.08,
        waypoint_tolerance: float = 0.02,
        grasp_tolerance: float = 0.012,
        open_steps: int = 10,
        close_steps: int = 20,
        stable_grip_steps: int = 5,
        max_attempts: int = 2,
        transport_trigger_steps: int = 30,
        transport_progress_window: int = 20,
        transport_minimum_progress: float = 0.005,
        transport_hover_height: float = 0.08,
        transport_horizontal_tolerance: float = 0.04,
        transport_tan_theta: float = 0.5,
        transport_max_steps: int = 300,
        transport_lift_steps: int = 20,
        transport_lift_delta: float = 0.03,
        transport_resume_policy_after_lift: bool = False,
        transport_max_activations: int = 1,
    ):
        super().__init__(env)
        self._trigger_distance = float(trigger_distance)
        self._trigger_steps = max(int(trigger_steps), 1)
        self._progress_window = max(int(progress_window), 2)
        self._minimum_progress = float(minimum_progress)
        self._hover_height = float(hover_height)
        self._waypoint_tolerance = float(waypoint_tolerance)
        self._grasp_tolerance = float(grasp_tolerance)
        self._open_steps = max(int(open_steps), 1)
        self._close_steps = max(int(close_steps), 1)
        self._stable_grip_steps_required = max(
            int(stable_grip_steps),
            1,
        )
        self._max_attempts = max(int(max_attempts), 0)
        self._transport_trigger_steps = max(
            int(transport_trigger_steps),
            1,
        )
        self._transport_progress_window = max(
            int(transport_progress_window),
            2,
        )
        self._transport_minimum_progress = float(
            transport_minimum_progress
        )
        self._transport_hover_height = float(transport_hover_height)
        self._transport_horizontal_tolerance = float(
            transport_horizontal_tolerance
        )
        self._transport_tan_theta = float(transport_tan_theta)
        self._transport_max_steps = max(int(transport_max_steps), 1)
        self._transport_lift_steps = max(int(transport_lift_steps), 1)
        self._transport_lift_delta = float(transport_lift_delta)
        self._transport_resume_policy_after_lift = bool(
            transport_resume_policy_after_lift
        )
        self._transport_max_activations = max(
            int(transport_max_activations),
            1,
        )
        self._distance_history = deque(maxlen=self._progress_window)
        self._target_distance_history = deque(
            maxlen=self._transport_progress_window
        )
        self._near_object_steps = 0
        self._transport_stall_steps = 0
        self._state = self.IDLE
        self._state_steps = 0
        self._attempts = 0
        self._steps = 0
        self._successful_recoveries = 0
        self._stable_grip_steps = 0
        self._transport_activations = 0
        self._transport_successes = 0
        self._last_transport_target = np.zeros(3, dtype=float)
        self._last_transport_action = np.zeros(3, dtype=float)

    def reset(self):
        obs = self.env.reset()
        self._distance_history.clear()
        self._target_distance_history.clear()
        self._near_object_steps = 0
        self._transport_stall_steps = 0
        self._state = self.IDLE
        self._state_steps = 0
        self._attempts = 0
        self._steps = 0
        self._successful_recoveries = 0
        self._stable_grip_steps = 0
        self._transport_activations = 0
        self._transport_successes = 0
        self._last_transport_target = np.zeros(3, dtype=float)
        self._last_transport_action = np.zeros(3, dtype=float)
        return obs

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        base = self._base_env()
        raw_object_gripped = self._object_gripped()
        if raw_object_gripped:
            self._stable_grip_steps += 1
        else:
            self._stable_grip_steps = 0
        object_gripped = (
            self._stable_grip_steps
            >= self._stable_grip_steps_required
        )
        vector = self._eef_to_object()
        distance = float(np.linalg.norm(vector))
        self._update_trigger(
            distance=distance,
            object_gripped=object_gripped,
        )
        self._update_transport_trigger(
            object_gripped=object_gripped,
        )

        if (
            object_gripped
            and self._state in (self.LIFT, self.DESCEND, self.CLOSE)
        ):
            self._successful_recoveries += 1
            self._state = self.IDLE
            self._state_steps = 0
            self._distance_history.clear()

        governed = np.asarray(action, dtype=float).copy()
        if (
            self._is_transport_state()
            or (self._state != self.IDLE and not object_gripped)
        ):
            governed = self._recovery_action(
                action=governed,
                vector=vector,
            )
            self._steps += 1
            self._state_steps += 1

        obs, reward, done, info = self.env.step(governed)
        if (
            self._is_transport_state()
            and int(info.get("v1_object_captured", 0)) > 0
        ):
            self._transport_successes += 1
            self._state = self.IDLE
            self._state_steps = 0
            self._target_distance_history.clear()
        info["pregrasp_recovery_steps"] = self._steps
        info["pregrasp_recovery_attempts"] = self._attempts
        info["pregrasp_recovery_successes"] = (
            self._successful_recoveries
        )
        info["pregrasp_recovery_state"] = self._state
        info["pregrasp_recovery_near_steps"] = self._near_object_steps
        info["pregrasp_recovery_last_distance"] = distance
        info["pregrasp_recovery_object_gripped"] = int(object_gripped)
        info["pregrasp_recovery_raw_object_gripped"] = int(
            raw_object_gripped
        )
        info["pregrasp_recovery_stable_grip_steps"] = (
            self._stable_grip_steps
        )
        info["pregrasp_recovery_max_attempts"] = self._max_attempts
        info["handover_transport_recovery_activations"] = (
            self._transport_activations
        )
        info["handover_transport_recovery_successes"] = (
            self._transport_successes
        )
        info["handover_transport_recovery_stall_steps"] = (
            self._transport_stall_steps
        )
        info["handover_transport_recovery_state_steps"] = self._state_steps
        info["handover_transport_target_x"] = float(
            self._last_transport_target[0]
        )
        info["handover_transport_target_y"] = float(
            self._last_transport_target[1]
        )
        info["handover_transport_target_z"] = float(
            self._last_transport_target[2]
        )
        info["handover_transport_action_x"] = float(
            self._last_transport_action[0]
        )
        info["handover_transport_action_y"] = float(
            self._last_transport_action[1]
        )
        info["handover_transport_action_z"] = float(
            self._last_transport_action[2]
        )
        return obs, reward, done, info

    def _update_trigger(
        self,
        distance: float,
        object_gripped: bool,
    ) -> None:
        base = self._base_env()
        if (
            object_gripped
            or getattr(base, "_v1_object_captured", False)
            or self._state != self.IDLE
        ):
            return

        if distance <= self._trigger_distance:
            self._near_object_steps += 1
            self._distance_history.append(distance)
        else:
            self._near_object_steps = 0
            self._distance_history.clear()

        if (
            self._attempts >= self._max_attempts
            or self._near_object_steps < self._trigger_steps
            or len(self._distance_history) < self._progress_window
        ):
            return

        progress = (
            self._distance_history[0]
            - self._distance_history[-1]
        )
        if progress >= self._minimum_progress:
            return

        self._attempts += 1
        self._state = self.LIFT
        self._state_steps = 0
        self._near_object_steps = 0
        self._distance_history.clear()

    def _update_transport_trigger(self, object_gripped: bool) -> None:
        base = self._base_env()
        phase_name = getattr(
            getattr(base, "task_phase", None),
            "name",
            "",
        )
        if (
            self._state != self.IDLE
            or not object_gripped
            or getattr(base, "_v1_object_captured", False)
            or phase_name != "REACH_OUT"
            or self._transport_activations
            >= self._transport_max_activations
        ):
            if self._state == self.IDLE:
                self._transport_stall_steps = 0
                self._target_distance_history.clear()
            return

        target_distance = float(np.linalg.norm(self._eef_to_target()))
        self._transport_stall_steps += 1
        self._target_distance_history.append(target_distance)
        if (
            self._transport_stall_steps < self._transport_trigger_steps
            or len(self._target_distance_history)
            < self._transport_progress_window
        ):
            return

        progress = (
            self._target_distance_history[0]
            - self._target_distance_history[-1]
        )
        if progress >= self._transport_minimum_progress:
            return

        self._transport_activations += 1
        self._state = self.TRANSPORT_LIFT
        self._state_steps = 0
        self._transport_stall_steps = 0
        self._target_distance_history.clear()

    def _recovery_action(
        self,
        action: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        recovered = action.copy()
        if recovered.size < 4:
            return recovered

        if self._state == self.LIFT:
            target = vector + np.array([0.0, 0.0, self._hover_height])
            recovered[:3] = target
            recovered[-1] = -1.0
            if (
                self._state_steps >= self._open_steps
                and np.linalg.norm(target) <= self._waypoint_tolerance
            ):
                self._state = self.DESCEND
                self._state_steps = 0
        elif self._state == self.DESCEND:
            recovered[:3] = vector
            recovered[-1] = -1.0
            if np.linalg.norm(vector) <= self._grasp_tolerance:
                self._state = self.CLOSE
                self._state_steps = 0
        elif self._state == self.CLOSE:
            recovered[:3] = vector
            recovered[-1] = 1.0
            if self._state_steps >= self._close_steps:
                if self._attempts < self._max_attempts:
                    self._state = self.LIFT
                else:
                    self._state = self.IDLE
                self._state_steps = 0
        elif self._state == self.TRANSPORT_LIFT:
            recovered[:3] = np.array(
                [0.0, 0.0, self._transport_lift_delta],
                dtype=float,
            )
            recovered[-1] = 1.0
            self._last_transport_target = self._eef_to_target()
            self._last_transport_action = recovered[:3].copy()
            if self._state_steps >= self._transport_lift_steps:
                self._state = (
                    self.IDLE
                    if self._transport_resume_policy_after_lift
                    else self.TRANSPORT_MOVE
                )
                self._state_steps = 0
        elif self._state == self.TRANSPORT_MOVE:
            target = self._eef_to_target()
            self._last_transport_target = target.copy()
            if (
                np.linalg.norm(target[:2])
                > self._transport_horizontal_tolerance
            ):
                recovered[:3] = np.array(
                    [target[0], target[1], 0.0],
                    dtype=float,
                )
            else:
                recovered[:3] = target
            recovered[-1] = 1.0
            self._last_transport_action = np.clip(
                recovered[:3],
                self.action_space.low[:3],
                self.action_space.high[:3],
            )
            if self._state_steps >= self._transport_max_steps:
                self._state = self.IDLE
                self._state_steps = 0

        recovered[:3] = np.clip(
            recovered[:3],
            self.action_space.low[:3],
            self.action_space.high[:3],
        )
        return recovered

    def _is_transport_state(self) -> bool:
        return self._state in (
            self.TRANSPORT_LIFT,
            self.TRANSPORT_MOVE,
        )

    def _eef_to_object(self) -> np.ndarray:
        base = self._base_env()
        eef_pos = np.asarray(
            base.sim.data.site_xpos[base.robots[0].eef_site_id],
            dtype=float,
        )
        object_pos = np.asarray(
            base.sim.data.body_xpos[base.manipulation_object_body_id],
            dtype=float,
        )
        return object_pos - eef_pos

    def _eef_to_target(self) -> np.ndarray:
        base = self._base_env()
        eef_pos = np.asarray(
            base.sim.data.site_xpos[base.robots[0].eef_site_id],
            dtype=float,
        )
        return np.asarray(base.target_pos, dtype=float) - eef_pos

    def _above_target(self, vector: np.ndarray) -> bool:
        max_radius = (
            self._transport_horizontal_tolerance
            - vector[2] * self._transport_tan_theta
        )
        return (
            vector[2] < 0
            and np.linalg.norm(vector[:2]) < max_radius
        )

    def _object_gripped(self) -> bool:
        """Use the same cached two-finger signal exposed to the policy."""
        cache = getattr(self._base_env(), "_obs_cache", {})
        if "object_gripped" in cache:
            value = np.asarray(cache["object_gripped"]).reshape(-1)
            if value.size:
                return bool(value[0])
        return False

    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env
