"""Evaluation-only RC-STAR environment for the RPIC 2027 stress test."""
from typing import Dict, List

import numpy as np

from human_robot_gym.environments.manipulation.human_motion_disturbance import (
    HumanMotionDisturbanceSchedule,
    disturbed_reach_progress,
)
from human_robot_gym.environments.manipulation.ours_v1_handover_env import (
    ReactiveHumanHandoverCartV1,
)
from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (
    RobotHumanHandoverPhase,
)


class ReactiveHumanHandoverCartV1RPIC(ReactiveHumanHandoverCartV1):
    """Add deterministic test-time reach disturbances without changing the task."""

    def __init__(
        self,
        robots,
        rpic_condition: str = "normal",
        rpic_disturbance_seed: int = 2701,
        rpic_stable_grasp_steps: int = 2,
        rpic_trigger_min_steps: int = 2,
        rpic_trigger_max_steps: int = 5,
        rpic_delay_duration_steps: int = 8,
        rpic_reversal_duration_steps: int = 6,
        rpic_reversal_speed: float = 0.5,
        rpic_deadlock_window_steps: int = 50,
        rpic_deadlock_progress_epsilon: float = 0.001,
        **kwargs,
    ):
        duration = (
            rpic_reversal_duration_steps
            if rpic_condition == "reach_reversal"
            else rpic_delay_duration_steps
        )
        self._rpic_schedule = HumanMotionDisturbanceSchedule(
            condition=rpic_condition,
            base_seed=int(rpic_disturbance_seed),
            trigger_min_steps=int(rpic_trigger_min_steps),
            trigger_max_steps=int(rpic_trigger_max_steps),
            duration_steps=int(duration),
        )
        self._rpic_stable_grasp_required = max(int(rpic_stable_grasp_steps), 1)
        self._rpic_reversal_speed = max(float(rpic_reversal_speed), 0.0)
        self._rpic_deadlock_window = max(int(rpic_deadlock_window_steps), 1)
        self._rpic_deadlock_epsilon = max(
            float(rpic_deadlock_progress_epsilon), 0.0
        )
        self._rpic_episode_index = -1
        self._rpic_construction_complete = False
        self._reset_rpic_episode_state()
        super().__init__(robots=robots, **kwargs)
        self._rpic_construction_complete = True

    def _reset_rpic_episode_state(self):
        self._rpic_stable_grasp_steps = 0
        self._rpic_disturbed_policy_steps = 0
        self._rpic_disturbed_low_level_updates = 0
        self._rpic_previous_target_distance = None
        self._rpic_stall_steps = 0
        self._rpic_max_stall_steps = 0
        self._rpic_deadlock = False
        self._rpic_shield_measurement_calls = 0
        self._rpic_shield_measurement_calls_during_disturbance = 0
        self._rpic_shield_measurement_changes_during_disturbance = 0
        self._rpic_shield_measurement_stale_mismatches = 0
        self._rpic_shield_measurement_time_regressions = 0
        self._rpic_pfl_measurement_forward_calls = 0
        self._rpic_pfl_measurement_forward_calls_during_disturbance = 0
        self._rpic_last_shield_measurement = None
        self._rpic_last_shield_measurement_time = None

    def _reset_internal(self):
        super()._reset_internal()
        if self._rpic_construction_complete:
            self._rpic_episode_index += 1
        self._reset_rpic_episode_state()
        self._rpic_schedule.reset(max(self._rpic_episode_index, 0))

    def _disturbance_eligible(self) -> bool:
        return (
            self.task_phase == RobotHumanHandoverPhase.REACH_OUT
            and self._rpic_stable_grasp_steps
            >= self._rpic_stable_grasp_required
            and not self._v1_object_captured
        )

    def _compute_animation_time(self, control_time: int) -> int:
        previous_progress = self._get_bounded_reach_progress()
        animation_time = super()._compute_animation_time(control_time)
        if (
            not self._rpic_schedule.active
            or self.task_phase != RobotHumanHandoverPhase.REACH_OUT
            or self._v1_object_captured
        ):
            return animation_time

        keyframes = self.human_animation_data[self.human_animation_id][1][
            "keyframes"
        ]
        self._reach_out_progress = disturbed_reach_progress(
            condition=self._rpic_schedule.condition,
            previous_progress=previous_progress,
            nominal_progress=self._reach_out_progress,
            reversal_speed=self._rpic_reversal_speed,
        )
        max_progress = float(keyframes[1] - keyframes[0])
        self._reach_out_progress = float(
            np.clip(self._reach_out_progress, 0.0, max_progress)
        )
        nominal_animation_time = animation_time
        animation_time = int(keyframes[0] + self._reach_out_progress)
        self._n_delayed_timesteps += (
            nominal_animation_time - animation_time
        )
        self._rpic_disturbed_low_level_updates += 1
        return animation_time

    def _get_bounded_reach_progress(self) -> float:
        """Return the inherited progress, recovering from absent/stale state."""
        keyframes = self.human_animation_data[self.human_animation_id][1][
            "keyframes"
        ]
        max_progress = float(keyframes[1] - keyframes[0])
        progress = getattr(self, "_reach_out_progress", None)
        if progress is None or not np.isfinite(progress):
            current_frame = max(int(getattr(self, "animation_time", 0)), 0)
            progress = current_frame - int(keyframes[0])
        progress = float(np.clip(progress, 0.0, max_progress))
        self._reach_out_progress = progress
        return progress

    def _set_human_measurement(self, human_measurement: List[np.ndarray], time):
        measurement = np.asarray(human_measurement, dtype=float)
        current = np.asarray(self.human_measurement, dtype=float)
        self._rpic_shield_measurement_calls += 1
        if not np.allclose(measurement, current, rtol=0.0, atol=1e-10):
            self._rpic_shield_measurement_stale_mismatches += 1
        if (
            self._rpic_last_shield_measurement_time is not None
            and float(time) <= self._rpic_last_shield_measurement_time
        ):
            self._rpic_shield_measurement_time_regressions += 1
        if self._rpic_schedule.active:
            self._rpic_shield_measurement_calls_during_disturbance += 1
            if (
                self._rpic_last_shield_measurement is not None
                and not np.allclose(
                    measurement,
                    self._rpic_last_shield_measurement,
                    rtol=0.0,
                    atol=1e-10,
                )
            ):
                self._rpic_shield_measurement_changes_during_disturbance += 1
        self._rpic_last_shield_measurement = measurement.copy()
        self._rpic_last_shield_measurement_time = float(time)
        expected_consumers = 0
        if self.failsafe_controller is not None and self.shield_type == "PFL":
            expected_consumers = sum(
                int(hasattr(robot.controller, "set_human_measurement"))
                for robot in self.robots
            )
        super()._set_human_measurement(human_measurement, time)
        self._rpic_pfl_measurement_forward_calls += expected_consumers
        if self._rpic_schedule.active:
            self._rpic_pfl_measurement_forward_calls_during_disturbance += (
                expected_consumers
            )

    def _update_rpic_deadlock(self):
        if (
            self.task_phase != RobotHumanHandoverPhase.REACH_OUT
            or not self._is_robot_gripping_object()
            or self._v1_object_captured
        ):
            self._rpic_previous_target_distance = None
            self._rpic_stall_steps = 0
            return
        object_pos = np.asarray(
            self.sim.data.body_xpos[self.manipulation_object_body_id],
            dtype=float,
        )
        distance = float(np.linalg.norm(object_pos - self.target_pos))
        if self._rpic_previous_target_distance is not None:
            progress = self._rpic_previous_target_distance - distance
            if progress > self._rpic_deadlock_epsilon:
                self._rpic_stall_steps = 0
            else:
                self._rpic_stall_steps += 1
        self._rpic_previous_target_distance = distance
        self._rpic_max_stall_steps = max(
            self._rpic_max_stall_steps, self._rpic_stall_steps
        )
        if self._rpic_stall_steps >= self._rpic_deadlock_window:
            self._rpic_deadlock = True

    def step(self, action: np.ndarray):
        pre_step_grasp = (
            self.task_phase == RobotHumanHandoverPhase.REACH_OUT
            and self._is_robot_gripping_object()
            and not self._v1_object_captured
        )
        if pre_step_grasp:
            self._rpic_stable_grasp_steps += 1
        else:
            self._rpic_stable_grasp_steps = 0

        disturbed = self._rpic_schedule.begin_policy_step(
            eligible=self._disturbance_eligible()
        )
        if disturbed:
            self._rpic_disturbed_policy_steps += 1

        obs, reward, done, info = super().step(action)
        self._update_rpic_deadlock()
        self._rpic_schedule.end_policy_step()
        info.update(self._get_rpic_info())
        return obs, reward, done, info

    def _get_rpic_info(self) -> Dict:
        return {
            "rpic_condition": self._rpic_schedule.condition,
            "rpic_episode_index": self._rpic_episode_index,
            "rpic_disturbance_seed": self._rpic_schedule.episode_seed,
            "rpic_trigger_delay_steps": self._rpic_schedule.trigger_delay_steps,
            "rpic_disturbance_triggered": int(self._rpic_schedule.triggered),
            "rpic_disturbance_completed": int(self._rpic_schedule.completed),
            "rpic_disturbance_interrupted": int(self._rpic_schedule.interrupted),
            "rpic_disturbed_policy_steps": self._rpic_disturbed_policy_steps,
            "rpic_disturbed_low_level_updates": (
                self._rpic_disturbed_low_level_updates
            ),
            "rpic_stable_grasp_steps": self._rpic_stable_grasp_steps,
            "rpic_deadlock": int(self._rpic_deadlock),
            "rpic_precapture_progress_stall": int(self._rpic_deadlock),
            "rpic_max_stall_steps": self._rpic_max_stall_steps,
            "rpic_shield_measurement_calls": self._rpic_shield_measurement_calls,
            "rpic_shield_measurement_calls_during_disturbance": (
                self._rpic_shield_measurement_calls_during_disturbance
            ),
            "rpic_shield_measurement_changes_during_disturbance": (
                self._rpic_shield_measurement_changes_during_disturbance
            ),
            "rpic_shield_measurement_stale_mismatches": (
                self._rpic_shield_measurement_stale_mismatches
            ),
            "rpic_shield_measurement_time_regressions": (
                self._rpic_shield_measurement_time_regressions
            ),
            "rpic_pfl_measurement_forward_calls": (
                self._rpic_pfl_measurement_forward_calls
            ),
            "rpic_pfl_measurement_forward_calls_during_disturbance": (
                self._rpic_pfl_measurement_forward_calls_during_disturbance
            ),
        }

    def _get_info(self) -> Dict:
        info = super()._get_info()
        info.update(self._get_rpic_info())
        return info
