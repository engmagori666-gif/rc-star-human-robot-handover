"""Experimental stopping-distance-inspired governor for ICRCA 2027.

Evaluation-only variant of HumanAwareActionGovernor. This is NOT a certified
safety mechanism: commanded displacement and assumed deceleration are proxies,
and the independent PFL shield remains responsible for its own safeguards.
"""
from typing import Any, Dict, Tuple

import numpy as np

from human_robot_gym.wrappers.human_aware_action_governor import (
    HumanAwareActionGovernor,
)


class VelocityAdaptiveActionGovernor(HumanAwareActionGovernor):
    """Adaptive activation distance, preserving V1 closing-component scaling.

    policy_dt: seconds per policy step (0.1 for the final 10 Hz evaluation).
    extra_response_time: additional response allowance beyond the existing
        prediction horizon, in seconds.
    effective_deceleration: positive *assumed* deceleration in m/s^2; must be
        calibrated before any physical stopping-distance interpretation.
    max_activation_distance: upper bound on the adaptive threshold in meters.

    Human closing speed is derived from consecutive measured human positions,
    not an oracle velocity. Robot closing speed is the proposed Cartesian
    action divided by policy_dt, not measured executed robot speed.
    """

    def __init__(
        self,
        env,
        policy_dt: float = 0.1,
        extra_response_time: float = 0.1,
        effective_deceleration: float = 2.0,
        max_activation_distance: float = 0.45,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self._policy_dt = float(policy_dt)
        self._extra_response_time = float(extra_response_time)
        self._effective_deceleration = float(effective_deceleration)
        self._max_activation_distance = float(max_activation_distance)
        if self._policy_dt <= 0:
            raise ValueError("policy_dt must be positive")
        if self._extra_response_time < 0:
            raise ValueError("extra_response_time must be nonnegative")
        if self._effective_deceleration <= 0:
            raise ValueError("effective_deceleration must be positive")
        if self._max_activation_distance < self._activation_distance:
            raise ValueError("max_activation_distance must be >= activation_distance")
        self._adaptive_last_threshold = self._activation_distance

    def reset(self):
        obs = super().reset()
        self._adaptive_last_threshold = self._activation_distance
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        info["human_governor_adaptive_threshold"] = float(
            self._adaptive_last_threshold
        )
        return obs, reward, done, info

    def _adaptive_threshold(self, robot_closing: float, relative_closing: float) -> float:
        """Compute distance threshold from per-policy-step projected displacements.

        robot_closing and relative_closing have units of meters/step.
        """
        robot_speed = max(float(robot_closing), 0.0) / self._policy_dt
        human_speed = max(float(relative_closing - robot_closing), 0.0) / self._policy_dt
        a = self._effective_deceleration
        distance = (
            self._activation_distance
            + (robot_speed + human_speed) * self._extra_response_time
            + robot_speed ** 2 / (2.0 * a)
            + human_speed * robot_speed / a
        )
        return float(np.clip(
            distance, self._activation_distance, self._max_activation_distance
        ))

    def _govern(self, action: np.ndarray):
        governed = np.asarray(action, dtype=float).copy()
        self._adaptive_last_threshold = self._activation_distance
        if (
            self._max_interventions > 0
            and self._governor_interventions >= self._max_interventions
        ):
            self._governor_budget_exhausted = True
            return governed, 1.0, np.inf
        if governed.size < 3:
            return governed, 1.0, np.inf
        base = self._base_env()
        if getattr(base, "_v1_object_captured", False):
            return governed, 1.0, np.inf
        human_positions = self._human_positions()
        if not len(human_positions):
            return governed, 1.0, np.inf

        robot_delta = governed[:3].copy()
        risk = self._select_risk(
            base=base, human_positions=human_positions, robot_delta=robot_delta
        )
        if risk is None:
            return governed, 1.0, np.inf
        (
            direction, robot_closing, relative_closing,
            predicted_distance, scale_all_translation,
        ) = risk
        threshold = self._adaptive_threshold(robot_closing, relative_closing)
        self._adaptive_last_threshold = threshold
        if (
            robot_closing <= 0
            or relative_closing <= 0
            or predicted_distance >= threshold
        ):
            return governed, 1.0, predicted_distance

        span = max(threshold - self._hard_distance, 1e-6)
        scale = float(np.clip(
            (predicted_distance - self._hard_distance) / span,
            self._minimum_scale, 1.0,
        ))
        if scale_all_translation:
            governed[:3] = robot_delta * scale
        else:
            governed[:3] = (
                robot_delta - (1.0 - scale) * robot_closing * direction
            )
        return governed, scale, predicted_distance
