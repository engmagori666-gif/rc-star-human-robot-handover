"""Predictive Cartesian action governor for dynamic human approach risk."""
from typing import Any, Dict, Tuple

import gym
import numpy as np


class HumanAwareActionGovernor(gym.Wrapper):
    """Scale only Cartesian motion that closes too quickly toward the human.

    The wrapper runs outside the IK wrapper, so the first three action
    dimensions are normalized Cartesian end-effector deltas. It estimates the
    nearest human-joint velocity from consecutive policy steps and predicts the
    next-step separation. Motion is scaled only when the predicted separation
    is below ``activation_distance`` and the relative motion is closing.
    Gripper commands are never changed.
    """

    def __init__(
        self,
        env: gym.Env,
        activation_distance: float = 0.30,
        hard_distance: float = 0.18,
        prediction_horizon: float = 1.0,
        minimum_scale: float = 0.25,
        max_interventions: int = 0,
        include_held_object: bool = False,
        object_clearance_margin: float = 0.0,
        include_robot_links: bool = False,
        robot_link_names: str = (
            "robot0_link_4_collision,robot0_link_5_collision"
        ),
        robot_link_clearance_margin: float = 0.02,
    ):
        super().__init__(env)
        self._activation_distance = float(activation_distance)
        self._hard_distance = float(hard_distance)
        self._prediction_horizon = float(prediction_horizon)
        self._minimum_scale = float(minimum_scale)
        self._max_interventions = max(int(max_interventions), 0)
        self._include_held_object = bool(include_held_object)
        self._object_clearance_margin = max(
            float(object_clearance_margin),
            0.0,
        )
        self._include_robot_links = bool(include_robot_links)
        self._robot_link_names = tuple(
            name.strip()
            for name in str(robot_link_names).split(",")
            if name.strip()
        )
        self._robot_link_clearance_margin = max(
            float(robot_link_clearance_margin),
            0.0,
        )
        self._previous_human_positions = None
        self._previous_robot_link_positions = {}
        self._governor_interventions = 0
        self._governor_scale_sum = 0.0
        self._governor_min_predicted_distance = np.inf
        self._governor_budget_exhausted = False

    def reset(self):
        obs = self.env.reset()
        self._previous_human_positions = self._human_positions()
        self._previous_robot_link_positions = self._robot_link_positions()
        self._governor_interventions = 0
        self._governor_scale_sum = 0.0
        self._governor_min_predicted_distance = np.inf
        self._governor_budget_exhausted = False
        return obs

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        governed_action, scale, predicted_distance = self._govern(action)
        obs, reward, done, info = self.env.step(governed_action)
        self._previous_human_positions = self._human_positions()
        self._previous_robot_link_positions = self._robot_link_positions()
        if scale < 0.999:
            self._governor_interventions += 1
            self._governor_scale_sum += scale
        self._governor_min_predicted_distance = min(
            self._governor_min_predicted_distance,
            predicted_distance,
        )
        info["human_governor_interventions"] = (
            self._governor_interventions
        )
        info["human_governor_last_scale"] = float(scale)
        info["human_governor_mean_scale"] = (
            self._governor_scale_sum / self._governor_interventions
            if self._governor_interventions
            else 1.0
        )
        info["human_governor_min_predicted_distance"] = float(
            self._governor_min_predicted_distance
        )
        info["human_governor_budget_exhausted"] = int(
            self._governor_budget_exhausted
        )
        return obs, reward, done, info

    def _govern(self, action: np.ndarray):
        governed = np.asarray(action, dtype=float).copy()
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

        # IKPositionDeltaWrapper exposes direct Cartesian deltas in meters.
        robot_delta = governed[:3].copy()
        risk = self._select_risk(
            base=base,
            human_positions=human_positions,
            robot_delta=robot_delta,
        )
        if risk is None:
            return governed, 1.0, np.inf
        (
            direction,
            robot_closing,
            relative_closing,
            predicted_distance,
            scale_all_translation,
        ) = risk
        if (
            robot_closing <= 0
            or
            relative_closing <= 0
            or predicted_distance >= self._activation_distance
        ):
            return governed, 1.0, predicted_distance

        span = max(
            self._activation_distance - self._hard_distance,
            1e-6,
        )
        scale = np.clip(
            (
                predicted_distance - self._hard_distance
            )
            / span,
            self._minimum_scale,
            1.0,
        )
        if scale_all_translation:
            governed[:3] = robot_delta * scale
        else:
            # Preserve tangential and retreating motion. Only reduce the
            # component of the Cartesian command that points directly toward
            # the human.
            governed[:3] = (
                robot_delta
                - (1.0 - scale) * robot_closing * direction
            )
        return governed, float(scale), predicted_distance

    def _select_risk(
        self,
        base,
        human_positions: np.ndarray,
        robot_delta: np.ndarray,
    ):
        human_deltas = np.zeros_like(human_positions)
        if (
            self._previous_human_positions is not None
            and self._previous_human_positions.shape
            == human_positions.shape
        ):
            human_deltas = human_positions - self._previous_human_positions

        candidates = []
        eef_pos = np.asarray(
            base.sim.data.site_xpos[base.robots[0].eef_site_id],
            dtype=float,
        )
        candidates.append((eef_pos, 0.0, robot_delta, False))
        if (
            self._include_held_object
            and getattr(base, "_is_robot_gripping_object", lambda: False)()
        ):
            for geom_name in getattr(
                getattr(base, "manipulation_object", None),
                "contact_geoms",
                [],
            ):
                try:
                    geom_id = base.sim.model.geom_name2id(geom_name)
                    point = np.asarray(
                        base.sim.data.geom_xpos[geom_id],
                        dtype=float,
                    )
                    radius = float(base.sim.model.geom_rbound[geom_id])
                    candidates.append(
                        (
                            point,
                            radius + self._object_clearance_margin,
                            robot_delta,
                            False,
                        )
                    )
                except Exception:
                    continue
        if self._include_robot_links:
            for name, point in self._robot_link_positions(base).items():
                point_delta = np.zeros(3, dtype=float)
                if name in self._previous_robot_link_positions:
                    point_delta = (
                        point - self._previous_robot_link_positions[name]
                    )
                try:
                    geom_id = base.sim.model.geom_name2id(name)
                    radius = float(base.sim.model.geom_rbound[geom_id])
                except Exception:
                    radius = 0.0
                candidates.append(
                    (
                        point,
                        radius + self._robot_link_clearance_margin,
                        point_delta,
                        True,
                    )
                )

        best = None
        best_predicted = np.inf
        for point, radius, point_delta, scale_all_translation in candidates:
            relative = human_positions - point
            distances = np.linalg.norm(relative, axis=1)
            nearest_index = int(np.argmin(distances))
            center_distance = float(distances[nearest_index])
            if center_distance <= 1e-8:
                continue
            direction = relative[nearest_index] / center_distance
            clearance = center_distance - radius
            robot_closing = float(np.dot(point_delta, direction))
            relative_closing = float(
                robot_closing
                - np.dot(human_deltas[nearest_index], direction)
            )
            predicted_distance = (
                clearance
                - self._prediction_horizon
                * max(relative_closing, 0.0)
            )
            if predicted_distance < best_predicted:
                best_predicted = predicted_distance
                best = (
                    direction,
                    robot_closing,
                    relative_closing,
                    predicted_distance,
                    scale_all_translation,
                )
        return best

    def _human_positions(self) -> np.ndarray:
        try:
            return np.asarray(
                self._base_env().human_measurement,
                dtype=float,
            )
        except Exception:
            return np.empty((0, 3), dtype=float)

    def _robot_link_positions(self, base=None):
        if base is None:
            base = self._base_env()
        positions = {}
        for name in self._robot_link_names:
            try:
                geom_id = base.sim.model.geom_name2id(name)
                positions[name] = np.asarray(
                    base.sim.data.geom_xpos[geom_id],
                    dtype=float,
                ).copy()
            except Exception:
                continue
        return positions

    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env
