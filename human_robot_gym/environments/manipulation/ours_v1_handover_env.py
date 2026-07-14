"""Ours V1: reactive, safe, success-directed robot-to-human handover."""
from typing import Any, Dict, List

import numpy as np

from human_robot_gym.environments.manipulation.reactive_handover_env import (
    ReactiveHumanHandoverCart,
)
from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (
    RobotHumanHandoverPhase,
)


class ReactiveHumanHandoverCartV1(ReactiveHumanHandoverCart):
    """Reactive handover with progress reward and active human acceptance.

    Exact palm-mesh contact is brittle for a moving hand and an elongated hammer.
    During REACH_OUT, the human accepts the object inside a configurable capture
    region, but only while the robot still has a valid grasp. The normal hand weld
    is then activated, so success still represents a physical transfer.
    """

    def __init__(
        self,
        robots,
        v1_step_penalty: float = 0.01,
        v1_approach_progress_scale: float = 2.0,
        v1_transport_progress_scale: float = 4.0,
        v1_grasp_bonus: float = 2.0,
        v1_drop_penalty: float = 4.0,
        v1_success_reward: float = 20.0,
        v1_capture_bonus: float = 5.0,
        v1_max_progress_per_step: float = 0.05,
        v1_capture_surface_dist: float = 0.06,
        v1_require_release: bool = False,
        v1_release_hold_steps: int = 5,
        v1_release_aperture_threshold: float = 0.02,
        v1_release_supervisor: bool = False,
        v1_gate_human_approach_until_grasp: bool = False,
        v1_human_yield_distance: float = 0.0,
        v1_cooperative_human_reach: bool = False,
        v1_cooperative_reach_target_dist: float = 0.10,
        v1_cooperative_reach_max_offset: float = 0.10,
        v1_cooperative_reach_step: float = 0.002,
        v1_cooperative_reach_delay_steps: int = 0,
        v1_reachout_human_yield_distance: float = 0.0,
        v1_reachout_human_yield_link_names: str = (
            "robot0_link_4_collision,robot0_link_5_collision"
        ),
        v1_link_hand_clearance_margin: float = 0.0,
        v1_link_hand_clearance_penalty: float = 0.0,
        v1_link_hand_clearance_link_names: str = (
            "robot0_link_4_collision,robot0_link_5_collision"
        ),
        **kwargs,
    ):
        self._v1_step_penalty = v1_step_penalty
        self._v1_approach_progress_scale = v1_approach_progress_scale
        self._v1_transport_progress_scale = v1_transport_progress_scale
        self._v1_grasp_bonus = v1_grasp_bonus
        self._v1_drop_penalty = v1_drop_penalty
        self._v1_success_reward = v1_success_reward
        self._v1_capture_bonus = v1_capture_bonus
        self._v1_max_progress_per_step = v1_max_progress_per_step
        self._v1_capture_surface_dist = v1_capture_surface_dist
        self._v1_require_release = v1_require_release
        self._v1_release_hold_steps = max(int(v1_release_hold_steps), 1)
        self._v1_release_aperture_threshold = (
            v1_release_aperture_threshold
        )
        self._v1_release_supervisor = v1_release_supervisor
        self._v1_gate_human_approach_until_grasp = (
            v1_gate_human_approach_until_grasp
        )
        self._v1_human_yield_distance = max(
            float(v1_human_yield_distance),
            0.0,
        )
        self._v1_cooperative_human_reach = bool(
            v1_cooperative_human_reach
        )
        self._v1_cooperative_reach_target_dist = max(
            float(v1_cooperative_reach_target_dist),
            0.0,
        )
        self._v1_cooperative_reach_max_offset = max(
            float(v1_cooperative_reach_max_offset),
            0.0,
        )
        self._v1_cooperative_reach_step = max(
            float(v1_cooperative_reach_step),
            0.0,
        )
        self._v1_cooperative_reach_delay_steps = max(
            int(v1_cooperative_reach_delay_steps),
            0,
        )
        self._v1_reachout_human_yield_distance = max(
            float(v1_reachout_human_yield_distance),
            0.0,
        )
        self._v1_reachout_human_yield_link_names = tuple(
            name.strip()
            for name in str(v1_reachout_human_yield_link_names).split(",")
            if name.strip()
        )
        self._v1_link_hand_clearance_margin = max(
            float(v1_link_hand_clearance_margin),
            0.0,
        )
        self._v1_link_hand_clearance_penalty = max(
            float(v1_link_hand_clearance_penalty),
            0.0,
        )
        self._v1_link_hand_clearance_link_names = tuple(
            name.strip()
            for name in str(v1_link_hand_clearance_link_names).split(",")
            if name.strip()
        )

        self._v1_prev_eef_obj_dist = None
        self._v1_prev_obj_hand_dist = None
        self._v1_prev_object_gripped = False
        self._v1_has_reward_history = False
        self._v1_capture_count = 0
        self._v1_drop_count = 0
        self._v1_object_captured = False
        self._v1_release_hold_count = 0
        self._v1_release_count = 0
        self._v1_release_confirmed = False
        self._v1_release_supervisor_steps = 0
        self._v1_human_approach_released = False
        self._v1_human_yield_steps = 0
        self._v1_nominal_human_pos_offset = np.zeros(3, dtype=float)
        self._v1_cooperative_reach_offset = np.zeros(3, dtype=float)
        self._v1_cooperative_reach_steps = 0
        self._v1_cooperative_reach_ready_steps = 0
        self._v1_reachout_human_yield_steps = 0
        self._v1_min_link_hand_clearance = np.inf
        self._v1_last_link_hand_clearance_penalty = 0.0
        self._v1_link_hand_clearance_violations = 0
        self._v1_previous_illegal_collision_count = 0
        self._v1_last_collision_penalty = 0.0
        super().__init__(robots=robots, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        self._v1_prev_eef_obj_dist = None
        self._v1_prev_obj_hand_dist = None
        self._v1_prev_object_gripped = False
        self._v1_has_reward_history = False
        self._v1_capture_count = 0
        self._v1_drop_count = 0
        self._v1_object_captured = False
        self._v1_release_hold_count = 0
        self._v1_release_count = 0
        self._v1_release_confirmed = False
        self._v1_release_supervisor_steps = 0
        self._v1_human_approach_released = False
        self._v1_human_yield_steps = 0
        self._v1_nominal_human_pos_offset = np.asarray(
            self.human_pos_offset,
            dtype=float,
        ).copy()
        self._v1_cooperative_reach_offset = np.zeros(3, dtype=float)
        self._v1_cooperative_reach_steps = 0
        self._v1_cooperative_reach_ready_steps = 0
        self._v1_reachout_human_yield_steps = 0
        self._v1_min_link_hand_clearance = np.inf
        self._v1_last_link_hand_clearance_penalty = 0.0
        self._v1_link_hand_clearance_violations = 0
        self._v1_previous_illegal_collision_count = 0
        self._v1_last_collision_penalty = 0.0

    def _compute_animation_time(self, control_time: int) -> int:
        if (
            self._v1_human_yield_distance > 0
            and self.task_phase == RobotHumanHandoverPhase.APPROACH
            and not self._is_robot_gripping_object()
            and self._get_min_eef_human_distance()
            < self._v1_human_yield_distance
        ):
            # Keep the current pose and move the animation origin forward so
            # the human resumes smoothly instead of jumping ahead.
            self.animation_start_time = control_time - self.animation_time
            self._v1_human_yield_steps += 1
            return self.animation_time

        if (
            self._v1_reachout_human_yield_distance > 0
            and self.task_phase == RobotHumanHandoverPhase.REACH_OUT
            and self._is_robot_gripping_object()
            and not self._v1_object_captured
            and self._get_min_reachout_link_hand_clearance()
            < self._v1_reachout_human_yield_distance
        ):
            self.animation_start_time = control_time - self.animation_time
            self._v1_reachout_human_yield_steps += 1
            return self.animation_time

        if (
            self._v1_gate_human_approach_until_grasp
            and not self._v1_human_approach_released
            and self.task_phase == RobotHumanHandoverPhase.APPROACH
        ):
            if not self._is_robot_gripping_object():
                return 0
            # Restart the normal animation clock at the first stable grasp.
            self.animation_start_time = control_time
            self._v1_human_approach_released = True

        return super()._compute_animation_time(control_time=control_time)

    def _get_min_eef_human_distance(self) -> float:
        try:
            eef_pos = np.asarray(
                self.sim.data.site_xpos[self.robots[0].eef_site_id],
                dtype=float,
            )
            return min(
                float(np.linalg.norm(eef_pos - np.asarray(joint_pos)))
                for joint_pos in self.human_measurement
            )
        except Exception:
            return np.inf

    def _get_min_reachout_link_hand_clearance(self) -> float:
        return self._get_min_link_hand_clearance(
            self._v1_reachout_human_yield_link_names
        )

    def _get_min_link_hand_clearance(self, link_names) -> float:
        try:
            hand_geom_ids = [
                self._l_palm_contact_geom_id,
                self._r_palm_contact_geom_id,
            ]
            best = np.inf
            for link_name in link_names:
                link_id = self.sim.model.geom_name2id(link_name)
                link_pos = np.asarray(
                    self.sim.data.geom_xpos[link_id],
                    dtype=float,
                )
                link_radius = float(self.sim.model.geom_rbound[link_id])
                for hand_id in hand_geom_ids:
                    hand_pos = np.asarray(
                        self.sim.data.geom_xpos[hand_id],
                        dtype=float,
                    )
                    hand_radius = float(self.sim.model.geom_rbound[hand_id])
                    clearance = (
                        float(np.linalg.norm(link_pos - hand_pos))
                        - link_radius
                        - hand_radius
                    )
                    best = min(best, clearance)
            return best
        except Exception:
            return np.inf

    def step(self, action: np.ndarray):
        """Execute an action, then let the reactive human accept a nearby object."""
        self._update_cooperative_human_reach()
        if (
            self._v1_require_release
            and self._v1_release_supervisor
            and self._v1_object_captured
            and not self._v1_release_confirmed
        ):
            action = np.array(action, dtype=float, copy=True)
            action[-1] = -1.0
            self._v1_release_supervisor_steps += 1

        phase_before = self.task_phase
        obs, reward, done, info = super().step(action)

        captured_by_contact = (
            phase_before == RobotHumanHandoverPhase.REACH_OUT
            and self.task_phase == RobotHumanHandoverPhase.RETREAT
        )
        captured_by_region = False

        if (
            self.task_phase == RobotHumanHandoverPhase.REACH_OUT
            and self._is_robot_gripping_object()
        ):
            contact_pos, surface_gap = self._get_capture_contact()
            if surface_gap <= self._v1_capture_surface_dist:
                self._human_pickup_object(contact_pos=contact_pos)
                self.task_phase = RobotHumanHandoverPhase.RETREAT
                captured_by_region = True

        captured = captured_by_contact or captured_by_region
        if captured and not self._v1_object_captured:
            self._v1_capture_count += 1
            self._v1_object_captured = True

            if self._v1_require_release:
                reward += self._scaled_v1_reward(self._v1_capture_bonus)
                info = self._get_info()
            else:
                self.goal_reached = True
                self.n_goal_reached += 1

                achieved_goal = self._get_achieved_goal_from_obs(
                    observation=obs
                )
                desired_goal = self._get_desired_goal_from_obs(
                    observation=obs
                )
                info = self._get_info()

                # Replace the pre-capture non-terminal reward computed by the
                # parent step with the terminal transfer reward.
                reward = self._terminal_transfer_reward()
                done = self._compute_done(
                    achieved_goal=achieved_goal,
                    desired_goal=desired_goal,
                    info=info,
                )

        if (
            self._v1_require_release
            and self._v1_object_captured
            and not self._v1_release_confirmed
        ):
            release_is_stable = (
                self._is_human_weld_active()
                and not self._is_robot_gripping_object()
                and self._get_gripper_aperture()
                >= self._v1_release_aperture_threshold
            )
            if release_is_stable:
                self._v1_release_hold_count += 1
            else:
                self._v1_release_hold_count = 0

            if self._v1_release_hold_count >= self._v1_release_hold_steps:
                self._v1_release_count += 1
                self._v1_release_confirmed = True
                self.goal_reached = True
                self.n_goal_reached += 1

                achieved_goal = self._get_achieved_goal_from_obs(
                    observation=obs
                )
                desired_goal = self._get_desired_goal_from_obs(
                    observation=obs
                )
                info = self._get_info()
                reward = self._terminal_transfer_reward()
                done = self._compute_done(
                    achieved_goal=achieved_goal,
                    desired_goal=desired_goal,
                    info=info,
                )

        info["v1_capture_count"] = self._v1_capture_count
        info["v1_drop_count"] = self._v1_drop_count
        info["v1_object_captured"] = int(self._v1_object_captured)
        info["v1_release_hold_count"] = self._v1_release_hold_count
        info["v1_release_count"] = self._v1_release_count
        info["v1_release_confirmed"] = int(self._v1_release_confirmed)
        info["v1_release_supervisor_steps"] = (
            self._v1_release_supervisor_steps
        )
        info["v1_human_approach_released"] = int(
            self._v1_human_approach_released
        )
        info["v1_human_yield_steps"] = self._v1_human_yield_steps
        info["v1_cooperative_reach_steps"] = (
            self._v1_cooperative_reach_steps
        )
        info["v1_cooperative_reach_offset"] = float(
            np.linalg.norm(self._v1_cooperative_reach_offset)
        )
        info["v1_cooperative_reach_ready_steps"] = (
            self._v1_cooperative_reach_ready_steps
        )
        info["v1_reachout_human_yield_steps"] = (
            self._v1_reachout_human_yield_steps
        )
        info["v1_min_link_hand_clearance"] = (
            self._safe_clearance_for_logging(
                self._v1_min_link_hand_clearance
            )
        )
        info["v1_link_hand_clearance_penalty"] = (
            self._v1_last_link_hand_clearance_penalty
        )
        info["v1_link_hand_clearance_violations"] = (
            self._v1_link_hand_clearance_violations
        )
        return obs, float(reward), done, info

    def _update_cooperative_human_reach(self) -> None:
        """Move the human smoothly toward a grasped but unreachable object."""
        if (
            not self._v1_cooperative_human_reach
            or self.task_phase != RobotHumanHandoverPhase.REACH_OUT
            or not self._is_robot_gripping_object()
            or self._v1_object_captured
        ):
            self._v1_cooperative_reach_ready_steps = 0
            return

        self._v1_cooperative_reach_ready_steps += 1
        if (
            self._v1_cooperative_reach_ready_steps
            <= self._v1_cooperative_reach_delay_steps
        ):
            return

        object_pos = np.asarray(
            self.sim.data.body_xpos[self.manipulation_object_body_id],
            dtype=float,
        )
        target_pos = np.asarray(self.target_pos, dtype=float)
        planar_delta = object_pos[:2] - target_pos[:2]
        planar_distance = float(np.linalg.norm(planar_delta))
        remaining_offset = (
            self._v1_cooperative_reach_max_offset
            - float(np.linalg.norm(self._v1_cooperative_reach_offset[:2]))
        )
        required_reach = (
            planar_distance
            - self._v1_cooperative_reach_target_dist
        )
        step = min(
            self._v1_cooperative_reach_step,
            remaining_offset,
            required_reach,
        )
        if step <= 0 or planar_distance <= 1e-8:
            return

        self._v1_cooperative_reach_offset[:2] += (
            step * planar_delta / planar_distance
        )
        self.human_pos_offset = (
            self._v1_nominal_human_pos_offset
            + self._v1_cooperative_reach_offset
        ).tolist()
        self._v1_cooperative_reach_steps += 1

    def _check_success(
        self,
        achieved_goal: List[float],
        desired_goal: List[float],
    ) -> bool:
        if self._v1_require_release:
            return self._v1_release_confirmed
        return super()._check_success(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )

    def _terminal_transfer_reward(self) -> float:
        reward = (
            self._v1_success_reward
            - self._v1_step_penalty
            + self._v1_last_collision_penalty
        )
        return self._scaled_v1_reward(reward)

    def _scaled_v1_reward(self, reward: float) -> float:
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return float(reward)

    def _is_human_weld_active(self) -> bool:
        return bool(
            self.sim.model.eq_active[
                self._manipulation_object_weld_eq_id
            ]
        )

    def _get_gripper_aperture(self) -> float:
        qpos = np.asarray(
            [
                self.sim.data.get_joint_qpos(joint)
                for joint in self.robots[0].gripper.joints
            ],
            dtype=float,
        ).reshape(-1)
        if qpos.size >= 2:
            return float(qpos[0] - qpos[1])
        if qpos.size == 1:
            return float(abs(qpos[0]))
        return 0.0

    def _get_capture_contact(self):
        """Return the nearest hammer-surface point and palm-to-surface gap.

        MuJoCo 2.1 in this project does not expose ``mj_geomDistance`` through
        mujoco_py, so compute exact point distances for the hammer's box and
        cylinder collision geoms. The palm collision geom is spherical here.
        """
        palm_geom_id = (
            self._l_palm_contact_geom_id
            if self.object_holding_hand == "left"
            else self._r_palm_contact_geom_id
        )
        palm_pos = np.asarray(self.sim.data.geom_xpos[palm_geom_id], dtype=float)
        palm_radius = float(self.sim.model.geom_rbound[palm_geom_id])

        best_contact = np.asarray(
            self.sim.data.body_xpos[self.manipulation_object_body_id],
            dtype=float,
        )
        best_gap = np.inf

        for geom_name in self.manipulation_object.contact_geoms:
            geom_id = self.sim.model.geom_name2id(geom_name)
            geom_pos = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
            rotation = np.asarray(
                self.sim.data.geom_xmat[geom_id],
                dtype=float,
            ).reshape(3, 3)
            local_point = rotation.T @ (palm_pos - geom_pos)
            geom_type = int(self.sim.model.geom_type[geom_id])
            geom_size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)

            if geom_type == 6:  # mjGEOM_BOX
                closest_local = np.clip(local_point, -geom_size, geom_size)
            elif geom_type == 5:  # mjGEOM_CYLINDER
                radial = local_point[:2]
                radial_norm = float(np.linalg.norm(radial))
                if radial_norm > geom_size[0]:
                    radial = radial * (geom_size[0] / radial_norm)
                closest_local = np.array(
                    [
                        radial[0],
                        radial[1],
                        np.clip(local_point[2], -geom_size[1], geom_size[1]),
                    ]
                )
            else:
                direction = local_point
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 0:
                    direction = (
                        direction
                        * float(self.sim.model.geom_rbound[geom_id])
                        / direction_norm
                    )
                closest_local = direction

            contact = geom_pos + rotation @ closest_local
            gap = max(float(np.linalg.norm(palm_pos - contact)) - palm_radius, 0.0)
            if gap < best_gap:
                best_gap = gap
                best_contact = contact

        return best_contact, best_gap

    def reward(
        self,
        achieved_goal: List[float],
        desired_goal: List[float],
        info: Dict[str, Any],
    ) -> float:
        eef_pos = np.asarray(achieved_goal[:3], dtype=float)
        obj_pos = np.asarray(achieved_goal[3:6], dtype=float)
        hand_pos = np.asarray(desired_goal, dtype=float)
        object_gripped = bool(achieved_goal[6])

        eef_obj_dist = float(np.linalg.norm(eef_pos - obj_pos))
        obj_hand_dist = float(np.linalg.norm(obj_pos - hand_pos))
        success = self._check_success(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )

        reward = -self._v1_step_penalty
        if success:
            reward += self._v1_success_reward
        elif self._v1_require_release and self._v1_object_captured:
            # After the human weld is active, the remaining objective is a
            # deliberate robot release. Do not guide the arm back to the object.
            pass
        elif object_gripped:
            reward += self._progress_reward(
                previous=self._v1_prev_obj_hand_dist,
                current=obj_hand_dist,
                scale=self._v1_transport_progress_scale,
            )
        else:
            reward += self._progress_reward(
                previous=self._v1_prev_eef_obj_dist,
                current=eef_obj_dist,
                scale=self._v1_approach_progress_scale,
            )

        if self._v1_has_reward_history:
            if object_gripped and not self._v1_prev_object_gripped:
                reward += self._v1_grasp_bonus
            elif (
                self._v1_prev_object_gripped
                and not object_gripped
                and not success
                and not (
                    self._v1_require_release
                    and self._v1_object_captured
                )
            ):
                reward -= self._v1_drop_penalty
                self._v1_drop_count += 1

        reward += self._link_hand_clearance_reward()
        reward += self._collision_reward(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            info=info,
        )

        self._v1_prev_eef_obj_dist = eef_obj_dist
        self._v1_prev_obj_hand_dist = obj_hand_dist
        self._v1_prev_object_gripped = object_gripped
        self._v1_has_reward_history = True

        if self.reward_scale is not None:
            reward *= self.reward_scale
        return float(reward)

    def _link_hand_clearance_reward(self) -> float:
        """Penalize risky arm-link proximity to the receiving hand.

        This shaping is disabled by default. V1.1 enables it only while the
        robot is carrying the object toward the human but before capture, which
        targets the observed failure mode without changing approach or release.
        """
        self._v1_last_link_hand_clearance_penalty = 0.0
        if (
            self._v1_link_hand_clearance_margin <= 0
            or self._v1_link_hand_clearance_penalty <= 0
            or self.task_phase != RobotHumanHandoverPhase.REACH_OUT
            or not self._is_robot_gripping_object()
            or self._v1_object_captured
        ):
            return 0.0

        clearance = self._get_min_link_hand_clearance(
            self._v1_link_hand_clearance_link_names
        )
        self._v1_min_link_hand_clearance = min(
            self._v1_min_link_hand_clearance,
            clearance,
        )
        if not np.isfinite(clearance):
            return 0.0

        violation = self._v1_link_hand_clearance_margin - clearance
        if violation <= 0:
            return 0.0

        normalized_violation = min(
            violation / self._v1_link_hand_clearance_margin,
            1.0,
        )
        self._v1_link_hand_clearance_violations += 1
        self._v1_last_link_hand_clearance_penalty = -float(
            self._v1_link_hand_clearance_penalty
            * normalized_violation
        )
        return self._v1_last_link_hand_clearance_penalty

    def _safe_clearance_for_logging(self, clearance: float) -> float:
        if np.isfinite(clearance):
            return float(clearance)
        return 999.0

    def _collision_reward(
        self,
        achieved_goal: List[float],
        desired_goal: List[float],
        info: Dict[str, Any],
    ) -> float:
        """Penalize new illegal collision events, not their persistent flag.

        ``HumanEnv`` aggregates collision flags across the low-level simulation
        substeps of one policy step. The episode counters are cumulative, so
        their positive delta identifies a new illegal event while guaranteeing
        at most one collision penalty per policy step.
        """
        illegal_collision_count = (
            int(info["n_collisions_static"])
            + int(info["n_collisions_robot"])
            + int(info["n_collisions_critical"])
        )
        has_new_illegal_collision = (
            illegal_collision_count
            > self._v1_previous_illegal_collision_count
        )
        self._v1_previous_illegal_collision_count = illegal_collision_count
        self._v1_last_collision_penalty = (
            float(self.collision_reward)
            if has_new_illegal_collision
            else 0.0
        )
        return self._v1_last_collision_penalty

    def _progress_reward(self, previous, current: float, scale: float) -> float:
        if previous is None:
            return 0.0
        progress = np.clip(
            previous - current,
            -self._v1_max_progress_per_step,
            self._v1_max_progress_per_step,
        )
        return float(scale * progress)

    def _get_info(self) -> Dict:
        info = super()._get_info()
        info["v1_capture_count"] = self._v1_capture_count
        info["v1_drop_count"] = self._v1_drop_count
        info["v1_object_captured"] = int(self._v1_object_captured)
        info["v1_release_hold_count"] = self._v1_release_hold_count
        info["v1_release_count"] = self._v1_release_count
        info["v1_release_confirmed"] = int(self._v1_release_confirmed)
        info["v1_release_supervisor_steps"] = (
            self._v1_release_supervisor_steps
        )
        info["v1_human_approach_released"] = int(
            self._v1_human_approach_released
        )
        info["v1_human_yield_steps"] = self._v1_human_yield_steps
        info["v1_cooperative_reach_steps"] = (
            self._v1_cooperative_reach_steps
        )
        info["v1_cooperative_reach_offset"] = float(
            np.linalg.norm(self._v1_cooperative_reach_offset)
        )
        info["v1_cooperative_reach_ready_steps"] = (
            self._v1_cooperative_reach_ready_steps
        )
        info["v1_reachout_human_yield_steps"] = (
            self._v1_reachout_human_yield_steps
        )
        info["v1_min_link_hand_clearance"] = (
            self._safe_clearance_for_logging(
                self._v1_min_link_hand_clearance
            )
        )
        info["v1_link_hand_clearance_penalty"] = (
            self._v1_last_link_hand_clearance_penalty
        )
        info["v1_link_hand_clearance_violations"] = (
            self._v1_link_hand_clearance_violations
        )
        return info
