import numpy as np
import unittest
from types import SimpleNamespace

from human_robot_gym.environments.manipulation.human_motion_disturbance import (
    HumanMotionDisturbanceSchedule,
    disturbed_reach_progress,
    initial_state_fingerprint,
)
from human_robot_gym.environments.manipulation.rpic_handover_env import (
    ReactiveHumanHandoverCartV1RPIC,
)


class HumanMotionDisturbanceTests(unittest.TestCase):
    def test_episode_seeded_trigger_is_deterministic(self):
        schedules = [
            HumanMotionDisturbanceSchedule(condition, 2701, 2, 5, 8)
            for condition in ("delayed_response", "reach_reversal")
        ]
        for schedule in schedules:
            schedule.reset(episode_index=7)
        self.assertEqual(schedules[0].episode_seed, schedules[1].episode_seed)
        self.assertEqual(
            schedules[0].trigger_delay_steps,
            schedules[1].trigger_delay_steps,
        )

    def test_delay_and_reversal_are_bounded(self):
        self.assertEqual(
            disturbed_reach_progress("delayed_response", 4.0, 5.0, 0.5),
            4.0,
        )
        self.assertEqual(
            disturbed_reach_progress("reach_reversal", 4.0, 5.0, 0.5),
            3.5,
        )
        self.assertEqual(
            disturbed_reach_progress("reach_reversal", 0.2, 1.0, 0.5),
            0.0,
        )
        self.assertTrue(
            np.isclose(
                disturbed_reach_progress("normal", 4.0, 5.0, 0.5), 5.0
            )
        )

    def test_disturbance_only_starts_after_eligible_trigger_count(self):
        schedule = HumanMotionDisturbanceSchedule(
            "delayed_response", 4, 3, 3, 2
        )
        self.assertFalse(schedule.begin_policy_step(eligible=False))
        self.assertFalse(schedule.begin_policy_step(eligible=True))
        self.assertFalse(schedule.begin_policy_step(eligible=True))
        self.assertTrue(schedule.begin_policy_step(eligible=True))
        schedule.end_policy_step()
        self.assertTrue(schedule.active)
        schedule.end_policy_step()
        self.assertTrue(schedule.completed)

    def test_active_disturbance_stops_when_eligibility_is_lost(self):
        schedule = HumanMotionDisturbanceSchedule(
            "reach_reversal", 4, 1, 1, 5
        )
        self.assertTrue(schedule.begin_policy_step(eligible=True))
        self.assertFalse(schedule.begin_policy_step(eligible=False))
        self.assertFalse(schedule.active)
        self.assertTrue(schedule.interrupted)
        self.assertFalse(schedule.completed)

    def test_initial_state_fingerprint_is_stable_and_sensitive(self):
        first = initial_state_fingerprint([1.0, 2.0], [3.0])
        repeated = initial_state_fingerprint([1.0, 2.0], [3.0])
        changed = initial_state_fingerprint([1.0, 2.1], [3.0])
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)

    def test_missing_reach_progress_recovers_from_animation_frame(self):
        env = ReactiveHumanHandoverCartV1RPIC.__new__(
            ReactiveHumanHandoverCartV1RPIC
        )
        env._human_animation_ids = np.array([0])
        env._human_animation_ids_index = 0
        env.human_animation_data = [({}, {"keyframes": [10, 20]})]
        env.animation_time = 14
        self.assertEqual(env._get_bounded_reach_progress(), 4.0)
        env._reach_out_progress = 100.0
        self.assertEqual(env._get_bounded_reach_progress(), 10.0)

    def test_active_measurement_is_current_and_forwarded_to_shield_path(self):
        class MeasurementHarness(ReactiveHumanHandoverCartV1RPIC):
            @property
            def human_measurement(self):
                return self.current_measurement

        env = MeasurementHarness.__new__(MeasurementHarness)
        env._rpic_schedule = HumanMotionDisturbanceSchedule(
            "reach_reversal", 2701, 0, 0, 1
        )
        env._rpic_schedule.active_steps_remaining = 1
        env._rpic_shield_measurement_calls = 0
        env._rpic_shield_measurement_calls_during_disturbance = 0
        env._rpic_shield_measurement_changes_during_disturbance = 0
        env._rpic_shield_measurement_stale_mismatches = 0
        env._rpic_shield_measurement_time_regressions = 0
        env._rpic_pfl_measurement_forward_calls = 0
        env._rpic_pfl_measurement_forward_calls_during_disturbance = 0
        env._rpic_last_shield_measurement = None
        env._rpic_last_shield_measurement_time = None
        env.current_measurement = [np.array([0.1, 0.2, 0.3])]
        received = []
        controller = SimpleNamespace(
            set_human_measurement=lambda measurement, time: received.append(
                (measurement, time)
            )
        )
        env.robots = [SimpleNamespace(controller=controller)]
        env.failsafe_controller = [object()]
        env.shield_type = "PFL"

        env._set_human_measurement(env.current_measurement, 1.25)

        self.assertEqual(len(received), 1)
        forwarded_measurement, forwarded_time = received[0]
        self.assertTrue(
            np.allclose(forwarded_measurement, env.current_measurement)
        )
        self.assertEqual(forwarded_time, 1.25)
        self.assertEqual(
            env._rpic_shield_measurement_calls_during_disturbance, 1
        )
        self.assertEqual(env._rpic_shield_measurement_stale_mismatches, 0)
        self.assertEqual(env._rpic_shield_measurement_time_regressions, 0)
        self.assertEqual(env._rpic_pfl_measurement_forward_calls, 1)
        self.assertEqual(
            env._rpic_pfl_measurement_forward_calls_during_disturbance, 1
        )


if __name__ == "__main__":
    unittest.main()
