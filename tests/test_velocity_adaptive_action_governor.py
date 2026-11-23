"""Pure numerical tests; do not instantiate MuJoCo or evaluate policies."""
import numpy as np
import pytest

from human_robot_gym.wrappers.velocity_adaptive_action_governor import (
    VelocityAdaptiveActionGovernor,
)


def governor_stub():
    """Bypass gym construction to test the threshold formula alone."""
    g = object.__new__(VelocityAdaptiveActionGovernor)
    g._policy_dt = 0.1
    g._extra_response_time = 0.1
    g._effective_deceleration = 2.0
    g._activation_distance = 0.22
    g._max_activation_distance = 0.45
    return g


def test_stationary_threshold_is_original():
    g = governor_stub()
    assert g._adaptive_threshold(0.0, 0.0) == pytest.approx(0.22)


def test_illustrative_threshold():
    g = governor_stub()
    # 0.25 m/s robot + 0.30 m/s human, with dt=0.1 s.
    assert g._adaptive_threshold(0.025, 0.055) == pytest.approx(0.328125)


def test_threshold_is_bounded():
    g = governor_stub()
    assert g._adaptive_threshold(1.0, 2.0) == pytest.approx(0.45)


def test_no_human_closing_component():
    g = governor_stub()
    assert g._adaptive_threshold(0.025, 0.025) == pytest.approx(
        0.22 + 0.25 * 0.1 + 0.25 ** 2 / 4.0
    )
