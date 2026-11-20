"""Deterministic, evaluation-only human-motion disturbance scheduling."""
from dataclasses import dataclass
import hashlib

import numpy as np


VALID_RPIC_CONDITIONS = ("normal", "delayed_response", "reach_reversal")


@dataclass
class HumanMotionDisturbanceSchedule:
    """Episode-local schedule that does not consume the environment RNG."""

    condition: str
    base_seed: int
    trigger_min_steps: int = 2
    trigger_max_steps: int = 5
    duration_steps: int = 8

    def __post_init__(self):
        if self.condition not in VALID_RPIC_CONDITIONS:
            raise ValueError(
                f"Unknown RPIC condition {self.condition!r}; expected one of "
                f"{VALID_RPIC_CONDITIONS}"
            )
        if self.trigger_min_steps < 0:
            raise ValueError("trigger_min_steps must be non-negative")
        if self.trigger_max_steps < self.trigger_min_steps:
            raise ValueError("trigger_max_steps must be >= trigger_min_steps")
        if self.duration_steps < 1:
            raise ValueError("duration_steps must be positive")
        self.reset(episode_index=0)

    def reset(self, episode_index: int):
        self.episode_index = int(episode_index)
        self.episode_seed = int(self.base_seed + 1009 * self.episode_index)
        rng = np.random.RandomState(self.episode_seed)
        self.trigger_delay_steps = int(
            rng.randint(self.trigger_min_steps, self.trigger_max_steps + 1)
        )
        self.eligible_steps = 0
        self.active_steps_remaining = 0
        self.triggered = False
        self.completed = False
        self.interrupted = False

    @property
    def active(self) -> bool:
        return self.active_steps_remaining > 0

    def begin_policy_step(self, eligible: bool) -> bool:
        """Advance the trigger clock and return whether this step is disturbed."""
        if self.condition == "normal" or self.completed:
            return False
        if self.active and not eligible:
            self.interrupt()
            return False
        if not self.triggered and eligible:
            self.eligible_steps += 1
            if self.eligible_steps >= self.trigger_delay_steps:
                self.triggered = True
                self.active_steps_remaining = self.duration_steps
        return self.active

    def end_policy_step(self):
        if self.active_steps_remaining > 0:
            self.active_steps_remaining -= 1
            if self.active_steps_remaining == 0:
                self.completed = True

    def interrupt(self):
        """Stop an active event immediately when pre-capture eligibility is lost."""
        if self.active:
            self.active_steps_remaining = 0
            self.interrupted = True


def initial_state_fingerprint(*arrays) -> str:
    """Return a stable fingerprint for paired numeric initial-state arrays."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        values = np.round(values, decimals=12)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def disturbed_reach_progress(
    condition: str,
    previous_progress: float,
    nominal_progress: float,
    reversal_speed: float,
) -> float:
    """Apply a bounded disturbance to one low-level reach update."""
    if condition == "delayed_response":
        return float(previous_progress)
    if condition == "reach_reversal":
        return float(max(previous_progress - reversal_speed, 0.0))
    return float(nominal_progress)
