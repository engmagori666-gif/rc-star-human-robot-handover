"""Wrapper that adds a per-step reward penalty each time the safety shield fires."""
import gym
import numpy as np
from typing import Any, Dict, Tuple


class FailsafePenaltyWrapper(gym.Wrapper):
    """Subtract `penalty` from the reward for every shield activation in the step.

    `info['failsafe_interventions']` is a cumulative episode counter, so we
    track the delta between consecutive steps to get the per-step count.

    Args:
        env: The environment to wrap.
        penalty: Reward deducted per activation. Default 0.01.
    """

    def __init__(self, env: gym.Env, penalty: float = 0.01):
        super().__init__(env)
        self._penalty = penalty
        self._prev_failsafe = 0

    def reset(self) -> np.ndarray:
        self._prev_failsafe = 0
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        current = info.get("failsafe_interventions", 0)
        delta = current - self._prev_failsafe
        if delta > 0:
            reward -= self._penalty * delta
        self._prev_failsafe = 0 if done else current
        return obs, reward, done, info
