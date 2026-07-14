"""Wrapper that adds a dense progress reward for approaching the human's right hand."""
import gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


class DenseApproachRewardWrapper(gym.Wrapper):
    """Add a shaping reward proportional to reduction in distance to human's right hand.

    r_approach = scale * max(0, d_{t-1} - d_t)

    Only rewards getting closer (not penalises moving away) to avoid conflicting
    with the safety shield's need to stop the robot near the human.

    Args:
        env: The environment to wrap.
        scale: Multiplier on the distance delta. Default 2.0.
    """

    def __init__(self, env: gym.Env, scale: float = 2.0):
        super().__init__(env)
        self._scale = scale
        self._prev_dist: Optional[float] = None

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._prev_dist = self._get_rh_dist()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        curr_dist = self._get_rh_dist()
        if self._prev_dist is not None and curr_dist is not None:
            progress = self._prev_dist - curr_dist
            if progress > 0:
                reward += self._scale * progress
        self._prev_dist = None if done else curr_dist
        return obs, reward, done, info

    def _get_rh_dist(self) -> Optional[float]:
        """Read dist_eef_to_human_rh from the nearest wrapper that has _obs_cache."""
        base = self.env
        while base is not None:
            cache = getattr(base, "_obs_cache", None)
            if isinstance(cache, dict) and "dist_eef_to_human_rh" in cache:
                return float(np.squeeze(cache["dist_eef_to_human_rh"]))
            base = getattr(base, "env", None)
        return None
