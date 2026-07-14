"""RSI curriculum callbacks."""
from stable_baselines3.common.callbacks import BaseCallback


BASE_SCHEDULE = [
    (0.00, 0.9),
    (0.33, 0.5),
    (0.66, 0.1),
    (0.90, 0.0),
]


class RSICurriculumCallback(BaseCallback):
    """Set RSI probability using the legacy stages or a smooth linear decay."""

    def __init__(
        self,
        total_timesteps: int,
        rsi_floor: float = 0.0,
        schedule_mode: str = "staged",
        rsi_start: float = 0.9,
        end_fraction: float = 0.7,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.rsi_floor = rsi_floor
        self.schedule_mode = schedule_mode
        self.rsi_start = rsi_start
        self.end_fraction = end_fraction
        self._stage = -1
        self._last_log_bucket = -1
        self._schedule = [(t, max(p, rsi_floor)) for t, p in BASE_SCHEDULE]

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        if self.schedule_mode == "linear":
            fraction = min(progress / max(self.end_fraction, 1e-6), 1.0)
            prob = self.rsi_start + fraction * (self.rsi_floor - self.rsi_start)
            self._set_rsi_prob(prob)
            self._log_value(prob)
            return True

        for stage, (threshold, prob) in enumerate(self._schedule):
            next_threshold = (
                self._schedule[stage + 1][0]
                if stage + 1 < len(self._schedule)
                else 1.1
            )
            if threshold <= progress < next_threshold:
                if stage != self._stage:
                    self._stage = stage
                    self._set_rsi_prob(prob)
                    self.logger.record("curriculum/rsi_prob", prob)
                    if self.verbose:
                        print(
                            f"[RSI Curriculum] step {self.num_timesteps}: "
                            f"rsi_prob -> {prob:.4f}"
                        )
                break
        return True

    def _log_value(self, prob: float):
        bucket = int(100 * self.num_timesteps / self.total_timesteps)
        if bucket != self._last_log_bucket:
            self._last_log_bucket = bucket
            self.logger.record("curriculum/rsi_prob", prob)
            if self.verbose and bucket % 10 == 0:
                print(
                    f"[RSI Curriculum] step {self.num_timesteps}: "
                    f"rsi_prob -> {prob:.4f}"
                )

    def _set_rsi_prob(self, prob: float):
        for env in self.training_env.envs:
            wrapper = env
            while wrapper is not None:
                if hasattr(wrapper, "_rsi_prob"):
                    wrapper._rsi_prob = prob
                    break
                wrapper = getattr(wrapper, "env", None)
