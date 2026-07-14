"""Reactive-human curriculum callbacks."""
from stable_baselines3.common.callbacks import BaseCallback


SCHEDULE = [
    (0.00, 0.0),
    (0.33, 0.5),
    (0.66, 1.0),
]


class ReactivityCurriculumCallback(BaseCallback):
    """Ramp human reactivity using legacy stages or a smooth linear schedule."""

    def __init__(
        self,
        total_timesteps: int,
        max_speed_scale: float = 0.010,
        schedule_mode: str = "staged",
        start_fraction: float = 0.0,
        end_fraction: float = 0.7,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.max_speed_scale = max_speed_scale
        self.schedule_mode = schedule_mode
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self._stage = -1
        self._last_log_bucket = -1

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        if self.schedule_mode == "linear":
            duration = max(self.end_fraction - self.start_fraction, 1e-6)
            fraction = min(max((progress - self.start_fraction) / duration, 0.0), 1.0)
            scale = fraction * self.max_speed_scale
            self._set_speed_scale(scale)
            self._log_value(scale)
            return True

        for stage, (threshold, fraction) in enumerate(SCHEDULE):
            next_threshold = (
                SCHEDULE[stage + 1][0]
                if stage + 1 < len(SCHEDULE)
                else 1.1
            )
            if threshold <= progress < next_threshold:
                if stage != self._stage:
                    self._stage = stage
                    scale = fraction * self.max_speed_scale
                    self._set_speed_scale(scale)
                    self.logger.record("curriculum/speed_scale", scale)
                    if self.verbose:
                        print(
                            f"[Reactivity Curriculum] step {self.num_timesteps}: "
                            f"speed_scale -> {scale:.4f}"
                        )
                break
        return True

    def _log_value(self, scale: float):
        bucket = int(100 * self.num_timesteps / self.total_timesteps)
        if bucket != self._last_log_bucket:
            self._last_log_bucket = bucket
            self.logger.record("curriculum/speed_scale", scale)
            if self.verbose and bucket % 10 == 0:
                print(
                    f"[Reactivity Curriculum] step {self.num_timesteps}: "
                    f"speed_scale -> {scale:.4f}"
                )

    def _set_speed_scale(self, scale: float):
        for env in self.training_env.envs:
            wrapper = env
            while wrapper is not None:
                if hasattr(wrapper, "_speed_scale"):
                    wrapper._speed_scale = scale
                    break
                wrapper = getattr(wrapper, "env", None)
