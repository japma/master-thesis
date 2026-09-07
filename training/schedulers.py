class BetaAnnealingScheduler:
    """Linear ramp from `beta_start` to `beta_end` over `num_steps`, optionally held
    at `beta_start` for the first `delay_steps` -- which is how the classification
    weight waits out the KL warmup before it starts pulling on the latent."""

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        num_steps: int,
        delay_steps: int = 0,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.delay_steps = delay_steps

        self._current_step = 0

    def step(self) -> None:
        self._current_step += 1

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._current_step = value

    @property
    def beta(self) -> float:
        step = self._current_step - self.delay_steps
        if step >= self.num_steps:
            return self.beta_end
        if step <= 0:
            return self.beta_start
        return self.beta_start + (self.beta_end - self.beta_start) * (
            step / self.num_steps
        )
