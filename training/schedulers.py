class BetaAnnealingScheduler:
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        num_steps: int,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps

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
        if self._current_step >= self.num_steps:
            return self.beta_end
        else:
            return self.beta_start + (self.beta_end - self.beta_start) * (
                self._current_step / self.num_steps
            )
