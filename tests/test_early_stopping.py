"""Early stopping, and the loop's use of it.

The point of stopping early is keeping the best model, not just saving time, so these
check the restore path as hard as the stop condition.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training.early_stopping import EarlyStopping
from training.loop import CheckpointSpec, run_training_loop
from training.objectives.base import AbstractObjective, Batch, StepOutput


def model_with(value: float) -> nn.Module:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_counts_epochs_without_improvement() -> None:
    stopper = EarlyStopping(patience=3, min_delta=0.0)
    model = model_with(0.0)

    assert stopper.step(10.0, model, epoch=0) is False
    assert stopper.step(9.0, model, epoch=1) is False
    assert stopper.epochs_without_improvement == 0

    assert stopper.step(9.5, model, epoch=2) is False
    assert stopper.step(9.5, model, epoch=3) is False
    assert stopper.epochs_without_improvement == 2
    assert stopper.step(9.5, model, epoch=4) is True

    assert stopper.best_loss == 9.0
    assert stopper.best_epoch == 1


def test_improvement_must_exceed_min_delta() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.5)
    model = model_with(0.0)

    stopper.step(10.0, model, epoch=0)
    # 0.4 better is not better enough, so the counter climbs.
    assert stopper.step(9.6, model, epoch=1) is False
    assert stopper.epochs_without_improvement == 1
    # 0.6 better clears min_delta, so it counts and the counter resets.
    assert stopper.step(9.4, model, epoch=2) is False
    assert stopper.epochs_without_improvement == 0

    assert stopper.step(9.0, model, epoch=3) is False
    assert stopper.step(8.9, model, epoch=4) is True
    assert stopper.best_loss == 9.4
    assert stopper.best_epoch == 2


def test_restores_the_best_weights_not_the_last() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.0)

    stopper.step(10.0, model_with(1.0), epoch=0)
    stopper.step(5.0, model_with(2.0), epoch=1)
    stopper.step(20.0, model_with(3.0), epoch=2)

    current = model_with(3.0)
    assert stopper.restore_best_weights(current) is True
    assert current.weight.item() == pytest.approx(2.0)


def test_restore_reports_when_there_is_nothing_to_restore() -> None:
    assert EarlyStopping().restore_best_weights(model_with(1.0)) is False


class ScriptedObjective(AbstractObjective):
    """Reports a fixed sequence of validation losses, one per epoch."""

    def __init__(self, val_losses: list[float]) -> None:
        self.val_losses = val_losses
        self.epoch = 0
        self.model = nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)
        self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)
        self.saved: list[float] = []

    def train_step(self, batch: Batch) -> StepOutput:
        # Stamp the epoch into the weights so the restored checkpoint is identifiable.
        with torch.no_grad():
            self.model.weight.fill_(float(self.epoch))
        return StepOutput(metrics={"total": torch.tensor(0.0)}, batch_size=1)

    def val_step(self, batch: Batch) -> StepOutput:
        loss = self.val_losses[min(self.epoch, len(self.val_losses) - 1)]
        return StepOutput(metrics={"total": torch.tensor(loss)}, batch_size=1)

    def on_epoch_end(self) -> None:
        self.epoch += 1

    def sample(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
        self.saved.append(self.model.weight.item())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


class StubRTPT:
    def start(self) -> None: ...

    def step(self, subtitle: str = "") -> None: ...


@pytest.fixture
def quiet_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("log_scalar_metrics", "log_images", "log_checkpoint_artifact"):
        monkeypatch.setattr("training.loop." + name, lambda *a, **k: None)


def run(objective: ScriptedObjective, tmp_path: Path, **kwargs) -> None:
    loader = [(torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long))]
    run_training_loop(
        objective=objective,
        device=torch.device("cpu"),
        epochs=20,
        train_loader=loader,
        test_loader=loader,
        rtpt=StubRTPT(),
        checkpoint=CheckpointSpec(
            intermediate_path=tmp_path / "intermediate.pt",
            final_path=tmp_path / "final.pt",
            artifact_type="test",
        ),
        **kwargs,
    )


def test_loop_stops_early_and_saves_the_best_epoch(
    quiet_loop: None, tmp_path: Path
) -> None:
    # Improves for four epochs, then flat forever.
    objective = ScriptedObjective([10.0, 8.0, 6.0, 4.0] + [4.0] * 20)
    stopper = EarlyStopping(patience=3, min_delta=0.0)

    run(objective, tmp_path, early_stopping=stopper)

    assert stopper.best_epoch == 3
    assert objective.epoch < 20, "should have stopped before the last epoch"
    # The final checkpoint holds the best epoch's weights, not the last epoch's.
    assert objective.saved[-1] == pytest.approx(3.0)


def test_loop_runs_every_epoch_without_early_stopping(
    quiet_loop: None, tmp_path: Path
) -> None:
    objective = ScriptedObjective([10.0] * 25)
    run(objective, tmp_path)
    assert objective.epoch == 20


def test_loop_rejects_a_metric_the_objective_does_not_report(
    quiet_loop: None, tmp_path: Path
) -> None:
    objective = ScriptedObjective([1.0] * 25)
    with pytest.raises(KeyError, match="does not report"):
        run(
            objective,
            tmp_path,
            early_stopping=EarlyStopping(),
            early_stopping_metric="elbo",
        )
