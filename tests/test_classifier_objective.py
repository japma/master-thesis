"""ClassifierObjective against the contract run_training_loop relies on."""

import tempfile
from pathlib import Path

import pytest
import torch

from models.classifier import DigitClassifier
from training.metrics import PerClassAccuracy
from training.objectives.base import Batch
from training.objectives.classifier import ClassifierObjective
from utils.checkpoints import load_classifier_from_path
from utils.config import ClassifierConfig
from utils.reproducibility import seed_everything

CARDINALITIES = [10, 6, 3]
IMAGE_SIZE = 8


def build_objective() -> ClassifierObjective:
    seed_everything(0)
    model = DigitClassifier(
        config=ClassifierConfig(image_size=IMAGE_SIZE, conv_channels=[4, 4, 8])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return ClassifierObjective(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2),
    )


def make_batch(n: int = 8) -> Batch:
    torch.manual_seed(1)
    labels = torch.stack([torch.randint(0, c, (n,)) for c in CARDINALITIES], dim=1)
    return Batch(images=torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE), labels=labels)


def test_train_step_reports_total_and_an_error_rate_per_factor() -> None:
    step = build_objective().train_step(make_batch())

    assert set(step.metrics) == {
        "total",
        "error_rate/digit",
        "error_rate/fg",
        "error_rate/bg",
    }
    assert step.batch_size == 8
    for name in ("digit", "fg", "bg"):
        assert 0.0 <= float(step.metrics[f"error_rate/{name}"]) <= 1.0


def test_forward_has_one_head_per_factor() -> None:
    objective = build_objective()
    batch = make_batch()
    assert batch.images is not None

    logits = objective.model(batch.images)

    assert [head.shape for head in logits] == [(8, c) for c in CARDINALITIES]
    assert objective.model.predict(batch.images).shape == (8, len(CARDINALITIES))


def test_labels_missing_a_factor_are_rejected() -> None:
    batch = make_batch()
    assert batch.labels is not None
    digits_only = Batch(images=batch.images, labels=batch.labels[:, :1])

    with pytest.raises(ValueError, match=r"dataset\.labels"):
        build_objective().train_step(digits_only)


def test_error_rate_agrees_with_the_logits() -> None:
    objective = build_objective()
    batch = make_batch()
    assert batch.images is not None and batch.labels is not None

    step = objective.val_step(batch)

    objective.model.eval()
    predictions = objective.model.predict(batch.images)
    for i, name in enumerate(("digit", "fg", "bg")):
        expected = (predictions[:, i] != batch.labels[:, i]).float().mean()
        assert torch.isclose(step.metrics[f"error_rate/{name}"], expected)


def test_val_step_accumulates_per_class_counts_for_every_factor() -> None:
    objective = build_objective()
    batch = make_batch(16)
    assert batch.labels is not None

    objective.val_step(batch)

    for factor, cardinality in enumerate(CARDINALITIES):
        seen = objective.val_accuracy[factor]._seen
        assert seen.sum() == 16
        for c in range(cardinality):
            assert seen[c] == (batch.labels[:, factor] == c).sum()


def test_train_step_reduces_the_loss() -> None:
    objective = build_objective()
    batch = make_batch()

    first = float(objective.train_step(batch).metrics["total"])
    for _ in range(20):
        objective.train_step(batch)
    last = float(objective.train_step(batch).metrics["total"])

    assert last < first


def test_epoch_end_resets_counts_and_advances_the_scheduler() -> None:
    objective = build_objective()
    objective.val_step(make_batch())
    before = objective.lr_scheduler.get_last_lr()[0]

    objective.on_epoch_end()

    assert all(accuracy._seen.sum() == 0 for accuracy in objective.val_accuracy)
    assert objective.lr_scheduler.get_last_lr()[0] != before
    assert objective.extra_train_state() == {"epoch": 1}


def test_epoch_counter_survives_a_resume() -> None:
    objective = build_objective()
    objective.load_extra_train_state({"epoch": 7})

    assert objective.extra_train_state() == {"epoch": 7}


def test_checkpoint_round_trips_through_its_config() -> None:
    objective = build_objective()
    objective.model.eval()
    images = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "judge.pt"
        objective.save_checkpoint(path)
        restored = load_classifier_from_path(path)

    restored.eval()
    assert restored.config == objective.model.config
    with torch.no_grad():
        for mine, theirs in zip(restored(images), objective.model(images), strict=True):
            assert torch.allclose(mine, theirs)


def test_per_class_accuracy_marks_unseen_classes() -> None:
    accuracy = PerClassAccuracy(10)
    accuracy.update(torch.tensor([1, 1, 2]), torch.tensor([1, 2, 2]))

    per_class = accuracy.per_class
    assert accuracy.overall == pytest.approx(2 / 3)
    assert per_class[1] == 1.0
    assert per_class[2] == 0.5
    assert torch.isnan(per_class[0])
