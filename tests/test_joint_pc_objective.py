"""JointPCObjective against the contract run_training_loop relies on."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from models.cspn.joint_pc import JointPC
from training.objectives.base import Batch
from training.objectives.joint_pc import JointPCObjective
from utils.checkpoints import load_joint_pc_from_path
from utils.config import JointPCConfig
from utils.reproducibility import seed_everything

NUM_LATENTS = 6
CARDINALITIES = [10, 6, 3]
IMAGE_SHAPE = (3, 8, 8)


class StubAutoencoder(nn.Module):
    """Deterministic stand-in for the pretrained VAE: a fixed linear encode/decode."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(int(torch.tensor(IMAGE_SHAPE).prod()), NUM_LATENTS)
        self.decoder = nn.Linear(NUM_LATENTS, int(torch.tensor(IMAGE_SHAPE).prod()))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, *IMAGE_SHAPE)


def build_objective() -> JointPCObjective:
    seed_everything(0)
    model = JointPC(
        config=JointPCConfig(
            num_latents=NUM_LATENTS,
            label_cardinalities=CARDINALITIES,
            num_repetitions=2,
            num_input_distributions=4,
            num_sums=4,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return JointPCObjective(
        model=model,
        autoencoder=StubAutoencoder(),
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2),
    )


def make_batch(n: int = 8) -> Batch:
    torch.manual_seed(1)
    labels = torch.stack(
        [torch.randint(0, c, (n,)) for c in CARDINALITIES], dim=1
    )
    return Batch(images=torch.rand(n, *IMAGE_SHAPE), labels=labels)


def test_train_step_reports_a_finite_loss_and_updates_weights() -> None:
    objective = build_objective()
    before = [p.detach().clone() for p in objective.model.parameters()]

    output = objective.train_step(make_batch())

    assert output.batch_size == 8
    assert torch.isfinite(output.metrics["total"])
    after = list(objective.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_val_step_leaves_weights_alone() -> None:
    objective = build_objective()
    before = [p.detach().clone() for p in objective.model.parameters()]

    output = objective.val_step(make_batch())

    assert torch.isfinite(output.metrics["total"])
    after = list(objective.model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_sample_decodes_to_images() -> None:
    objective = build_objective()
    probe = torch.zeros(4, len(CARDINALITIES), dtype=torch.long)
    probe[:, 0] = torch.arange(4)

    images = objective.sample(probe)

    assert images.shape == (4, *IMAGE_SHAPE)
    assert torch.isfinite(images).all()


def test_checkpoint_is_written_and_reloads() -> None:
    objective = build_objective()
    objective.train_step(make_batch())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "joint_pc.pt"
        objective.save_checkpoint(path)
        restored = load_joint_pc_from_path(path)

    z, labels = torch.randn(4, NUM_LATENTS), make_batch(4).labels
    assert labels is not None
    assert torch.allclose(restored(z, labels), objective.model(z, labels), atol=1e-5)


def test_epoch_end_advances_the_scheduler() -> None:
    objective = build_objective()
    before = objective.optimizer.param_groups[0]["lr"]
    objective.on_epoch_end()
    assert objective.optimizer.param_groups[0]["lr"] != before
