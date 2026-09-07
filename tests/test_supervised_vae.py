"""The supervised VAE's whole claim is *where* label information ends up, so these
check the routing (a head sees only its own block), that the extra term is exactly a
cross-entropy on top of an unchanged VAE loss, and that a checkpoint comes back as the
same model -- the heads are new weights a plain VariationalAutoencoder cannot hold.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from models.autoencoder import SupervisedVAE
from training.losses.supervised_vae import SupervisedVAELoss
from training.losses.vae import VAELoss
from training.objectives.base import Batch
from training.objectives.supervised_vae import SupervisedVAEObjective
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import load_ae_from_path, save_autoencoder
from utils.config import (
    AERunConfig,
    AutoencoderConfig,
    AutoencoderType,
    SupervisionConfig,
)

LATENT_DIM = 10
DIMS = [4, 3, 2]
CARDINALITIES = [10, 6, 3]
IMAGE_SIZE = 8


def model_config(**overrides) -> AutoencoderConfig:
    fields = {
        "model_type": AutoencoderType.SUPERVISED,
        "latent_dim": LATENT_DIM,
        "num_blocks": 2,
        "base_channels": 8,
        "image_size": IMAGE_SIZE,
        "channels": 3,
        "supervision": SupervisionConfig(
            dims=DIMS, cardinalities=CARDINALITIES, names=["digit", "fg", "bg"]
        ),
    }
    return AutoencoderConfig(**{**fields, **overrides})


def build_model() -> SupervisedVAE:
    torch.manual_seed(0)
    return SupervisedVAE(config=model_config())


def batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.stack(
        [torch.randint(0, c, (n,), generator=generator) for c in CARDINALITIES],
        dim=1,
    )
    return images, labels


# --- routing ---
def test_head_reads_only_its_own_latent_block() -> None:
    model = build_model()
    bounds = [0, 4, 7, 9]

    for factor in range(len(DIMS)):
        z = torch.randn(4, LATENT_DIM, requires_grad=True)
        model.classify(z)[factor].sum().backward()
        assert z.grad is not None

        own = z.grad[:, bounds[factor] : bounds[factor + 1]]
        assert own.abs().sum() > 0
        outside = torch.cat(
            [z.grad[:, : bounds[factor]], z.grad[:, bounds[factor + 1] :]], dim=1
        )
        assert torch.equal(outside, torch.zeros_like(outside))


def test_dimensions_beyond_the_last_block_stay_free() -> None:
    model = build_model()
    z = torch.randn(4, LATENT_DIM, requires_grad=True)
    torch.stack([logits.sum() for logits in model.classify(z)]).sum().backward()

    assert z.grad is not None
    assert torch.equal(z.grad[:, 9:], torch.zeros_like(z.grad[:, 9:]))


def test_forward_keeps_the_vae_interface() -> None:
    model = build_model()
    images, _ = batch()
    outputs = model(images)

    assert outputs.reconstructed.shape == images.shape
    assert outputs.mu.shape == (4, LATENT_DIM)
    assert model.encode(images).shape == (4, LATENT_DIM)
    assert [logits.shape for logits in outputs.logits] == [
        (4, c) for c in CARDINALITIES
    ]


# --- loss ---
def test_reduces_to_the_vae_loss_at_gamma_zero() -> None:
    model = build_model()
    images, labels = batch()
    outputs = model(images)

    vae_loss = VAELoss(beta=1.0, lambda_perceptual=0.0)
    supervised = SupervisedVAELoss(vae_loss, gamma=0.0)

    assert torch.allclose(
        supervised(images, outputs, labels).total, vae_loss(images, outputs).total
    )


def test_classification_term_is_the_summed_cross_entropy() -> None:
    model = build_model()
    images, labels = batch()
    outputs = model(images)

    loss = SupervisedVAELoss(VAELoss(lambda_perceptual=0.0), gamma=2.0)(
        images, outputs, labels
    )

    expected = torch.stack(
        [
            F.cross_entropy(logits, labels[:, i])
            for i, logits in enumerate(outputs.logits)
        ]
    ).sum()
    assert torch.allclose(loss.classification, expected)
    assert torch.allclose(
        loss.total, loss.recon + loss.kl + loss.perceptual + 2.0 * expected
    )


def test_rejects_a_batch_with_the_wrong_number_of_factors() -> None:
    model = build_model()
    images, labels = batch()
    outputs = model(images)

    with pytest.raises(ValueError, match="label factors"):
        SupervisedVAELoss(VAELoss(lambda_perceptual=0.0))(
            images, outputs, labels[:, :2]
        )


# --- training ---
def build_objective(model: SupervisedVAE, gamma: float = 10.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return SupervisedVAEObjective(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        loss_fn=SupervisedVAELoss(VAELoss(lambda_perceptual=0.0), gamma=gamma),
        beta_scheduler=BetaAnnealingScheduler(1.0, 1.0, num_steps=0),
        gamma_scheduler=BetaAnnealingScheduler(gamma, gamma, num_steps=0),
        factor_names=["digit", "fg", "bg"],
    )


def test_training_drives_the_label_blocks_towards_their_factors() -> None:
    model = build_model()
    objective = build_objective(model)
    images, labels = batch(n=8)

    first = objective.train_step(Batch(images=images, labels=labels))
    for _ in range(30):
        last = objective.train_step(Batch(images=images, labels=labels))

    assert last.metrics["classification"] < first.metrics["classification"]
    assert set(objective.val_step(Batch(images=images, labels=labels)).metrics) >= {
        "total",
        "classification",
        "ce/digit",
        "acc/digit",
        "acc/fg",
        "acc/bg",
    }


def test_gamma_waits_out_the_kl_warmup() -> None:
    scheduler = BetaAnnealingScheduler(0.0, 4.0, num_steps=4, delay_steps=2)

    assert scheduler.beta == 0.0
    scheduler.step()
    scheduler.step()
    assert scheduler.beta == 0.0
    scheduler.step()
    assert scheduler.beta == 1.0
    for _ in range(10):
        scheduler.step()
    assert scheduler.beta == 4.0


def test_train_state_round_trips_both_schedulers() -> None:
    objective = build_objective(build_model())
    objective.beta_scheduler.current_step = 7
    objective.gamma_scheduler.current_step = 3

    restored = build_objective(build_model())
    restored.load_extra_train_state(objective.extra_train_state())

    assert restored.beta_scheduler.current_step == 7
    assert restored.gamma_scheduler.current_step == 3


# --- checkpoints ---
def test_checkpoint_round_trips_as_a_supervised_vae(tmp_path: Path) -> None:
    model = build_model()
    path = tmp_path / "supervised.pt"
    save_autoencoder(model, path)

    loaded = load_ae_from_path(path)
    assert isinstance(loaded, SupervisedVAE)

    images, _ = batch()
    model.eval()
    loaded.eval()
    for original, restored in zip(
        model.classify(model.encode(images)),
        loaded.classify(loaded.encode(images)),
        strict=True,
    ):
        assert torch.allclose(original, restored)


# --- config ---
def test_supervision_needs_matching_dims_and_cardinalities() -> None:
    with pytest.raises(ValidationError, match="same label factors"):
        SupervisionConfig(dims=[4, 3], cardinalities=[10, 6, 3])


def test_supervision_needs_a_name_per_factor() -> None:
    with pytest.raises(ValidationError, match="one entry per label factor"):
        SupervisionConfig(dims=[4, 3], cardinalities=[10, 6], names=["digit"])


def test_blocks_must_fit_in_the_latent() -> None:
    with pytest.raises(ValidationError, match="latent_dim"):
        model_config(latent_dim=5)


def test_supervised_model_type_requires_a_supervision_block() -> None:
    with pytest.raises(ValidationError, match="needs a `supervision` block"):
        model_config(supervision=None)


def test_supervision_without_the_supervised_model_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="only read by model_type=supervised"):
        model_config(model_type=AutoencoderType.VARIATIONAL)


def run_config(model_type: AutoencoderType, gamma_end: float, **model_overrides) -> None:
    AERunConfig.model_validate(
        {
            "type": "ae",
            "dataset": {
                "name": "colour_mnist_uniform",
                "channels": 3,
                "height": IMAGE_SIZE,
                "width": IMAGE_SIZE,
                "num_classes": 10,
            },
            "model": model_config(
                model_type=model_type, **model_overrides
            ).model_dump(),
            "training": {
                "vae_type": "beta",
                "epochs": 1,
                "learning_rate": 1e-3,
                "batch_size": 8,
                "beta": 1.0,
                "beta_start": 0.0,
                "beta_end": 1.0,
                "kl_warmup_epochs": 1,
                "gamma_end": gamma_end,
            },
        }
    )


def test_supervised_run_needs_a_positive_gamma() -> None:
    with pytest.raises(ValidationError, match="gamma_end is 0"):
        run_config(AutoencoderType.SUPERVISED, gamma_end=0.0)


def test_gamma_without_supervision_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no `supervision` block"):
        run_config(AutoencoderType.VARIATIONAL, gamma_end=1.0, supervision=None)


def test_supervised_run_config_validates() -> None:
    run_config(AutoencoderType.SUPERVISED, gamma_end=10.0)
