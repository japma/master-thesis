"""The anchored VAE's claim is stronger than the supervised one's -- not just that a
factor is decodable from its block, but that it sits at named coordinates inside it --
so these check the tables themselves, that the anchored dimensions are pulled towards
their anchor and *only* towards it, and that a checkpoint comes back as the same model.
"""

import math
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from dataset_loaders.colour_mnist import BG_COLOURS, FG_COLOURS
from models.autoencoder import AnchoredVAE
from training.losses.anchored_vae import AnchoredVAELoss
from training.losses.anchors import anchor_table, anchor_tables, anchor_targets
from training.losses.base import anchor_kl_per_dim
from training.losses.vae import VAELoss
from training.objectives.anchored_vae import AnchoredVAEObjective
from training.objectives.base import Batch
from training.schedulers import BetaAnnealingScheduler
from utils.checkpoints import load_ae_from_path, save_autoencoder
from utils.config import (
    AnchorScheme,
    AutoencoderConfig,
    AutoencoderType,
    SupervisionConfig,
)

LATENT_DIM = 10
IMAGE_SIZE = 8
CARDINALITIES = [4, 6, 3]
NAMES = ["digit", "fg", "bg"]
ANCHOR_STD = 0.1

# digit on a head, colours anchored -- the primary variant.
MIXED = SupervisionConfig(
    dims=[2, 3, 3],
    cardinalities=CARDINALITIES,
    names=NAMES,
    anchors=[
        AnchorScheme.NONE,
        AnchorScheme.COLOUR_MNIST_FG,
        AnchorScheme.COLOUR_MNIST_BG,
    ],
    anchor_std=ANCHOR_STD,
)
# everything anchored, digits one-hot.
FULLY = SupervisionConfig(
    dims=[4, 3, 3],
    cardinalities=CARDINALITIES,
    names=NAMES,
    anchors=[
        AnchorScheme.ONEHOT,
        AnchorScheme.COLOUR_MNIST_FG,
        AnchorScheme.COLOUR_MNIST_BG,
    ],
    anchor_std=ANCHOR_STD,
)


def model_config(
    supervision: SupervisionConfig = MIXED, **overrides
) -> AutoencoderConfig:
    fields = {
        "model_type": AutoencoderType.ANCHORED,
        "latent_dim": LATENT_DIM,
        "num_blocks": 2,
        "base_channels": 8,
        "image_size": IMAGE_SIZE,
        "channels": 3,
        "supervision": supervision,
    }
    return AutoencoderConfig(**{**fields, **overrides})


def build_model(supervision: SupervisionConfig = MIXED) -> AnchoredVAE:
    torch.manual_seed(0)
    return AnchoredVAE(config=model_config(supervision))


def build_loss(
    supervision: SupervisionConfig = MIXED,
    gamma: float = 1.0,
    anchor_weight: float = 1.0,
) -> AnchoredVAELoss:
    return AnchoredVAELoss(
        VAELoss(lambda_perceptual=0.0, free_bits=0.0),
        supervision=supervision,
        latent_dim=LATENT_DIM,
        gamma=gamma,
        anchor_weight=anchor_weight,
    )


def batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.stack(
        [torch.randint(0, c, (n,), generator=generator) for c in CARDINALITIES],
        dim=1,
    )
    return images, labels


# --- anchor tables ---
def test_colour_anchors_are_the_dataset_palette() -> None:
    fg = anchor_table(AnchorScheme.COLOUR_MNIST_FG, 6, 3)
    bg = anchor_table(AnchorScheme.COLOUR_MNIST_BG, 3, 3)

    assert torch.allclose(fg[0], torch.tensor([1.0, 0.0, 0.0]))  # red
    assert torch.allclose(bg[0], torch.tensor([1.0, 1.0, 1.0]))  # white
    assert torch.allclose(bg[1], torch.zeros(3))  # black
    # grey is on the diagonal between them, which is what makes "half way" a coordinate
    assert torch.allclose(bg[2], torch.full((3,), 128 / 255))
    assert fg.shape == (len(FG_COLOURS), 3)
    assert bg.shape == (len(BG_COLOURS), 3)


def test_onehot_anchors_are_equidistant() -> None:
    table = anchor_table(AnchorScheme.ONEHOT, 4, 4)
    distances = torch.cdist(table, table)

    assert torch.allclose(table, torch.eye(4))
    assert torch.allclose(
        distances[~torch.eye(4, dtype=torch.bool)], torch.full((12,), math.sqrt(2))
    )


def test_a_scheme_rejects_a_block_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="needs a block of shape"):
        anchor_table(AnchorScheme.COLOUR_MNIST_FG, 6, 2)
    with pytest.raises(ValueError, match="needs a block of shape"):
        anchor_table(AnchorScheme.ONEHOT, 4, 3)


def test_targets_read_each_table_from_its_own_label_column() -> None:
    tables = anchor_tables(MIXED)
    labels = torch.tensor([[3, 0, 1], [1, 4, 0]])

    targets = anchor_targets(tables, MIXED.anchored_factors, labels)

    # fg=red then bg=black for the first row; fg=yellow then bg=white for the second
    assert targets.shape == (2, 6)
    assert torch.allclose(targets[0], torch.tensor([1.0, 0, 0, 0, 0, 0]))
    assert torch.allclose(targets[1], torch.tensor([1.0, 1, 0, 1, 1, 1]))


def test_anchored_dims_skip_the_head_block() -> None:
    assert MIXED.anchored_dims() == [2, 3, 4, 5, 6, 7]
    assert MIXED.head_factors == [0]
    assert FULLY.head_factors == []
    assert FULLY.anchored_dims() == list(range(10))


# --- model ---
def test_heads_are_built_only_for_unanchored_factors() -> None:
    assert len(build_model(MIXED).heads) == 1
    assert len(build_model(FULLY).heads) == 0


def test_forward_keeps_the_vae_interface() -> None:
    model = build_model()
    images, _ = batch()
    outputs = model(images)

    assert outputs.reconstructed.shape == images.shape
    assert outputs.mu.shape == (4, LATENT_DIM)
    assert model.encode(images).shape == (4, LATENT_DIM)
    assert [logits.shape for logits in outputs.logits] == [(4, CARDINALITIES[0])]


def test_the_surviving_head_reads_only_its_own_block() -> None:
    model = build_model()
    z = torch.randn(4, LATENT_DIM, requires_grad=True)
    model.classify(z)[0].sum().backward()

    assert z.grad is not None
    assert z.grad[:, :2].abs().sum() > 0
    assert torch.equal(z.grad[:, 2:], torch.zeros_like(z.grad[:, 2:]))


# --- loss ---
def test_anchor_kl_vanishes_when_the_posterior_is_the_conditional_prior() -> None:
    anchor = torch.tensor([[1.0, 0.0, 0.5]])
    at_prior = anchor_kl_per_dim(
        anchor, torch.full_like(anchor, 2 * math.log(ANCHOR_STD)), anchor, ANCHOR_STD
    )
    assert torch.allclose(at_prior, torch.zeros_like(at_prior), atol=1e-6)

    off = anchor_kl_per_dim(
        anchor + 1.0,
        torch.full_like(anchor, 2 * math.log(ANCHOR_STD)),
        anchor,
        ANCHOR_STD,
    )
    assert torch.allclose(off, torch.full_like(off, 0.5 / ANCHOR_STD**2))


def test_anchored_dimensions_are_left_out_of_the_standard_normal_kl() -> None:
    """The whole point of the conditional prior: an anchored dimension is pulled towards
    its anchor, not towards zero as well."""
    model = build_model()
    images, labels = batch()
    outputs = model(images)
    outputs.mu.retain_grad()

    build_loss()(images, outputs, labels).kl.backward()

    assert outputs.mu.grad is not None
    anchored = outputs.mu.grad[:, MIXED.anchored_dims()]
    assert torch.equal(anchored, torch.zeros_like(anchored))
    assert outputs.mu.grad[:, :2].abs().sum() > 0
    assert outputs.mu.grad[:, 8:].abs().sum() > 0


def test_reduces_to_the_masked_vae_loss_at_gamma_zero() -> None:
    model = build_model()
    images, labels = batch()
    outputs = model(images)

    vae_loss = VAELoss(beta=1.0, lambda_perceptual=0.0, free_bits=0.0)
    anchored = build_loss(gamma=0.0)

    free = torch.ones(LATENT_DIM, dtype=torch.bool)
    free[MIXED.anchored_dims()] = False
    assert torch.allclose(
        anchored(images, outputs, labels).total,
        vae_loss(images, outputs, kl_mask=free).total,
    )


def test_total_is_recon_plus_both_weighted_supervision_terms() -> None:
    model = build_model()
    images, labels = batch()
    outputs = model(images)

    loss = build_loss(gamma=2.0, anchor_weight=0.5)(images, outputs, labels)

    assert torch.allclose(torch.stack(loss.per_anchor).sum(), loss.anchor)
    assert torch.allclose(torch.stack(loss.per_head).sum(), loss.classification)
    assert torch.allclose(
        loss.total,
        loss.recon
        + loss.kl
        + loss.perceptual
        + 2.0 * (0.5 * loss.anchor + loss.classification),
    )


def test_a_fully_anchored_model_has_no_classification_term() -> None:
    model = build_model(FULLY)
    images, labels = batch()

    loss = build_loss(FULLY)(images, model(images), labels)

    assert loss.per_head == []
    assert loss.classification.item() == 0.0


def test_rejects_a_batch_with_the_wrong_number_of_factors() -> None:
    model = build_model()
    images, labels = batch()

    with pytest.raises(ValueError, match="label factors"):
        build_loss()(images, model(images), labels[:, :2])


# --- training ---
def build_objective(
    model: AnchoredVAE, supervision: SupervisionConfig = MIXED, gamma: float = 1.0
) -> AnchoredVAEObjective:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return AnchoredVAEObjective(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        loss_fn=build_loss(supervision, gamma=gamma),
        beta_scheduler=BetaAnnealingScheduler(1.0, 1.0, num_steps=0),
        gamma_scheduler=BetaAnnealingScheduler(gamma, gamma, num_steps=0),
        factor_names=NAMES,
    )


def test_training_moves_the_colour_blocks_onto_their_anchors() -> None:
    model = build_model()
    objective = build_objective(model)
    images, labels = batch(n=8)

    first = objective.train_step(Batch(images=images, labels=labels))
    for _ in range(30):
        last = objective.train_step(Batch(images=images, labels=labels))

    assert last.metrics["anchor"] < first.metrics["anchor"]

    metrics = objective.val_step(Batch(images=images, labels=labels)).metrics
    assert set(metrics) >= {
        "total",
        "anchor",
        "anchor_kl/fg",
        "anchor_kl/bg",
        "anchor_rmse/fg",
        "anchor_rmse/bg",
        "ce/digit",
        "acc/digit",
    }
    # the head factor has no anchor and the anchored ones have no accuracy
    assert "anchor_rmse/digit" not in metrics
    assert "acc/fg" not in metrics


def test_train_state_round_trips_both_schedulers() -> None:
    objective = build_objective(build_model())
    objective.beta_scheduler.current_step = 7
    objective.gamma_scheduler.current_step = 3

    restored = build_objective(build_model())
    restored.load_extra_train_state(objective.extra_train_state())

    assert restored.beta_scheduler.current_step == 7
    assert restored.gamma_scheduler.current_step == 3


# --- checkpoints ---
def test_checkpoint_round_trips_as_an_anchored_vae(tmp_path: Path) -> None:
    model = build_model()
    path = tmp_path / "anchored.pt"
    save_autoencoder(model, path)

    loaded = load_ae_from_path(path)
    assert isinstance(loaded, AnchoredVAE)
    assert loaded.supervision.anchor_schemes == MIXED.anchor_schemes

    images, _ = batch()
    model.eval()
    loaded.eval()
    assert torch.allclose(model.encode(images), loaded.encode(images))


# --- config ---
def test_anchored_model_type_needs_at_least_one_anchor() -> None:
    with pytest.raises(ValidationError, match="needs at least one factor"):
        model_config(
            SupervisionConfig(dims=[2, 3, 3], cardinalities=CARDINALITIES, names=NAMES)
        )


def test_supervised_model_type_rejects_anchors() -> None:
    with pytest.raises(ValidationError, match="classification heads only"):
        model_config(MIXED, model_type=AutoencoderType.SUPERVISED)


def test_anchors_need_one_entry_per_factor() -> None:
    with pytest.raises(ValidationError, match="one entry per label factor"):
        SupervisionConfig(
            dims=[2, 3, 3],
            cardinalities=CARDINALITIES,
            anchors=[AnchorScheme.NONE, AnchorScheme.ONEHOT],
        )


def test_anchor_std_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="anchor_std must be positive"):
        SupervisionConfig(dims=[2], cardinalities=[4], anchor_std=0.0)
