"""JointPC at colour-MNIST's shape: 16 latents plus [digit, fg, bg].

Trains against a synthetic latent space where the background colour dictates one
latent dimension, so the queries have a known right answer.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from models.cspn.joint_pc import JointPC
from utils.checkpoints import load_joint_pc_from_path, save_joint_pc
from utils.config import JointPCConfig
from utils.reproducibility import seed_everything

NUM_LATENTS = 6
CARDINALITIES = [10, 6, 3]
# Latent 0 encodes the background colour; the rest are noise.
BG_MEANS = torch.tensor([-2.0, 0.0, 2.0])


def make_batch(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.stack(
        [
            torch.randint(0, c, (n,), generator=generator)
            for c in CARDINALITIES
        ],
        dim=1,
    )
    z = 0.2 * torch.randn(n, NUM_LATENTS, generator=generator)
    z[:, 0] += BG_MEANS[labels[:, 2]]
    return z, labels


def build(normalize: bool = False) -> JointPC:
    seed_everything(0)
    return JointPC(
        config=JointPCConfig(
            num_latents=NUM_LATENTS,
            label_cardinalities=CARDINALITIES,
            num_repetitions=4,
            num_input_distributions=8,
            num_sums=8,
            normalize_latents=normalize,
        )
    )


@pytest.fixture(scope="module")
def trained() -> JointPC:
    model = build()
    z, labels = make_batch(1024, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(250):
        optimizer.zero_grad()
        loss = -model(z, labels).mean()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    return model


def test_variable_layout() -> None:
    model = build()
    assert model.num_vars == NUM_LATENTS + len(CARDINALITIES)
    assert model.latent_idx == list(range(NUM_LATENTS))
    assert model.label_idx == [NUM_LATENTS, NUM_LATENTS + 1, NUM_LATENTS + 2]


def test_forward_is_a_normalized_joint_density(trained: JointPC) -> None:
    z, labels = make_batch(16, seed=1)
    log_prob = trained(z, labels)
    assert log_prob.shape == (16,)
    assert torch.isfinite(log_prob).all()
    # In-distribution points must beat latents nowhere near any mode.
    far = torch.full_like(z, 50.0)
    assert (log_prob > trained(far, labels)).all()


def test_label_marginal_sums_to_one(trained: JointPC) -> None:
    labels = torch.tensor(
        [[d, f, b] for d in range(10) for f in range(6) for b in range(3)]
    )
    total = trained.label_log_marginal(labels).exp().sum().item()
    assert total == pytest.approx(1.0, abs=1e-3)


def test_conditional_sampling_follows_the_background_label(trained: JointPC) -> None:
    torch.manual_seed(2)
    for bg in range(3):
        labels = torch.tensor([[3, 1, bg]] * 128)
        z = trained.sample(labels)
        assert z.shape == (128, NUM_LATENTS)
        assert z[:, 0].mean().item() == pytest.approx(BG_MEANS[bg].item(), abs=0.4)


def test_sample_partial_labels_completes_the_unspecified_factors(
    trained: JointPC,
) -> None:
    torch.manual_seed(4)
    z, labels = trained.sample_partial_labels({0: 7}, batch_size=256)

    assert z.shape == (256, NUM_LATENTS)
    assert labels.shape == (256, 3)
    assert labels[:, 0].eq(7).all()
    for factor, cardinality in enumerate(CARDINALITIES):
        assert labels[:, factor].min() >= 0
        assert labels[:, factor].max() < cardinality
    # The unspecified background was drawn from p(bg | digit=7), so the latent it
    # controls must track the label that came back with it.
    for bg in range(3):
        rows = labels[:, 2] == bg
        if rows.sum() < 10:
            continue
        assert z[rows, 0].mean().item() == pytest.approx(BG_MEANS[bg].item(), abs=0.5)


def test_conditional_partial_holds_known_latents(trained: JointPC) -> None:
    labels = torch.tensor([[3, 1, 2]] * 8)
    z = trained.sample_conditional_partial(labels, known={1: 1.5, 4: -2.0})
    assert z[:, 1].eq(1.5).all()
    assert z[:, 4].eq(-2.0).all()


def test_log_marginal_ignores_the_dims_it_marginalizes(trained: JointPC) -> None:
    z, labels = make_batch(16, seed=5)
    observed = [0, 1]

    marginal = trained.log_marginal(z, labels, observed)
    scrambled = z.clone()
    scrambled[:, 2:] = 100.0
    assert torch.allclose(
        marginal, trained.log_marginal(scrambled, labels, observed), atol=1e-5
    )

    # Observing everything is the joint itself.
    assert torch.allclose(
        trained.log_marginal(z, labels, trained.latent_idx),
        trained(z, labels),
        atol=1e-5,
    )

    with pytest.raises(ValueError, match="out of range"):
        trained.log_marginal(z, labels, [NUM_LATENTS])


def test_normalization_shifts_the_density_by_the_jacobian() -> None:
    z, labels = make_batch(32, seed=6)
    plain = build()
    scaled = build(normalize=True)
    scaled.load_state_dict(plain.state_dict(), strict=False)

    std = torch.full((NUM_LATENTS,), 2.0)
    scaled.set_latent_stats(torch.zeros(NUM_LATENTS), std)

    expected = plain(z / std, labels) - std.log().sum()
    assert torch.allclose(scaled(z, labels), expected, atol=1e-4)


def test_checkpoint_round_trip(trained: JointPC) -> None:
    z, labels = make_batch(8, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "joint_pc.pt"
        save_joint_pc(trained, path)
        restored = load_joint_pc_from_path(path)

    assert restored.get_config() == trained.get_config()
    assert torch.allclose(restored(z, labels), trained(z, labels), atol=1e-5)
