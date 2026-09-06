"""The neural baselines exist to be compared against a circuit, so what matters is
that their densities are the ones they claim to be and that they go through the same
evaluation path a CSPN does.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from evaluation import run_generation_probe
from models.neural_baseline import (
    DeterministicBaseline,
    MixtureDensityBaseline,
    build_neural_baseline,
)
from utils.config import CSPNEncoderConfig, CSPNEncoderType, NeuralBaselineConfig

NUM_VARS = 6
CARDINALITIES = [10, 6, 3]
ENCODER = CSPNEncoderConfig(
    encoder_type=CSPNEncoderType.MULTI_CATEGORICAL, num_classes=CARDINALITIES
)


def build(kind: str, **overrides):
    torch.manual_seed(0)
    return build_neural_baseline(
        NeuralBaselineConfig(
            model_type=kind,
            num_vars=NUM_VARS,
            h_dims=[32],
            encoder_config=ENCODER,
            **overrides,
        )
    )


def random_labels(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randint(0, c, (n,), generator=generator) for c in CARDINALITIES],
        dim=1,
    )


def test_deterministic_density_is_a_fixed_width_gaussian() -> None:
    model = build("deterministic", fixed_std=0.7)
    assert isinstance(model, DeterministicBaseline)
    labels, z = random_labels(16), torch.randn(16, NUM_VARS)

    expected = Normal(model._mean(labels), 0.7).log_prob(z).sum(-1)
    assert torch.allclose(model(z, labels), expected, atol=1e-5)


def test_mixture_density_matches_a_gaussian_mixture() -> None:
    model = build("mixture", num_components=5)
    assert isinstance(model, MixtureDensityBaseline)
    labels, z = random_labels(16), torch.randn(16, NUM_VARS)

    logits, means, log_stds = model._params(labels)
    expected = MixtureSameFamily(
        Categorical(logits=logits), Independent(Normal(means, log_stds.exp()), 1)
    ).log_prob(z)
    assert torch.allclose(model(z, labels), expected, atol=1e-5)


def test_mixture_density_integrates_to_one() -> None:
    torch.manual_seed(0)
    model = build_neural_baseline(
        NeuralBaselineConfig(
            model_type="mixture",
            num_vars=1,
            h_dims=[32],
            encoder_config=ENCODER,
            num_components=5,
        )
    )
    grid = torch.linspace(-40.0, 40.0, 200001).unsqueeze(1)
    labels = random_labels(1).expand(grid.shape[0], -1)
    mass = torch.trapz(model(grid, labels).exp(), grid.squeeze(1))
    assert mass.item() == pytest.approx(1.0, abs=1e-3)


def test_mixture_samples_follow_its_own_density() -> None:
    torch.manual_seed(0)
    model = build_neural_baseline(
        NeuralBaselineConfig(
            model_type="mixture",
            num_vars=1,
            h_dims=[32],
            encoder_config=ENCODER,
            num_components=5,
        )
    )
    label = random_labels(1)

    torch.manual_seed(1)
    draws = model.sample(label.expand(20000, -1)).squeeze(1)
    edges = torch.linspace(-6.0, 6.0, 61)
    empirical = torch.histogram(draws, bins=edges).hist / 20000
    centres = 0.5 * (edges[:-1] + edges[1:])
    modelled = model(centres.unsqueeze(1), label.expand(60, -1)).exp() * (
        edges[1] - edges[0]
    )
    assert 0.5 * (empirical - modelled).abs().sum().item() < 0.05


def test_std_head_keeps_a_gradient_past_the_bounds() -> None:
    """A clamp has exactly zero gradient outside [min_std, max_std], so a component
    that collapses can never recover. These raw values are all well past the bounds a
    clamp would have used (log 0.001 = -6.9, log 2.0 = 0.69); the squash still pulls
    back. Only float32 sigmoid saturation, around |raw| > 30, kills it.
    """
    model = build("mixture", num_components=4, min_std=0.001, max_std=2.0)
    labels, z = random_labels(8), torch.randn(8, NUM_VARS)

    for extreme in (-10.0, 7.0):
        model.zero_grad()
        with torch.no_grad():
            model.log_std_head.bias.fill_(extreme)
        (-model(z, labels).mean()).backward()
        assert model.log_std_head.bias.grad.abs().sum().item() > 0.0

    for extreme in (-30.0, 30.0):
        with torch.no_grad():
            model.log_std_head.bias.fill_(extreme)
        stds = model._params(labels)[2].exp()
        assert stds.min().item() >= model.min_std - 1e-9
        assert stds.max().item() <= model.max_std + 1e-9


def test_std_correction_scales_only_the_stochastic_baseline() -> None:
    labels = random_labels(1).expand(512, -1)

    mixture = build("mixture", num_components=4)
    torch.manual_seed(2)
    wide = mixture.sample(labels, std_correction=1.0)
    torch.manual_seed(2)
    narrow = mixture.sample(labels, std_correction=0.25)
    assert narrow.std(0).mean().item() < wide.std(0).mean().item()

    deterministic = build("deterministic")
    assert torch.equal(
        deterministic.sample(labels),
        deterministic.sample(labels, std_correction=0.25),
    )
    # A point predictor emits one latent per label, by construction.
    assert len(torch.unique(deterministic.sample(labels), dim=0)) == 1


def test_both_baselines_train() -> None:
    torch.manual_seed(0)
    labels = random_labels(512)
    z = 0.3 * torch.randn(512, NUM_VARS)
    z[:, 0] += labels[:, 2].float()

    for kind, kwargs in (("deterministic", {}), ("mixture", {"num_components": 8})):
        model = build(kind, **kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        first = None
        for _ in range(200):
            optimizer.zero_grad()
            loss = -model(z, labels).mean()
            loss.backward()
            optimizer.step()
            first = loss.item() if first is None else first
        assert math.isfinite(loss.item())
        assert loss.item() < first


class StubDecoder(nn.Module):
    """Turns a latent into an image the colour probe can measure."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Linear(NUM_VARS, 3 * 28 * 28)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, 3, 28, 28).sigmoid()


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [("deterministic", {}), ("mixture", {"num_components": 4})],
)
def test_baselines_run_through_the_generation_probe(kind: str, kwargs: dict) -> None:
    """The whole point of the baselines: probed exactly like a circuit."""
    probe = run_generation_probe(
        build(kind, **kwargs),
        StubDecoder(),
        torch.device("cpu"),
        samples_per_combination=2,
        std_correction=0.8,
    )
    assert probe.bg_accuracy.shape == (10, 6, 3)
    assert probe.fg_accuracy.shape == (10, 6, 3)
    assert (probe.contrast_table >= 0).all()
