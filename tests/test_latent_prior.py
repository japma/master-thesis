"""The unconditional latent baselines: N(0, I) and the ex-post Gaussian mixture."""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.mixture import GaussianMixture

from models.latent_prior import GaussianMixturePrior, StandardNormalPrior
from scripts.fit_gmm import to_prior
from utils.checkpoints import load_gmm_from_path, save_gmm


def two_blobs(n: int = 4000) -> np.ndarray:
    generator = np.random.default_rng(0)
    left = generator.normal([-4.0, 0.0], [0.5, 1.0], size=(n // 4, 2))
    right = generator.normal([4.0, 1.0], [1.0, 0.3], size=(3 * n // 4, 2))
    return np.concatenate([left, right])


def fitted() -> GaussianMixturePrior:
    mixture = GaussianMixture(n_components=2, random_state=0).fit(two_blobs())
    return to_prior(mixture)


def test_the_prior_ignores_the_labels_except_for_their_count() -> None:
    torch.manual_seed(0)
    z = StandardNormalPrior(8).sample(torch.zeros(5000, 40), std_correction=0.5)

    assert z.shape == (5000, 8)
    assert z.std().item() == pytest.approx(0.5, abs=0.02)


def test_mixture_samples_reproduce_the_fitted_moments() -> None:
    torch.manual_seed(0)
    data = torch.from_numpy(two_blobs())

    z = fitted().sample(torch.zeros(20000, 3))

    assert torch.allclose(z.mean(0).float(), data.mean(0).float(), atol=0.1)
    assert torch.allclose(z.std(0).float(), data.std(0).float(), atol=0.1)
    assert (z[:, 0] < 0).float().mean().item() == pytest.approx(0.25, abs=0.02)


def test_std_correction_shrinks_each_component_not_the_mixture() -> None:
    torch.manual_seed(0)
    z = fitted().sample(torch.zeros(20000, 3), std_correction=0.0)

    assert z.unique(dim=0).shape[0] == 2


def test_a_mixture_survives_a_checkpoint(tmp_path: Path) -> None:
    prior = fitted()
    path = tmp_path / "gmm.pt"

    save_gmm(prior, path, source_artifact="vae:v1")
    loaded = load_gmm_from_path(path)

    assert loaded.get_config() == prior.get_config()
    for name, buffer in prior.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], buffer)
