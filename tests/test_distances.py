"""The set-metric maths against torch-fidelity 0.4.0, which it is copied from.

Random features stand in for network outputs, so no weights are needed.
"""

import pytest
import torch
from torch_fidelity import KEY_METRIC_FID, KEY_METRIC_PRECISION, KEY_METRIC_RECALL
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, kid_features_to_metric
from torch_fidelity.metric_prc import prc_features_to_metric

from evaluation import distances
from evaluation.distances import (
    cmmd,
    frechet_distance,
    kernel_inception_distance,
    precision,
    recall,
)


def features(n: int, d: int, shift: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scale = 1 + shift * torch.rand(d, generator=generator)
    return torch.randn(n, d, generator=generator) * scale + shift


@pytest.fixture
def samples() -> torch.Tensor:
    return features(600, 32, 0.4, seed=0)


@pytest.fixture
def reference() -> torch.Tensor:
    return features(600, 32, 0.0, seed=1)


def test_frechet_distance_matches_torch_fidelity(
    samples: torch.Tensor, reference: torch.Tensor
) -> None:
    theirs = fid_statistics_to_metric(
        fid_features_to_statistics(samples.double()),
        fid_features_to_statistics(reference.double()),
        verbose=False,
    )[KEY_METRIC_FID]
    assert frechet_distance(samples, reference) == pytest.approx(theirs, rel=1e-9)


def test_kid_matches_torch_fidelity_on_one_subset_of_everything(
    samples: torch.Tensor, reference: torch.Tensor
) -> None:
    # One subset the size of the whole set is a permutation of it, and the unbiased
    # MMD does not depend on order: torch-fidelity's KID is then exactly ours.
    theirs = kid_features_to_metric(
        samples.double(),
        reference.double(),
        kid_subsets=1,
        kid_subset_size=samples.shape[0],
        verbose=False,
    )[KEY_METRIC_KID_MEAN]
    ours = kernel_inception_distance(samples, reference)
    assert ours == pytest.approx(theirs, rel=1e-9)


def test_precision_and_recall_match_torch_fidelity(
    samples: torch.Tensor, reference: torch.Tensor
) -> None:
    # torch-fidelity's convention: the first input is generated, the second real.
    theirs = prc_features_to_metric(samples.double(), reference.double(), verbose=False)
    assert precision(samples, reference) == pytest.approx(theirs[KEY_METRIC_PRECISION])
    assert recall(samples, reference) == pytest.approx(theirs[KEY_METRIC_RECALL])


def test_blocked_kernels_match_one_block(
    monkeypatch: pytest.MonkeyPatch, samples: torch.Tensor, reference: torch.Tensor
) -> None:
    whole = [
        f(samples, reference)
        for f in (kernel_inception_distance, cmmd, precision, recall)
    ]
    monkeypatch.setattr(distances, "KERNEL_BLOCK", 7)
    blocked = [
        f(samples, reference)
        for f in (kernel_inception_distance, cmmd, precision, recall)
    ]
    assert blocked == pytest.approx(whole, rel=1e-9)


def test_cmmd_is_zero_for_a_set_against_itself_and_grows_with_a_shift() -> None:
    x = torch.nn.functional.normalize(features(400, 16, 0.0, seed=2), dim=-1)
    y = torch.nn.functional.normalize(features(400, 16, 0.5, seed=3), dim=-1)
    assert cmmd(x, x) == pytest.approx(0.0, abs=1e-9)
    assert cmmd(y, x) > cmmd(x[:200], x[200:]) > 0
