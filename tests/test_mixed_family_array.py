"""MixedFamilyArray: Normal latents and categorical labels in one circuit.

Fits a small joint p(z, y) whose ground truth is known -- each label picks the mean
of a latent -- and checks the three queries the latent+label PC exists for: the
label marginal p(y), conditional sampling p(z | y), and the posterior p(y | z).
"""

import itertools

import pytest
import torch

from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import (
    CategoricalArray,
    MixedFamilyArray,
    NormalArray,
)
from models.cspn.psinet.graph import random_binary_trees
from utils.reproducibility import seed_everything

NUM_LATENT = 2
K0, K1 = 3, 2
NUM_VAR = NUM_LATENT + 2
LATENT_IDX = [0, 1]
LABEL_IDX = [2, 3]
NOISE = 0.15
# p(y1); y0 is uniform.
P_Y1 = 0.3


def latent_means(y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    return torch.stack([y0.float() - 1.0, 2.0 * y1.float() - 1.0], dim=1)


def make_data(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    y0 = torch.randint(0, K0, (n,), generator=generator)
    y1 = (torch.rand(n, generator=generator) < P_Y1).long()
    z = latent_means(y0, y1) + NOISE * torch.randn(n, NUM_LATENT, generator=generator)
    return torch.cat([z, y0.unsqueeze(1).float(), y1.unsqueeze(1).float()], dim=1)


def build_net(seed: int = 0) -> EinsumNetwork:
    seed_everything(seed)
    graph = random_binary_trees(num_var=NUM_VAR, depth=2, num_repetitions=4)
    args = Args(
        num_var=NUM_VAR,
        num_dims=1,
        num_input_distributions=8,
        num_sums=8,
        num_classes=1,
        exponential_family=MixedFamilyArray,
        exponential_family_args={
            "blocks": [
                (NUM_LATENT, NormalArray, {"min_var": 1e-4, "max_var": 4.0}),
                (1, CategoricalArray, {"K": K0}),
                (1, CategoricalArray, {"K": K1}),
            ]
        },
    )
    net = EinsumNetwork(graph=graph, param_nn=None, args=args)
    net.initialize()
    return net


@pytest.fixture(scope="module")
def trained_net() -> EinsumNetwork:
    net = build_net()
    data = make_data(2048)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        loss = -net.forward(data).mean()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    return net


def label_marginal(net: EinsumNetwork, var: int, num_classes: int) -> torch.Tensor:
    other = [i for i in range(NUM_VAR) if i != var]
    x = torch.zeros(num_classes, NUM_VAR)
    x[:, var] = torch.arange(num_classes).float()
    net.set_marginalization_idx(other)
    try:
        with torch.no_grad():
            return net.forward(x).squeeze(-1).exp()
    finally:
        net.set_marginalization_idx(None)


def test_label_marginals_match_the_data(trained_net: EinsumNetwork) -> None:
    p_y0 = label_marginal(trained_net, LABEL_IDX[0], K0)
    p_y1 = label_marginal(trained_net, LABEL_IDX[1], K1)

    assert p_y0.sum().item() == pytest.approx(1.0, abs=1e-3)
    assert p_y1.sum().item() == pytest.approx(1.0, abs=1e-3)
    assert p_y0.tolist() == pytest.approx([1 / K0] * K0, abs=0.05)
    assert p_y1[1].item() == pytest.approx(P_Y1, abs=0.05)


def test_conditional_samples_land_on_the_right_latent_mode(
    trained_net: EinsumNetwork,
) -> None:
    torch.manual_seed(3)
    for y0, y1 in itertools.product(range(K0), range(K1)):
        evidence = torch.zeros(256, NUM_VAR)
        evidence[:, LABEL_IDX[0]] = y0
        evidence[:, LABEL_IDX[1]] = y1

        trained_net.set_marginalization_idx(LATENT_IDX)
        try:
            samples = trained_net.sample(x=evidence)
        finally:
            trained_net.set_marginalization_idx(None)

        assert samples is not None
        # The labels were observed, so they must come back untouched.
        assert samples[:, LABEL_IDX[0]].eq(y0).all()
        assert samples[:, LABEL_IDX[1]].eq(y1).all()

        expected = latent_means(torch.tensor([y0]), torch.tensor([y1]))[0]
        drawn = samples[:, LATENT_IDX]
        assert drawn.mean(0).tolist() == pytest.approx(expected.tolist(), abs=0.15)
        assert drawn.std(0).max().item() < 0.4


def test_posterior_recovers_the_label_from_the_latent(
    trained_net: EinsumNetwork,
) -> None:
    data = make_data(256, seed=7)
    z, true_y0 = data[:, LATENT_IDX], data[:, LABEL_IDX[0]].long()

    log_joint = []
    for y0 in range(K0):
        x = data.clone()
        x[:, LABEL_IDX[0]] = y0
        trained_net.set_marginalization_idx([LABEL_IDX[1]])
        try:
            with torch.no_grad():
                log_joint.append(trained_net.forward(x).squeeze(-1))
        finally:
            trained_net.set_marginalization_idx(None)

    predicted = torch.stack(log_joint, dim=1).argmax(dim=1)
    assert (predicted == true_y0).float().mean().item() > 0.95
    assert z.shape == (256, NUM_LATENT)


def test_mpe_is_a_valid_assignment(trained_net: EinsumNetwork) -> None:
    trained_net.set_marginalization_idx(list(range(NUM_VAR)))
    try:
        mpe = trained_net.mpe(x=torch.zeros(1, NUM_VAR))
    finally:
        trained_net.set_marginalization_idx(None)

    assert mpe is not None
    assert mpe.shape == (1, NUM_VAR)
    assert 0 <= mpe[0, LABEL_IDX[0]].item() < K0
    assert 0 <= mpe[0, LABEL_IDX[1]].item() < K1
    assert mpe[0, LABEL_IDX[0]].item() == int(mpe[0, LABEL_IDX[0]].item())


def test_blocks_must_tile_the_variables() -> None:
    with pytest.raises(AssertionError, match="blocks cover"):
        MixedFamilyArray(
            NUM_VAR,
            1,
            (4, 2),
            [(1, NormalArray, {}), (1, CategoricalArray, {"K": K0})],
        )
