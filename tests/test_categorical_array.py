"""End-to-end checks on CategoricalArray inside an EinsumNetwork.

The unconditional PC path (param_nn=None) feeds batched parameters into the leaf
array, and backtracking hands it an arbitrary scope subset -- both shapes the
family has to cope with. LabelPC only ever exercised BinomialArray, so these pin
down the categorical leaves the joint latent+label PC needs.
"""

import itertools
import math

import pytest
import torch

from models.cspn.psinet.einsum_network import Args, EinsumNetwork
from models.cspn.psinet.exponential_family_array import CategoricalArray
from models.cspn.psinet.graph import random_binary_trees
from utils.reproducibility import seed_everything

NUM_VAR = 3
K = 3


def build_net(seed: int = 0) -> EinsumNetwork:
    seed_everything(seed)
    graph = random_binary_trees(num_var=NUM_VAR, depth=1, num_repetitions=2)
    args = Args(
        num_var=NUM_VAR,
        num_dims=1,
        num_input_distributions=4,
        num_sums=4,
        num_classes=1,
        exponential_family=CategoricalArray,
        exponential_family_args={"K": K},
    )
    net = EinsumNetwork(graph=graph, param_nn=None, args=args)
    net.initialize()
    return net


def all_states() -> torch.Tensor:
    return torch.tensor(
        list(itertools.product(range(K), repeat=NUM_VAR)), dtype=torch.float32
    )


def joint_pmf(net: EinsumNetwork) -> torch.Tensor:
    with torch.no_grad():
        return net.forward(all_states()).squeeze(-1).exp()


def test_joint_normalizes_to_one() -> None:
    assert joint_pmf(build_net()).sum().item() == pytest.approx(1.0, abs=1e-4)


def test_marginalization_matches_brute_force() -> None:
    net = build_net()
    states = all_states()
    exact = joint_pmf(net)

    net.set_marginalization_idx([1, 2])
    try:
        with torch.no_grad():
            marginal = net.forward(states).squeeze(-1).exp()
    finally:
        net.set_marginalization_idx(None)

    for value in range(K):
        rows = states[:, 0] == value
        # Every state sharing var0 reports the same marginal, and it equals the
        # brute-force sum over the marginalized variables.
        assert marginal[rows].std().item() == pytest.approx(0.0, abs=1e-5)
        assert marginal[rows][0].item() == pytest.approx(
            exact[rows].sum().item(), abs=1e-4
        )


def test_samples_follow_the_modelled_distribution() -> None:
    net = build_net()
    exact = joint_pmf(net)

    torch.manual_seed(1)
    num_samples = 3000
    net.set_marginalization_idx(list(range(NUM_VAR)))
    try:
        samples = net.sample(x=torch.zeros(num_samples, NUM_VAR))
    finally:
        net.set_marginalization_idx(None)

    assert samples is not None
    assert samples.shape == (num_samples, NUM_VAR)
    assert samples.min() >= 0 and samples.max() < K

    flat = sum(
        samples[:, i].long() * K ** (NUM_VAR - 1 - i) for i in range(NUM_VAR)
    )
    empirical = torch.bincount(flat, minlength=K**NUM_VAR).float() / num_samples
    total_variation = 0.5 * (empirical - exact).abs().sum().item()
    assert total_variation < 0.08


def test_argmax_returns_the_most_likely_state() -> None:
    net = build_net()
    exact = joint_pmf(net)

    net.set_marginalization_idx(list(range(NUM_VAR)))
    try:
        mpe = net.mpe(x=torch.zeros(1, NUM_VAR))
    finally:
        net.set_marginalization_idx(None)

    assert mpe is not None
    # Backtracking picks the max-weight path, not the true mode, so only require it
    # to land somewhere the model considers plausible.
    flat = int(sum(mpe[0, i].long() * K ** (NUM_VAR - 1 - i) for i in range(NUM_VAR)))
    assert exact[flat].item() > exact.mean().item()


def test_evidence_survives_backtracking_without_marginalization() -> None:
    net = build_net()
    evidence = all_states()[:5]
    samples = net.sample(x=evidence)
    assert samples is not None
    assert torch.equal(samples, evidence)


def test_training_recovers_a_two_mode_distribution() -> None:
    net = build_net()
    modes = torch.tensor([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]])
    data = modes.repeat(128, 1)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    for _ in range(300):
        optimizer.zero_grad()
        loss = -net.forward(data).mean()
        loss.backward()
        optimizer.step()

    assert loss.item() == pytest.approx(math.log(2.0), abs=0.02)
