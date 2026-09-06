"""The leaf layer is shared, so LabelPC and PsiNetCSPN guard the paths a mixed-family
change could quietly break: Binomial leaves without a hypernetwork, and Normal leaves
with one."""

import pytest
import torch

from models.cspn.psinet.label_pc import LabelPC
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.config import CSPNConfig, CSPNEncoderConfig, CSPNEncoderType, CSPNType
from utils.reproducibility import seed_everything

NUM_VARS = 8


@pytest.fixture
def cspn() -> PsiNetCSPN:
    seed_everything(0)
    return PsiNetCSPN(
        config=CSPNConfig(
            model_type=CSPNType.PSINET,
            num_vars=NUM_VARS,
            num_repetitions=2,
            num_input_distributions=4,
            num_sums=4,
            min_var=1e-3,
            max_var=4.0,
            h_dims=[16],
            encoder_config=CSPNEncoderConfig(
                encoder_type=CSPNEncoderType.MULTI_CATEGORICAL,
                num_classes=[10, 6, 3],
            ),
        )
    )


def test_label_pc_trains_and_completes() -> None:
    seed_everything(0)
    model = LabelPC(num_attributes=4, num_input_distributions=4, num_sums=4,
                    num_repetitions=2)
    attributes = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]).repeat(32, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        loss = -model.log_likelihood(attributes).mean()
        loss.backward()
        optimizer.step()

    completed = model.complete_partial({0: 1.0}, batch_size=32)
    assert completed.shape == (32, 4)
    assert completed[:, 0].eq(1.0).all()
    assert set(completed.unique().tolist()) <= {0.0, 1.0}


def test_cspn_forward_sample_and_conditional(cspn: PsiNetCSPN) -> None:
    # The label encoder carries dropout, so densities are only reproducible in eval.
    cspn.eval()
    labels = torch.tensor([[3, 1, 2], [7, 0, 0]])
    z = torch.randn(2, NUM_VARS)

    log_prob = cspn(z, labels)
    assert log_prob.shape == (2,)
    assert torch.isfinite(log_prob).all()

    samples = cspn.sample(labels, std_correction=0.8)
    assert samples.shape == (2, NUM_VARS)
    assert torch.isfinite(samples).all()

    conditional = cspn.sample_conditional_partial(labels, known={0: 1.5, 3: -2.0})
    assert conditional[:, 0].eq(1.5).all()
    assert conditional[:, 3].eq(-2.0).all()

    marginal = cspn.log_marginal(z, labels, observed_idx=[0, 1])
    assert marginal.shape == (2,)
    assert torch.isfinite(marginal).all()
    scrambled = z.clone()
    scrambled[:, 2:] = 100.0
    assert torch.allclose(
        marginal, cspn.log_marginal(scrambled, labels, observed_idx=[0, 1]), atol=1e-5
    )

    assert cspn.mpe(labels).shape == (2, NUM_VARS)
