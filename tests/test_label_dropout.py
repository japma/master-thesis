"""The "don't care" label: an extra value per factor that the hypernetwork learns.

Gated behind `encoder_config.label_dropout_prob`, because turning it on widens the
conditioning input and so breaks checkpoint compatibility.
"""

import pytest
import torch

from models.cspn.psinet.label_encoder import build_label_encoder
from utils.config import CSPNEncoderConfig, CSPNEncoderType

CARDINALITIES = [10, 6, 3]


def encoder(prob: float):
    return build_label_encoder(
        CSPNEncoderConfig(
            encoder_type=CSPNEncoderType.MULTI_CATEGORICAL,
            num_classes=CARDINALITIES,
            label_dropout_prob=prob,
        )
    )


def test_the_gate_is_off_by_default() -> None:
    off = encoder(0.0)
    assert not off.allow_unknown
    assert off.unknown_indices == []
    assert off.factor_sizes == CARDINALITIES
    assert off.num_classes == sum(CARDINALITIES)


def test_enabling_it_adds_one_slot_per_factor() -> None:
    on = encoder(0.15)
    assert on.allow_unknown
    # The unknown index is the slot past each factor's real values.
    assert on.unknown_indices == CARDINALITIES
    assert on.factor_sizes == [11, 7, 4]
    assert on.num_classes == sum(CARDINALITIES) + len(CARDINALITIES)


def test_the_unknown_index_encodes_to_its_own_slot() -> None:
    on = encoder(0.15)
    encoded = on(torch.tensor([[10, 6, 3]]))

    assert encoded.shape == (1, 22)
    # One hot per factor, each in the last position of its block.
    assert encoded[0, 10] == 1.0
    assert encoded[0, 11 + 6] == 1.0
    assert encoded[0, 11 + 7 + 3] == 1.0
    assert encoded.sum() == 3.0


def test_a_real_label_still_encodes_where_it_did() -> None:
    """Adding the slot must not move the real values, or a retrain means something
    different by the same index."""
    real = torch.tensor([[5, 2, 1]])
    off, on = encoder(0.0), encoder(0.15)

    assert off(real)[0, :10].tolist() == on(real)[0, :10].tolist()
    assert off(real)[0, 5] == 1.0 and on(real)[0, 5] == 1.0


def test_an_unknown_index_is_out_of_range_when_the_gate_is_off() -> None:
    with pytest.raises(RuntimeError):
        encoder(0.0)(torch.tensor([[10, 6, 3]]))


def test_dropout_only_fires_in_training_mode() -> None:
    from models.cspn.psinet.label_encoder import LabelDropout

    labels = torch.zeros(512, 3, dtype=torch.long)
    dropout = LabelDropout(unknown_indices=CARDINALITIES, dropout_prob=0.5)

    dropout.eval()
    assert torch.equal(dropout(labels), labels)

    dropout.train()
    dropped = dropout(labels)
    share = float((dropped == torch.tensor(CARDINALITIES)).float().mean())
    assert 0.4 < share < 0.6


def test_binary_attributes_get_a_third_value() -> None:
    """CelebA's 40 binary attributes: 0, 1, and "not specified"."""
    celeba = build_label_encoder(
        CSPNEncoderConfig(
            encoder_type=CSPNEncoderType.MULTI_BINARY,
            num_classes=[40],
            label_dropout_prob=0.1,
        )
    )
    assert celeba.factor_sizes == [3] * 40
    assert celeba.unknown_indices == [2] * 40
