"""The unconditional SPN: a CSPN's circuit with its parameters learned directly."""

from pathlib import Path

import torch

from models.cspn.spn import SPN
from utils.checkpoints import load_spn_from_path, save_spn
from utils.config import SPNConfig

NUM_VARS = 8


def build(normalize: bool = False) -> SPN:
    torch.manual_seed(0)
    return SPN(
        SPNConfig(
            num_vars=NUM_VARS,
            num_repetitions=2,
            num_input_distributions=3,
            num_sums=3,
            min_var=0.01,
            max_var=4.0,
            normalize_latents=normalize,
        )
    ).eval()


def test_the_density_ignores_the_labels() -> None:
    spn = build()
    z = torch.randn(5, NUM_VARS)

    with torch.no_grad():
        some = spn(z, torch.zeros(5, 40, dtype=torch.long))
        other = spn(z, torch.ones(5, 3, dtype=torch.long))

    assert some.shape == (5,)
    assert torch.equal(some, other)


def test_samples_one_latent_per_label_row() -> None:
    spn = build()

    with torch.no_grad():
        z = spn.sample(torch.zeros(7, 40, dtype=torch.long))

    assert z.shape == (7, NUM_VARS)
    assert torch.isfinite(z).all()


def test_an_spn_survives_a_checkpoint(tmp_path: Path) -> None:
    spn = build(normalize=True)
    spn.set_latent_stats(torch.full((NUM_VARS,), 2.0), torch.full((NUM_VARS,), 3.0))
    path = tmp_path / "spn.pt"
    z = torch.randn(4, NUM_VARS)
    labels = torch.zeros(4, 1)

    save_spn(spn, path, source_artifact="vae:v1")
    loaded = load_spn_from_path(path).eval()

    with torch.no_grad():
        assert torch.allclose(loaded(z, labels), spn(z, labels))
