"""Artifact names: derived from config, underscores only, one collection per model."""

from pathlib import Path

import pytest
import yaml

from dataset_loaders.helpers import _DATASETS
from evaluation.generate import resolve_autoencoder
from utils.config import (
    AERunConfig,
    AutoencoderConfig,
    CSPNConfig,
    DatasetConfig,
    DatasetName,
    PretrainedAutoencoderConfig,
)
from utils.config.loading import _apply_dataset_defaults
from utils.naming import (
    VAE_KIND_KEY,
    ModelFamily,
    artifact_name,
    cspn_extras,
    pretrained_vae_kind,
    vae_kind,
)
from utils.wandb_utils import is_versioned


def load(path: Path, cls):
    return cls.model_validate(_apply_dataset_defaults(yaml.safe_load(path.read_text())))


def dataset(name: str = "colour_mnist_skewed", **extra) -> DatasetConfig:
    return DatasetConfig(
        name=name, channels=3, height=28, width=28, num_classes=10, **extra
    )


def test_every_loadable_dataset_has_an_enum_member() -> None:
    assert set(DatasetName) == set(_DATASETS)


def test_a_conditional_model_is_named_type_dataset_vae_kind() -> None:
    assert (
        artifact_name(ModelFamily.CSPN, dataset(), "anchored_digit")
        == "cspn_colour_mnist_skewed_anchored_digit"
    )
    assert (
        artifact_name(ModelFamily.JOINT_PC, dataset(labels=("digit",)), "variational")
        == "joint_pc_colour_mnist_skewed_labels_digit_variational"
    )


def test_the_vae_kind_names_objective_and_beta_only_when_they_differ() -> None:
    configs = {
        path.stem: load(path, AERunConfig)
        for path in Path("configs/autoencoder").glob("*.yaml")
    }

    def kind(stem: str) -> str:
        return vae_kind(configs[stem].model, configs[stem].training)

    assert kind("colour_mnist_skewed") == "variational"
    assert kind("colour_mnist_skewed_anchored") == "anchored"
    assert kind("colour_mnist_skewed_anchored_digit") == "anchored_digit"
    assert kind("mnist_tcvae") == "variational_tcvae"
    assert kind("mnist_beta4") == "variational_beta4"
    assert kind("flowers") == "variational_beta0p5"


def test_every_shipped_autoencoder_gets_its_own_collection() -> None:
    names = [
        artifact_name(ModelFamily.VAE, cfg.dataset, vae_kind(cfg.model, cfg.training))
        for cfg in (
            load(path, AERunConfig)
            for path in Path("configs/autoencoder").glob("*.yaml")
        )
    ]
    assert len(names) == len(set(names))
    assert all("-" not in name for name in names)


def test_cspn_extras_come_from_conditioning_and_label_dropout() -> None:
    base = yaml.safe_load(Path("configs/cspn/colour_mnist_skewed.yaml").read_text())[
        "model"
    ]
    dontcare = yaml.safe_load(
        Path("configs/cspn/colour_mnist_skewed_dontcare.yaml").read_text()
    )["model"]
    factorized = yaml.safe_load(
        Path("configs/cspn/colour_mnist_skewed_factorized.yaml").read_text()
    )["model"]

    assert cspn_extras(CSPNConfig.model_validate(base)) == []
    assert cspn_extras(CSPNConfig.model_validate(dontcare)) == ["dontcare"]
    assert cspn_extras(CSPNConfig.model_validate(factorized)) == ["factorized"]


def test_a_vae_kind_comes_from_its_artifact_then_its_checkpoint() -> None:
    cfg = PretrainedAutoencoderConfig(name="vae_x", external=False)
    model = AutoencoderConfig(
        model_type="variational", latent_dim=16, num_blocks=2, base_channels=32
    )

    assert pretrained_vae_kind(cfg, {VAE_KIND_KEY: "variational_beta4"}, model) == (
        "variational_beta4"
    )
    assert pretrained_vae_kind(cfg, {}, model) == "variational"


def test_a_hugging_face_vae_is_named_after_its_repo() -> None:
    cfg = PretrainedAutoencoderConfig(name="stabilityai/sd-vae-ft-mse", external=True)

    assert pretrained_vae_kind(cfg, {}, None) == "stabilityai_sd_vae_ft_mse"


def test_checkpoints_from_before_the_enums_still_load() -> None:
    legacy_ae = AutoencoderConfig.model_validate(
        {
            "model_type": "variational",
            "latent_dim": 16,
            "num_blocks": 2,
            "base_channels": 32,
            "variant": "",
        }
    )
    base = yaml.safe_load(Path("configs/cspn/colour_mnist_skewed.yaml").read_text())[
        "model"
    ]
    legacy_cspn = CSPNConfig.model_validate({**base, "variant": "anchored"})

    assert legacy_ae.variant is None
    assert "variant" not in legacy_cspn.model_dump()


def test_only_a_concrete_version_counts_as_resolved() -> None:
    assert is_versioned("vae_mnist_variational:v3")
    assert not is_versioned("vae_mnist_variational:latest")
    assert not is_versioned("vae_mnist_variational")


def test_an_unversioned_autoencoder_record_defers_to_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "evaluation.generate.read_source_artifact", lambda path: "anchored_x:latest"
    )
    monkeypatch.setattr("evaluation.generate.trained_with", lambda ref: "anchored_x:v2")

    assert resolve_autoencoder(None, tmp_path / "m.pt", "cspn:v1")[0] == "anchored_x:v2"
