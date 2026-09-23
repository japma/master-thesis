"""Which autoencoder a latent-space model belongs to.

Scoring a PC against the wrong autoencoder is silent -- the latent dimensions still
line up, the numbers are just meaningless -- so the pairing is recorded twice: in wandb
as a run input, and inside the checkpoint itself for when wandb is not around.
"""

from pathlib import Path

import torch

import utils.wandb_utils as wandb_utils
from models.cspn.joint_pc import JointPC
from models.neural_baseline import build_neural_baseline
from utils.checkpoints import (
    load_joint_pc_from_path,
    read_source_artifact,
    save_joint_pc,
    save_nn_baseline,
)
from utils.config import (
    CSPNEncoderConfig,
    CSPNEncoderType,
    JointPCConfig,
    NeuralBaselineConfig,
)
from utils.reproducibility import seed_everything
from utils.wandb_utils import _qualified, artifact_ref, trained_with

AE_REF = "variational_colour_mnist_uniform:v3"


def build_joint_pc() -> JointPC:
    seed_everything(0)
    return JointPC(
        config=JointPCConfig(
            num_latents=4,
            label_cardinalities=[10, 6, 3],
            num_repetitions=2,
            num_input_distributions=4,
            num_sums=4,
        )
    )


def build_baseline():
    seed_everything(0)
    return build_neural_baseline(
        NeuralBaselineConfig(
            model_type="deterministic",
            num_vars=4,
            h_dims=[8],
            encoder_config=CSPNEncoderConfig(
                encoder_type=CSPNEncoderType.MULTI_CATEGORICAL,
                num_classes=[10, 6, 3],
            ),
        )
    )


# --- artifact references ---
def test_a_name_without_a_version_takes_the_tag() -> None:
    assert _qualified("psinet_colour_mnist", "latest").endswith(
        "/psinet_colour_mnist:latest"
    )


def test_a_name_that_already_pins_a_version_keeps_it() -> None:
    """A reference resolved from lineage is passed straight back into the loader, so it
    must not be re-tagged with whatever `--tag` happened to be."""
    assert _qualified("psinet_colour_mnist:v7", "latest").endswith(
        "/psinet_colour_mnist:v7"
    )


class FakeArtifact:
    def __init__(self, name: str, version: str = "v1", type: str = "autoencoder"):
        self.name = name
        self.version = version
        self.type = type


def test_artifact_ref_always_pins_a_version() -> None:
    assert artifact_ref(FakeArtifact("ae:v3", version="v3")) == "ae:v3"
    assert artifact_ref(FakeArtifact("ae", version="v3")) == "ae:v3"
    # Fetched by alias, wandb names the artifact after the alias, not the version.
    assert artifact_ref(FakeArtifact("ae:latest", version="v3")) == "ae:v3"


# --- wandb lineage ---
class FakeRun:
    name = "cspn_colour_mnist_psinet"

    def __init__(self, used: list[FakeArtifact], config: dict | None = None):
        self._used = used
        self.config = config or {}

    def used_artifacts(self) -> list[FakeArtifact]:
        return self._used


class FakeLoggedArtifact:
    def __init__(self, run: FakeRun | None):
        self._run = run

    def logged_by(self) -> FakeRun | None:
        return self._run


def fake_api(monkeypatch, artifact) -> None:
    class Api:
        def artifact(self, name: str):
            if isinstance(artifact, Exception):
                raise artifact
            return artifact

    monkeypatch.setattr(wandb_utils.wandb, "Api", Api)


def test_lineage_names_the_exact_version_the_run_consumed(monkeypatch) -> None:
    run = FakeRun([FakeArtifact("variational_colour_mnist_uniform", version="v3")])
    fake_api(monkeypatch, FakeLoggedArtifact(run))

    assert trained_with("psinet_colour_mnist") == AE_REF


def test_artifacts_of_other_types_are_ignored(monkeypatch) -> None:
    run = FakeRun(
        [
            FakeArtifact("label_pc_colour_mnist", version="v2", type="label_pc"),
            FakeArtifact("variational_colour_mnist_uniform", version="v3"),
        ]
    )
    fake_api(monkeypatch, FakeLoggedArtifact(run))

    assert trained_with("psinet_colour_mnist") == AE_REF


def test_falls_back_to_the_run_config_when_nothing_was_recorded(monkeypatch) -> None:
    """Runs from before the dependency was tracked, and `external: true` autoencoders
    that never become artifacts at all: the name is known, the version is not."""
    run = FakeRun(
        [], config={"autoencoder": {"name": "variational_colour_mnist_uniform"}}
    )
    fake_api(monkeypatch, FakeLoggedArtifact(run))

    assert trained_with("psinet_colour_mnist") == "variational_colour_mnist_uniform"


def test_returns_none_rather_than_guessing(monkeypatch) -> None:
    fake_api(monkeypatch, FakeLoggedArtifact(FakeRun([], config={})))
    assert trained_with("psinet_colour_mnist") is None

    fake_api(monkeypatch, FakeLoggedArtifact(None))
    assert trained_with("psinet_colour_mnist") is None


def test_an_unreachable_api_is_not_an_error(monkeypatch) -> None:
    """Eval should fall through to the checkpoint's own record, not crash, when wandb
    cannot be reached."""
    fake_api(monkeypatch, ConnectionError("no network"))
    assert trained_with("psinet_colour_mnist") is None


# --- self-describing checkpoints ---
def test_joint_pc_checkpoint_records_its_autoencoder(tmp_path: Path) -> None:
    path = tmp_path / "joint_pc.pt"
    save_joint_pc(build_joint_pc(), path, source_artifact=AE_REF)

    assert read_source_artifact(path) == AE_REF


def test_a_checkpoint_saved_without_one_reads_back_as_none(tmp_path: Path) -> None:
    path = tmp_path / "joint_pc.pt"
    save_joint_pc(build_joint_pc(), path)

    assert read_source_artifact(path) is None


def test_recording_it_does_not_disturb_the_weights(tmp_path: Path) -> None:
    model = build_joint_pc()
    path = tmp_path / "joint_pc.pt"
    save_joint_pc(model, path, source_artifact=AE_REF)

    loaded = load_joint_pc_from_path(path)
    for original, restored in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        assert torch.equal(original, restored)


def test_the_baseline_records_it_too(tmp_path: Path) -> None:
    path = tmp_path / "baseline.pt"
    save_nn_baseline(build_baseline(), path, source_artifact=AE_REF)

    assert read_source_artifact(path) == AE_REF


def test_a_pre_existing_checkpoint_without_the_key_reads_back_as_none(
    tmp_path: Path,
) -> None:
    """Checkpoints written before the key existed must keep loading."""
    path = tmp_path / "old.pt"
    torch.save({"model_cfg": {}, "model_state": {}}, path)

    assert read_source_artifact(path) is None
