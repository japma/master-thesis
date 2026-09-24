from pathlib import Path

import networkx
import numpy
import torch
from networkx.classes import DiGraph

from models.autoencoder import (
    AbstractAutoencoder,
    AnchoredVAE,
    SupervisedVAE,
    VariationalAutoencoder,
)
from models.classifier import DigitClassifier
from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.joint_pc import JointPC
from models.cspn.psinet.graph import DistributionVector, EiNetAddress, Product
from models.cspn.psinet.label_pc import LabelPC
from models.cspn.psinet_cspn import PsiNetCSPN
from models.cspn.spn import SPN
from models.latent_prior import GaussianMixturePrior, StandardNormalPrior
from models.neural_baseline import AbstractNeuralBaseline, build_neural_baseline
from utils.compilation import uncompiled
from utils.config import (
    AnchorScheme,
    AutoencoderConfig,
    AutoencoderType,
    AutoencoderVariant,
    ClassifierConfig,
    CSPNConfig,
    CSPNType,
    JointPCConfig,
    NeuralBaselineConfig,
    NeuralBaselineType,
    SPNConfig,
)
from utils.naming import intermediate_name
from utils.reproducibility import get_rng_state, set_rng_state


# --- Resumable training state ---
# Kept entirely separate from the model checkpoints above: this sidecar file is
# purely additive (optimizer/scheduler/epoch/RNG state for resuming a crashed run),
# so existing model checkpoints (and code that only ever loads those) are completely
# unaffected whether or not a sidecar exists next to them.
def intermediate_checkpoint_path(name: str) -> Path:
    """`name` as `utils.naming` derives it; the artifact is logged under the stem."""
    return Path("checkpoints/intermediate") / f"{intermediate_name(name)}.pt"


def final_checkpoint_path(name: str) -> Path:
    return Path("checkpoints") / f"{name}.pt"


def label_pc_checkpoint_path(dataset_name: str) -> Path:
    return final_checkpoint_path(f"label_pc_{dataset_name}")


def train_state_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".trainstate.pt")


def save_train_state(
    path: Path,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
            "rng_state": get_rng_state(),
            "extra": extra or {},
        },
        path,
    )
    print("Saved training state to", path)


def load_train_state(path: Path, device: torch.device | None = None) -> dict | None:
    """Returns None (rather than raising) if no sidecar exists -- the normal case for
    a fresh run, or for any checkpoint saved before this feature existed."""
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def restore_train_state(
    state: dict,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Applies a loaded train-state dict and returns the epoch to resume from."""
    optimizer.load_state_dict(state["optimizer_state"])
    lr_scheduler.load_state_dict(state["lr_scheduler_state"])
    set_rng_state(state["rng_state"])
    return state["epoch"] + 1


_LEGACY_CONFIG_GLOBALS: list = [
    (AutoencoderType, "utils.config.autoencoder.AutoencoderType"),
]


# --- Autoencoder ---
def save_autoencoder(model: AbstractAutoencoder, path: Path) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
        },
        path,
    )
    print("Saved autoencoder checkpoint to", path)


def _create_autoencoder_from_checkpoint(cfg: AutoencoderConfig) -> AbstractAutoencoder:
    """Dispatches on model_type: a supervised or anchored checkpoint carries
    classification head weights a plain VariationalAutoencoder has nowhere to put."""
    match cfg.model_type:
        case AutoencoderType.SUPERVISED:
            return SupervisedVAE(config=cfg)
        case AutoencoderType.ANCHORED:
            return AnchoredVAE(config=cfg)
        case _:
            return VariationalAutoencoder(config=cfg)


def load_ae_from_path(path: Path, device=None) -> AbstractAutoencoder:
    with torch.serialization.safe_globals(
        [AnchorScheme, AutoencoderType, AutoencoderVariant, *_LEGACY_CONFIG_GLOBALS]
    ):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = AutoencoderConfig.model_validate(ckpt["model_cfg"])
    model = _create_autoencoder_from_checkpoint(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_vae_prior_from_path(path: Path, device=None) -> StandardNormalPrior:
    """The prior of the VAE checkpoint at `path`; only its latent size is read."""
    vae = load_ae_from_path(path, device="cpu")
    return StandardNormalPrior(int(vae.get_latent_dim().numel()))


# --- CSPN ---
# The autoencoder artifact a latent-space model was trained against, stored inside its
# own checkpoint so the pairing survives without wandb -- a local file, an offline box,
# a run whose lineage was never recorded. `utils.wandb_utils.trained_with` is the same
# answer read off the server instead.
SOURCE_ARTIFACT_KEY = "source_artifact"


def read_source_artifact(path: Path) -> str | None:
    """The `name:version` of the autoencoder `path` was trained with, if it recorded
    one. Checkpoints written before this existed simply have no entry."""
    with torch.serialization.safe_globals(
        [
            networkx.classes.digraph.DiGraph,
            DistributionVector,
            EiNetAddress,
            Product,
            numpy._core.multiarray.scalar,
            numpy.dtype,
        ]
    ):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    source = ckpt.get(SOURCE_ARTIFACT_KEY)
    return str(source) if source else None


def save_cspn(
    model: AbstractCSPN, path: Path, source_artifact: str | None = None
) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = getattr(model, "graph", None)
    if graph is None:
        raise AssertionError("model has no `.graph` attribute")

    if not isinstance(model, PsiNetCSPN):
        raise AssertionError("model is not a PsiNetCSPN")

    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
            "graph": model.get_graph(),
            SOURCE_ARTIFACT_KEY: source_artifact,
        },
        path,
    )
    print("Saved CSPN checkpoint to", path)


def _create_cspn_from_checkpoint(cfg: CSPNConfig, graph: DiGraph) -> AbstractCSPN:
    return PsiNetCSPN(config=cfg, graph=graph)


def load_cspn_from_path(path: Path, device=None) -> AbstractCSPN:
    with (
        # TODO check this, the whole graph gets saved, maybe there is some better way??
        torch.serialization.safe_globals([CSPNType]),
        torch.serialization.safe_globals([networkx.classes.digraph.DiGraph]),
        torch.serialization.safe_globals([DistributionVector]),
        torch.serialization.safe_globals([EiNetAddress]),
        torch.serialization.safe_globals([Product]),
        torch.serialization.safe_globals([numpy._core.multiarray.scalar]),
        torch.serialization.safe_globals([numpy.dtype]),
    ):
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if "graph" not in ckpt:
        raise AssertionError(f"Checkpoint at {path} has no saved `graph` entry")

    cfg = CSPNConfig.model_validate(ckpt["model_cfg"])
    model = _create_cspn_from_checkpoint(cfg, graph=ckpt["graph"])
    model.load_state_dict(ckpt["model_state"])
    return model


# --- Neural baseline ---
def save_nn_baseline(
    model: AbstractNeuralBaseline, path: Path, source_artifact: str | None = None
) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
            SOURCE_ARTIFACT_KEY: source_artifact,
        },
        path,
    )
    print("Saved neural baseline checkpoint to", path)


def load_nn_baseline_from_path(path: Path, device=None) -> AbstractNeuralBaseline:
    with torch.serialization.safe_globals([NeuralBaselineType]):
        ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg = NeuralBaselineConfig.model_validate(ckpt["model_cfg"])
    model = build_neural_baseline(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device) if device is not None else model


# --- Unconditional SPN ---
def save_spn(
    model: AbstractCSPN, path: Path, source_artifact: str | None = None
) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, SPN):
        raise AssertionError("model is not an SPN")
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
            "graph": model.get_graph(),
            SOURCE_ARTIFACT_KEY: source_artifact,
        },
        path,
    )
    print("Saved SPN checkpoint to", path)


def load_spn_from_path(path: Path, device=None) -> SPN:
    with (
        torch.serialization.safe_globals([networkx.classes.digraph.DiGraph]),
        torch.serialization.safe_globals([DistributionVector]),
        torch.serialization.safe_globals([EiNetAddress]),
        torch.serialization.safe_globals([Product]),
        torch.serialization.safe_globals([numpy._core.multiarray.scalar]),
        torch.serialization.safe_globals([numpy.dtype]),
    ):
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if "graph" not in ckpt:
        raise AssertionError(f"Checkpoint at {path} has no saved `graph` entry")

    cfg = SPNConfig.model_validate(ckpt["model_cfg"])
    model = SPN(config=cfg, graph=ckpt["graph"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device) if device is not None else model


# --- Gaussian mixture ---
def save_gmm(
    model: GaussianMixturePrior, path: Path, source_artifact: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
            SOURCE_ARTIFACT_KEY: source_artifact,
        },
        path,
    )
    print("Saved Gaussian mixture checkpoint to", path)


def load_gmm_from_path(path: Path, device=None) -> GaussianMixturePrior:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GaussianMixturePrior(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device) if device is not None else model


# --- LabelPC ---
def save_label_pc(model: LabelPC, path: Path) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
            "graph": model.get_graph(),
        },
        path,
    )
    print("Saved LabelPC checkpoint to", path)


def _create_label_pc_from_checkpoint(cfg: dict, graph: DiGraph) -> LabelPC:
    return LabelPC(
        num_attributes=cfg["num_attributes"],
        num_input_distributions=cfg["num_input_distributions"],
        num_sums=cfg["num_sums"],
        num_repetitions=cfg["num_repetitions"],
        graph=graph,
    )


def load_label_pc_from_path(path: Path, device=None) -> LabelPC:
    with (
        torch.serialization.safe_globals([networkx.classes.digraph.DiGraph]),
        torch.serialization.safe_globals([DistributionVector]),
        torch.serialization.safe_globals([EiNetAddress]),
        torch.serialization.safe_globals([Product]),
        torch.serialization.safe_globals([numpy._core.multiarray.scalar]),
        torch.serialization.safe_globals([numpy.dtype]),
    ):
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if "graph" not in ckpt:
        raise AssertionError(f"Checkpoint at {path} has no saved `graph` entry")

    model = _create_label_pc_from_checkpoint(ckpt["model_cfg"], graph=ckpt["graph"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device) if device is not None else model


# --- Joint latent+label PC ---
def joint_pc_checkpoint_path(dataset_name: str) -> Path:
    return Path("checkpoints") / f"joint_pc_{dataset_name}.pt"


def save_joint_pc(
    model: AbstractCSPN, path: Path, source_artifact: str | None = None
) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, JointPC):
        raise AssertionError("model is not a JointPC")
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
            "graph": model.get_graph(),
            SOURCE_ARTIFACT_KEY: source_artifact,
        },
        path,
    )
    print("Saved JointPC checkpoint to", path)


def load_joint_pc_from_path(path: Path, device=None) -> JointPC:
    with (
        torch.serialization.safe_globals([networkx.classes.digraph.DiGraph]),
        torch.serialization.safe_globals([DistributionVector]),
        torch.serialization.safe_globals([EiNetAddress]),
        torch.serialization.safe_globals([Product]),
        torch.serialization.safe_globals([numpy._core.multiarray.scalar]),
        torch.serialization.safe_globals([numpy.dtype]),
    ):
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if "graph" not in ckpt:
        raise AssertionError(f"Checkpoint at {path} has no saved `graph` entry")

    cfg = JointPCConfig.model_validate(ckpt["model_cfg"])
    model = JointPC(config=cfg, graph=ckpt["graph"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device) if device is not None else model


# --- Digit classifier ---
def save_classifier(model: DigitClassifier, path: Path) -> None:
    model = uncompiled(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.get_config(),
            "model_state": model.state_dict(),
        },
        path,
    )
    print("Saved classifier checkpoint to", path)


def load_classifier_from_path(path: Path, device=None) -> DigitClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ClassifierConfig.model_validate(ckpt["model_cfg"])
    model = DigitClassifier(config=cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device) if device is not None else model
