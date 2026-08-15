from models.cspn.psinet.label_pc import LabelPC
from pathlib import Path

import networkx
import numpy
import torch
from networkx.classes import DiGraph

import wandb
from models.autoencoder import (
    AbstractAutoencoder,
    VariationalAutoencoder,
)
from models.cspn.abstract_cspn import AbstractCSPN
from models.cspn.psinet.graph import DistributionVector, EiNetAddress, Product
from models.cspn.psinet_cspn import PsiNetCSPN
from utils.config import AutoencoderConfig, AutoencoderType, CSPNConfig, CSPNType
from utils.reproducibility import get_rng_state, set_rng_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# --- Resumable training state ---
# Kept entirely separate from the model checkpoints above: this sidecar file is
# purely additive (optimizer/scheduler/epoch/RNG state for resuming a crashed run),
# so existing model checkpoints (and code that only ever loads those) are completely
# unaffected whether or not a sidecar exists next to them.
def intermediate_checkpoint_path(model_type: str, dataset_name: str) -> Path:
    name = f"intermediate_{model_type}_{dataset_name}"
    return Path("checkpoints/intermediate") / f"{name}.pt"


def label_pc_checkpoint_path(dataset_name: str) -> Path:
    return Path("checkpoints") / f"label_pc_{dataset_name}.pt"


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


# --- General ---
def load_from_wandb(
    ckpt_name: str,
    tag: str = "latest",
) -> Path:
    """Load a checkpoint from wandb artifacts. Uses the most recent checkpoint unless tag is provided."""
    # TODO use use_artifact instead (might solve other inconsistencies as well)
    print(f"Loading checkpoint {ckpt_name}:{tag} from Weights & Biases artifacts...")
    entity = "jmartini-tu-darmstadt"
    project = "master-thesis"
    name = f"{entity}/{project}/{ckpt_name}:{tag}"
    api = wandb.Api()
    artifact = api.artifact(name)
    file = artifact.file(str(ARTIFACTS_DIR))
    print(f"Loaded {file} from Weights & Biases artifact {name}")
    return Path(file)


# --- Autoencoder ---
def save_autoencoder(model: AbstractAutoencoder, path: Path) -> None:
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
    return VariationalAutoencoder(config=cfg)


def load_ae_from_path(path: Path, device=None) -> AbstractAutoencoder:
    with torch.serialization.safe_globals([AutoencoderType]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = AutoencoderConfig.model_validate(ckpt["model_cfg"])
    model = _create_autoencoder_from_checkpoint(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model


# --- CSPN ---
def save_cspn(model: AbstractCSPN, path: Path) -> None:
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


# --- LabelPC ---
def save_label_pc(model: LabelPC, path: Path) -> None:
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
