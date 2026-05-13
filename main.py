from hydra import main
from omegaconf import DictConfig
from rtpt import RTPT
from torch import device

from train_autoencoder import run_autoencoder_training
from train_cspn import run_cspn_training
from inference import run_inference
from utils import seed_everything
from utils.tracking import WandbTracker
from utils.train import resolve_device


def setup_training(cfg: DictConfig) -> tuple[device, RTPT, WandbTracker]:
    # TODO add config parsing
    name = "placeholder"
    dev = resolve_device()
    rtpt = RTPT(
        name_initials="JM",
        experiment_name=name,
        max_iterations=max(cfg.training.epochs, 1),
    )
    rtpt.start()
    wandb_run = WandbTracker(cfg)

    return dev, rtpt, wandb_run


@main(version_base=None, config_path="configs", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    if cfg.mode == "train_ae":
        run_autoencoder_training(cfg)
    elif cfg.mode == "train_cspn":
        run_cspn_training(cfg)
    elif cfg.mode == "inference":
        run_inference(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main_hydra()
