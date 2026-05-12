from hydra import main
from omegaconf import DictConfig

from train_autoencoder import run_autoencoder_training
from train_cspn import run_cspn_training
from inference import run_inference
from utils import seed_everything


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
