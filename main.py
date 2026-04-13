from hydra import main
from omegaconf import DictConfig
from train import train_model


@main(version_base=None, config_path="configs", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    train_model(cfg)


if __name__ == "__main__":
    main_hydra()
