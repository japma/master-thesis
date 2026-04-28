import logging

from hydra import main
from inference import run_inference
from train import train_model
from utils import seed_everything
from utils.config import parse_inference_config, parse_mode, parse_train_config

logger = logging.getLogger(__name__)


@main(version_base=None, config_path="configs", config_name="config")
def main_hydra(cfg) -> None:
    seed = seed_everything(cfg.get("seed"))
    logger.info("Using seed: %s", seed)

    mode = parse_mode(cfg)
    if mode == "train":
        train_model(parse_train_config(cfg))
        return
    if mode == "inference":
        run_inference(parse_inference_config(cfg))
        return

    raise ValueError(
        f"Unsupported mode '{mode}'. Supported modes are: 'train', 'inference'."
    )


if __name__ == "__main__":
    main_hydra()
