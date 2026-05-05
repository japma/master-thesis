"""Hydra entrypoint for CSPN training."""

import logging
import time

from hydra import main
from rtpt import RTPT

from dataset_loaders import get_data_loaders
from models.cspn import SPFlowCSPN
from utils import seed_everything, create_run_directories
from utils.config import parse_cspn_train_config
from utils.train import resolve_device

logger = logging.getLogger(__name__)


def _build_cspn(cfg, device):
    cspn_cfg = cfg.model.cspn
    return SPFlowCSPN(
        latent_size=cfg.data.latent_size,
        num_labels=cfg.data.num_classes,
        context_hidden_dim=cspn_cfg.get("context_hidden_dim", 128),
        num_mixture_components=cspn_cfg.get("num_mixture_components", 4),
        num_sum_components=cspn_cfg.get("num_sum_components", 2),
    ).to(device)

def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

def run_cspn_training(cfg):
    start_time = time.perf_counter()

    dataset_name = cfg.data.dataset_name
    output_dir = cfg.run_dir

    device = resolve_device()
    logger.info(
        "Device: %s, Dataset: %s, Output Dir: %s", device, dataset_name, output_dir
    )

    raise NotImplementedError

@main(version_base=None, config_path="configs", config_name="train_cspn")
def main_hydra(cfg) -> None:
    seed = seed_everything(cfg.get("seed"))
    logger.info("Using seed: %s", seed)
    run_cspn_training(parse_cspn_train_config(cfg))


if __name__ == "__main__":
    main_hydra()
