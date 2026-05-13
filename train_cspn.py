"""CSPN training."""

import torch
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from rtpt import RTPT
from tqdm import tqdm

from losses import compute_nll
from utils import (
    create_run_directories,
    save_checkpoint,
)
from utils.io import build_ae_path, load_checkpoint
from utils.models import build_autoencoder, build_cspn
from utils.tracking import WandbTracker
from dataset_loaders import get_data_loaders
from utils.train import resolve_device


def train_epoch(model, ae, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_log_prob = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            z_target = ae.encode(images)

        loss, log_prob = compute_nll(model, z_target, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_log_prob += log_prob

    n = len(train_loader)
    return total_loss / n, total_log_prob / n


def evaluate(model, ae, test_loader, device):
    model.eval()
    total_loss = 0.0
    total_log_prob = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                z_target = ae.encode(images)
            loss, log_prob = compute_nll(model, z_target, labels)

            total_loss += loss.item()
            total_log_prob += log_prob

    n = len(test_loader)
    return total_loss / n, total_log_prob / n


def run_cspn_training(cfg):
    dataset_cfg = cfg.dataset
    hydra_cfg = HydraConfig().get()
    output_dir = hydra_cfg.runtime.output_dir

    device = resolve_device()

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=f"{dataset_cfg.name}_cspn_training",
        max_iterations=max(cfg.training.epochs, 1),
    )
    rtpt.start()

    wandb_run = WandbTracker(cfg)

    run_dirs = create_run_directories(output_dir)

    train_loader, test_loader = get_data_loaders(
        dataset_cfg,
        cfg.training.batch_size,
    )

    ae_path = build_ae_path(cfg)
    ae = build_autoencoder(cfg, device)
    ae.load_state_dict(load_checkpoint(ae_path, map_location=device))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    model = build_cspn(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    for epoch in range(cfg.training.epochs):
        train_loss, train_log_prob = train_epoch(
            model, ae, train_loader, optimizer, device
        )
        test_loss, test_log_prob = evaluate(model, ae, test_loader, device)

        wandb_run.log(
            {
                "train_loss": train_loss,
                "train_log_prob": train_log_prob,
                "test_loss": test_loss,
                "test_log_prob": test_log_prob,
            }
        )
        rtpt.step(subtitle=f"CSPN {epoch + 1}/{cfg.training.epochs}")

    save_checkpoint(model.state_dict(), run_dirs.checkpoints_dir, "cspn")
