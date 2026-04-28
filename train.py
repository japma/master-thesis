"""Training script."""

import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from rtpt import RTPT
from tqdm import tqdm

from models.autoencoder.variational_autoencoder import VariationalAutoencoder
from models.cspn import SPFlowCSPN
from utils import (
    create_run_directories,
    format_elapsed_time,
    get_data_loaders,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


def reconstruction_target(images, reconstructed):
    """Return target tensor shape compatible with reconstructed output."""
    if reconstructed.dim() == 2:
        return images.view(images.size(0), -1)
    return images


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.

    Args:
        model: Neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (cpu or cuda).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)

        # Reconstruct the images
        reconstructed = model(images)
        target = reconstruction_target(images, reconstructed)
        loss = criterion(reconstructed, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set.

    Args:
        model: Neural network model (autoencoder).
        test_loader: DataLoader for test data.
        criterion: Loss function (reconstruction loss).
        device: Device to evaluate on.

    Returns:
        Average reconstruction loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)

            reconstructed = model(images)
            target = reconstruction_target(images, reconstructed)
            loss = criterion(reconstructed, target)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def train_cspn_epoch(
    cspn,
    autoencoder,
    train_loader,
    optimizer,
    device,
    label_transform=None,
):
    """Train CSPN for one epoch on latent vectors from a frozen autoencoder."""
    cspn.train()
    autoencoder.eval()
    total_nll = 0.0

    for images, labels in tqdm(train_loader, desc="CSPN Training"):
        images = images.to(device)
        labels = labels.to(device).long()
        if label_transform is not None:
            labels = label_transform(labels)

        with torch.no_grad():
            z = autoencoder.encode(images)

        log_likelihood = cspn(z, labels)
        nll = -log_likelihood.mean()

        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        total_nll += nll.item()

    return total_nll / len(train_loader)


def evaluate_cspn(cspn, autoencoder, test_loader, device, label_transform=None):
    """Evaluate CSPN on latent vectors from a frozen autoencoder."""
    cspn.eval()
    autoencoder.eval()
    total_nll = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="CSPN Evaluating"):
            images = images.to(device)
            labels = labels.to(device).long()
            if label_transform is not None:
                labels = label_transform(labels)

            z = autoencoder.encode(images)
            log_likelihood = cspn(z, labels)
            nll = -log_likelihood.mean()
            total_nll += nll.item()

    return total_nll / len(test_loader)


def train_autoencoder(cfg, device, train_loader, test_loader, run_dirs, rtpt):
    """Train and checkpoint the autoencoder stage."""
    input_size = cfg.data.input_size

    ae_epochs = cfg.model.training.epochs
    ae_learning_rate = cfg.model.training.learning_rate
    latent_size = cfg.data.latent_size

    model = VariationalAutoencoder(
        input_size=input_size,
        latent_size=latent_size,
        image_shape=cfg.data.image_shape,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ae_learning_rate)

    logger.info("Training autoencoder for %s epochs...", ae_epochs)
    for epoch in range(ae_epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        test_loss = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        logger.info(
            "AE Epoch %s/%s - Train Loss: %.4f, Test Loss: %.4f",
            epoch + 1,
            ae_epochs,
            train_loss,
            test_loss,
        )

        rtpt.step(subtitle=f"AE {epoch + 1}/{ae_epochs}")

    save_checkpoint(model.state_dict(), run_dirs.checkpoints_dir, "autoencoder")
    return model


def train_cspn(cfg, device, train_loader, test_loader, run_dirs, autoencoder, rtpt):
    """Train and checkpoint the CSPN stage."""
    latent_size = cfg.data.latent_size
    num_labels = cfg.data.num_classes

    cspn_cfg = cfg.model.cspn
    cspn_epochs = cspn_cfg.epochs
    cspn_learning_rate = cspn_cfg.learning_rate

    cspn = SPFlowCSPN(
        latent_size=latent_size,
        num_labels=num_labels,
    ).to(device)

    cspn_optimizer = optim.Adam(cspn.parameters(), lr=cspn_learning_rate)
    logger.info("Training CSPN on latent space for %s epochs...", cspn_epochs)
    for epoch in range(cspn_epochs):
        train_nll = train_cspn_epoch(
            cspn,
            autoencoder=autoencoder,
            train_loader=train_loader,
            optimizer=cspn_optimizer,
            device=device,
        )
        test_nll = evaluate_cspn(
            cspn,
            autoencoder=autoencoder,
            test_loader=test_loader,
            device=device,
        )

        logger.info(
            "CSPN Epoch %s/%s - Train NLL: %.4f, Test NLL: %.4f",
            epoch + 1,
            cspn_epochs,
            train_nll,
            test_nll,
        )

        rtpt.step(subtitle=f"CSPN {epoch + 1}/{cspn_epochs}")

    save_checkpoint(cspn.state_dict(), run_dirs.checkpoints_dir, "cspn")
    return cspn


def train_model(cfg):
    """Train an autoencoder and CSPN prior on the specified dataset.
    Args:
        cfg: Parsed training configuration.

    Usage:
        python main.py
    """
    start_time = time.perf_counter()

    dataset_name = cfg.data.name
    output_dir = cfg.run_dir

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(
        "Device: %s, Dataset: %s, Output Dir: %s", device, dataset_name, output_dir
    )

    total_epochs = cfg.model.training.epochs + cfg.model.cspn.epochs
    rtpt = RTPT(
        name_initials="JM",
        experiment_name=f"{dataset_name}_training",
        max_iterations=max(total_epochs, 1),
    )
    rtpt.start()

    run_dirs = create_run_directories(output_dir)
    logger.info("Run directory: %s", run_dirs.run_dir)

    train_loader, test_loader = get_data_loaders(
        dataset_name,
        cfg.model.training.batch_size,
        dataset_kwargs=cfg.data.dataset_kwargs,
    )

    model = train_autoencoder(cfg, device, train_loader, test_loader, run_dirs, rtpt)

    train_cspn(cfg, device, train_loader, test_loader, run_dirs, model, rtpt)

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    elapsed_formatted = format_elapsed_time(elapsed_seconds)
    logger.info(
        "Training completed in %s (%.2fs)",
        elapsed_formatted,
        elapsed_seconds,
    )

    return None
