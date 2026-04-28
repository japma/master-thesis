"""Training script."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchinfo import summary
from rtpt import RTPT
import time

from models.autoencoder.variational_autoencoder import VariationalAutoencoder
from models.cspn import SPFlowCSPN
from utils import (
    create_run_directories,
    format_elapsed_time,
    get_data_loaders,
    save_checkpoint,
)


def _resolve_label_transform(label_mode: str, dataset_name: str):
    if label_mode == "dataset":
        return lambda labels: labels

    if label_mode == "mnist_parity":
        if dataset_name != "MNIST":
            raise ValueError(
                "label_mode='mnist_parity' is only supported with data.name='MNIST'."
            )
        return lambda labels: labels % 2

    raise ValueError(
        f"Unsupported CSPN label_mode '{label_mode}'. Supported modes: "
        "'dataset', 'mnist_parity'."
    )


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
    channels = cfg.data.channels
    height = cfg.data.height
    width = cfg.data.width

    ae_epochs = cfg.model.training.epochs
    ae_batch_size = cfg.model.training.batch_size
    ae_learning_rate = cfg.model.training.learning_rate
    latent_size = cfg.model.training.latent_size

    model = VariationalAutoencoder(
        input_size=input_size,
        latent_size=latent_size,
        image_shape=cfg.data.image_shape,
    ).to(device)
    summary(
        model,
        input_size=(ae_batch_size, channels, height, width),
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ae_learning_rate)

    print(f"\nTraining autoencoder for {ae_epochs} epochs...")
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

        print(
            f"AE Epoch {epoch + 1}/{ae_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )

        rtpt.step(subtitle=f"AE {epoch + 1}/{ae_epochs}")

    save_checkpoint(model.state_dict(), run_dirs.checkpoints_dir, "autoencoder")
    return model


def train_cspn(cfg, device, train_loader, test_loader, run_dirs, autoencoder, rtpt):
    """Train and checkpoint the CSPN stage."""
    dataset_name = cfg.data.name
    ae_batch_size = cfg.model.training.batch_size
    latent_size = cfg.model.training.latent_size

    cspn_cfg = cfg.model.cspn
    cspn_epochs = cspn_cfg.epochs
    cspn_learning_rate = cspn_cfg.learning_rate
    cspn_label_embedding_dim = cspn_cfg.label_embedding_dim
    cspn_context_hidden_dim = cspn_cfg.context_hidden_dim
    cspn_context_num_layers = cspn_cfg.context_num_layers
    cspn_num_mixture_components = cspn_cfg.num_mixture_components
    cspn_num_sum_components = cspn_cfg.num_sum_components
    cspn_label_config = cfg.cspn_label
    cspn_label_transform = _resolve_label_transform(
        cspn_label_config.label_mode,
        dataset_name,
    )
    cspn_num_labels = cspn_label_config.num_labels

    cspn = SPFlowCSPN(
        latent_size=latent_size,
        num_labels=cspn_num_labels,
        label_embedding_dim=cspn_label_embedding_dim,
        context_hidden_dim=cspn_context_hidden_dim,
        context_num_layers=cspn_context_num_layers,
        num_mixture_components=cspn_num_mixture_components,
        num_sum_components=cspn_num_sum_components,
    ).to(device)

    # Summarize CSPN with representative latent + label inputs.
    cspn_summary_z = torch.zeros((ae_batch_size, latent_size), device=device)
    cspn_summary_labels = torch.zeros(
        (ae_batch_size,),
        dtype=torch.long,
        device=device,
    )
    summary(
        cspn,
        input_data=(cspn_summary_z, cspn_summary_labels),
        device=device,
    )

    cspn_optimizer = optim.Adam(cspn.parameters(), lr=cspn_learning_rate)
    print(f"\nTraining CSPN on latent space for {cspn_epochs} epochs...")
    for epoch in range(cspn_epochs):
        train_nll = train_cspn_epoch(
            cspn,
            autoencoder=autoencoder,
            train_loader=train_loader,
            optimizer=cspn_optimizer,
            device=device,
            label_transform=cspn_label_transform,
        )
        test_nll = evaluate_cspn(
            cspn,
            autoencoder=autoencoder,
            test_loader=test_loader,
            device=device,
            label_transform=cspn_label_transform,
        )

        print(
            f"CSPN Epoch {epoch + 1}/{cspn_epochs} - "
            f"Train NLL: {train_nll:.4f}, "
            f"Test NLL: {test_nll:.4f}"
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
    cspn_label_config = cfg.cspn_label

    output_dir = cfg.run_dir

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print(
        f"CSPN label mode: {cspn_label_config.label_mode} "
        f"(num_labels={cspn_label_config.num_labels})"
    )

    total_epochs = cfg.model.training.epochs + cfg.model.cspn.epochs
    rtpt = RTPT(
        name_initials="JM",
        experiment_name=f"{dataset_name}_training",
        max_iterations=max(total_epochs, 1),
    )
    rtpt.start()

    run_dirs = create_run_directories(output_dir)
    print(f"Run directory: {run_dirs.run_dir}")

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
    print(f"Training completed in {elapsed_formatted} ({elapsed_seconds:.2f}s)")

    return None
