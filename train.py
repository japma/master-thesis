"""Training script."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from rtpt import RTPT
import time

from file_utils import create_run_directories, save_checkpoint
from models import Autoencoder, CombinedModel
from models.cspn import SPFlowCSPN
from utils import (
    format_elapsed_time,
    get_data_loaders,
    visualize_autoencoder,
    visualize_cspn,
    visualize_losses,
)


def reconstruction_target(images, reconstructed):
    """Return target tensor shape compatible with reconstructed output."""
    if reconstructed.dim() == 2:
        return images.view(images.size(0), -1)
    return images


def infer_image_shape_from_input_size(input_size):
    """Infer default image shape from flattened size when explicit shape is missing."""
    known_shapes = {
        784: (1, 28, 28),
        3072: (3, 32, 32),
    }
    if input_size not in known_shapes:
        raise ValueError(
            "Unable to infer image shape from input_size. "
            "Please set channels, height, and width in data config."
        )
    return known_shapes[input_size]


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


def train_cspn_epoch(cspn, autoencoder, train_loader, optimizer, device):
    """Train CSPN for one epoch on latent vectors from a frozen autoencoder."""
    cspn.train()
    autoencoder.eval()
    total_nll = 0.0

    for images, labels in tqdm(train_loader, desc="CSPN Training"):
        images = images.to(device)
        labels = labels.to(device).long()

        with torch.no_grad():
            z = autoencoder.encode(images)

        log_likelihood = cspn(z, labels)
        nll = -log_likelihood.mean()

        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        total_nll += nll.item()

    return total_nll / len(train_loader)


def evaluate_cspn(cspn, autoencoder, test_loader, device):
    """Evaluate CSPN on latent vectors from a frozen autoencoder."""
    cspn.eval()
    autoencoder.eval()
    total_nll = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="CSPN Evaluating"):
            images = images.to(device)
            labels = labels.to(device).long()

            z = autoencoder.encode(images)
            log_likelihood = cspn(z, labels)
            nll = -log_likelihood.mean()
            total_nll += nll.item()

    return total_nll / len(test_loader)


def train_model(cfg: DictConfig):
    """Train an autoencoder and CSPN prior on the specified dataset.
    Args:
        cfg: Hydra configuration object containing:
            - data.name: Dataset name (e.g., "MNIST", "CIFAR10")
            - data.input_size: Input size for the model
            - data.channels: Number of channels (1 for grayscale, 3 for RGB)
            - data.height: Image height in pixels
            - data.width: Image width in pixels
            - model.training.epochs: Number of epochs
            - model.training.batch_size: Batch size
            - model.training.learning_rate: Learning rate
            - model.training.latent_size: Size of latent representation
            - output_dir: Directory to save models

    Usage:
        python main.py
    """
    start_time = time.perf_counter()

    dataset_name = cfg.data.name
    input_size = cfg.data.input_size
    channels = cfg.data.channels
    height = cfg.data.height
    width = cfg.data.width

    if channels is None or height is None or width is None:
        channels, height, width = infer_image_shape_from_input_size(input_size)

    ae_epochs = cfg.model.training.epochs
    ae_batch_size = cfg.model.training.batch_size
    ae_learning_rate = cfg.model.training.learning_rate
    latent_size = cfg.model.training.latent_size
    output_dir = cfg.output_dir

    cspn_cfg = cfg.model.cspn
    cspn_epochs = cspn_cfg.get("epochs", 10)
    cspn_learning_rate = cspn_cfg.get("learning_rate", 1e-3)
    cspn_label_embedding_dim = cspn_cfg.get("label_embedding_dim", 32)
    cspn_context_hidden_dim = cspn_cfg.get("context_hidden_dim", 128)
    cspn_context_num_layers = cspn_cfg.get("context_num_layers", 3)
    cspn_num_mixture_components = cspn_cfg.get("num_mixture_components", 4)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")

    total_epochs = cfg.model.training.epochs + cfg.model.cspn.get("epochs", 10)
    rtpt = RTPT(
        name_initials="JM",
        experiment_name=f"{dataset_name}_training",
        max_iterations=max(total_epochs, 1),
    )
    rtpt.start()

    run_dirs = create_run_directories(output_dir, dataset_name)
    print(f"Run directory: {run_dirs.run_dir}")

    model = Autoencoder(
        input_size=input_size,
        latent_size=latent_size,
    ).to(device)
    summary(
        model,
        input_size=(ae_batch_size, channels, height, width),
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ae_learning_rate)

    dataset_kwargs = (
        OmegaConf.to_container(cfg.data.dataset_kwargs, resolve=True)
        if "dataset_kwargs" in cfg.data
        else None
    )
    train_loader, test_loader = get_data_loaders(
        dataset_name,
        ae_batch_size,
        dataset_kwargs=dataset_kwargs,
    )

    ae_train_losses = []
    ae_test_losses = []

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

        ae_train_losses.append(train_loss)
        ae_test_losses.append(test_loss)

        print(
            f"AE Epoch {epoch + 1}/{ae_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )

        rtpt.step(subtitle=f"AE {epoch + 1}/{ae_epochs}")

    save_checkpoint(model.state_dict(), run_dirs.checkpoints_dir, "autoencoder")
    visualize_losses(
        train_values=ae_train_losses,
        test_values=ae_test_losses,
        output_dir=run_dirs.images_dir,
        filename="autoencoder_loss.png",
        title="Autoencoder Reconstruction Loss",
        y_label="MSE Loss",
        train_label="Train",
        test_label="Test",
    )

    visualize_autoencoder(
        model,
        test_loader,
        device,
        run_dirs.images_dir,
        num_samples=10,
    )

    # Start training CSPN
    num_labels = cfg.data.get("num_classes", 10)
    cspn = SPFlowCSPN(
        latent_size=latent_size,
        num_labels=num_labels,
        label_embedding_dim=cspn_label_embedding_dim,
        context_hidden_dim=cspn_context_hidden_dim,
        context_num_layers=cspn_context_num_layers,
        num_mixture_components=cspn_num_mixture_components,
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
    cspn_train_losses = []
    cspn_test_losses = []

    print(f"\nTraining CSPN on latent space for {cspn_epochs} epochs...")
    for epoch in range(cspn_epochs):
        train_nll = train_cspn_epoch(
            cspn,
            autoencoder=model,
            train_loader=train_loader,
            optimizer=cspn_optimizer,
            device=device,
        )
        test_nll = evaluate_cspn(
            cspn,
            autoencoder=model,
            test_loader=test_loader,
            device=device,
        )

        cspn_train_losses.append(train_nll)
        cspn_test_losses.append(test_nll)

        print(
            f"CSPN Epoch {epoch + 1}/{cspn_epochs} - "
            f"Train NLL: {train_nll:.4f}, "
            f"Test NLL: {test_nll:.4f}"
        )

        rtpt.step(subtitle=f"CSPN {epoch + 1}/{cspn_epochs}")

    save_checkpoint(cspn.state_dict(), run_dirs.checkpoints_dir, "cspn")
    visualize_losses(
        train_values=cspn_train_losses,
        test_values=cspn_test_losses,
        output_dir=run_dirs.images_dir,
        filename="cspn_nll.png",
        title="CSPN Negative Log-Likelihood",
        y_label="NLL",
        train_label="Train",
        test_label="Test",
    )

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    elapsed_formatted = format_elapsed_time(elapsed_seconds)
    print(f"Training completed in {elapsed_formatted} ({elapsed_seconds:.2f}s)")

    visualize_cspn(
        autoencoder=model,
        cspn=cspn,
        test_loader=test_loader,
        device=device,
        output_dir=run_dirs.images_dir,
        num_labels=cfg.data.get("num_classes", 10),
        num_samples=8,
    )

    final_model = CombinedModel(autoencoder=model, cspn=cspn)
    save_checkpoint(final_model.state_dict(), run_dirs.checkpoints_dir, "latent_model")

    return final_model
