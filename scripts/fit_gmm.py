"""Entry point for the ex-post density baseline: a Gaussian mixture over a VAE's
encoded training set (Ghosh et al., 2020)."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset_loaders import build_dataset
from models.autoencoder import AbstractAutoencoder
from models.latent_prior import GaussianMixturePrior
from training.inputs import load_training_autoencoder
from utils.checkpoints import final_checkpoint_path, save_gmm
from utils.config import DatasetConfig, GMMRunConfig, load_config
from utils.naming import ModelFamily, artifact_name, gmm_extras
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, log_checkpoint_artifact, log_summary, rename_run

LOADER_WORKERS = 8


@torch.no_grad()
def encode_split(
    ae: AbstractAutoencoder,
    dataset: DatasetConfig,
    train: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """The posterior means of a whole split, in float64 for EM."""
    split = build_dataset(
        dataset.name,
        train=train,
        size=(dataset.height, dataset.width),
        labels=dataset.labels,
    )
    loader = DataLoader(split, batch_size=batch_size, num_workers=LOADER_WORKERS)
    latents = [
        ae.encode(images.to(device)).float().cpu()
        for images, _ in tqdm(loader, desc="train" if train else "val")
    ]
    return torch.cat(latents).double().numpy()


def to_prior(mixture: GaussianMixture) -> GaussianMixturePrior:
    num_components, latent_dim = mixture.means_.shape
    prior = GaussianMixturePrior(num_components, latent_dim)
    covariances = torch.from_numpy(mixture.covariances_)
    with torch.no_grad():
        prior.weights.copy_(torch.from_numpy(mixture.weights_))
        prior.means.copy_(torch.from_numpy(mixture.means_))
        prior.scale_tril.copy_(torch.linalg.cholesky(covariances))
    return prior


def main() -> None:
    cfg, cfg_seed, _ = load_config()
    assert isinstance(cfg, GMMRunConfig)
    dataset_cfg = cfg.dataset
    training_cfg = cfg.training

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    init_run(cfg.wandb, f"gmm_{dataset_cfg.artifact_name}", cfg.model_dump())

    autoencoder = load_training_autoencoder(cfg.autoencoder, dataset_cfg, device)
    ae = autoencoder.model.eval()
    run_name = artifact_name(
        ModelFamily.GMM, dataset_cfg, autoencoder.kind, *gmm_extras(cfg.model)
    )
    rename_run(run_name)

    train = encode_split(ae, dataset_cfg, True, training_cfg.batch_size, device)
    val = encode_split(ae, dataset_cfg, False, training_cfg.batch_size, device)
    print(
        f"Fitting {cfg.model.num_components} components to {train.shape[0]} "
        f"{train.shape[1]}-dim latents | seed={seed}"
    )

    mixture = GaussianMixture(
        n_components=cfg.model.num_components,
        covariance_type="full",
        reg_covar=cfg.model.reg_covar,
        max_iter=training_cfg.epochs,
        random_state=seed,
        verbose=2,
    )
    mixture.fit(train)
    if not mixture.converged_:
        print(f"WARNING: EM did not converge in {training_cfg.epochs} iterations")

    # Mean log-likelihood per latent, comparable to a PC's NLL on the same latents.
    summary = {
        "train_log_likelihood": float(mixture.score(train)),
        "val_log_likelihood": float(mixture.score(val)),
        "em_iterations": float(mixture.n_iter_),
        "converged": float(mixture.converged_),
    }
    print(summary)
    log_summary(summary)

    path = final_checkpoint_path(run_name)
    save_gmm(to_prior(mixture), path, source_artifact=autoencoder.ref)
    log_checkpoint_artifact(
        path,
        name=path.stem,
        type="gmm",
        metadata={**autoencoder.metadata(), "config": cfg.model_dump(mode="json")},
    )
    wandb.finish()


if __name__ == "__main__":
    main()
