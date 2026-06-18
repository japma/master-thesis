import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import numpy as np
import umap


def show(tensor, title=None, width=8):
    grid = vutils.make_grid(tensor.cpu(), nrow=width, normalize=True)
    plt.figure(figsize=(16, 4), dpi=300)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def plot_latent_comparison(
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    reference: torch.Tensor | None = None,
    name_a: str = "Model A",
    name_b: str = "Model B",
    name_reference: str = "Ground truth",
    title: str = "Latent space comparison",
):
    a = latents_a.detach().cpu().numpy()
    b = latents_b.detach().cpu().numpy()
    n_a = a.shape[0]
    n_b = b.shape[0]

    tensors = [a, b]
    if reference is not None:
        ref = reference.detach().cpu().numpy()
        tensors.append(ref)

    combined = np.concatenate(tensors, axis=0)

    reducer = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(combined)

    proj_a = projected[:n_a]
    proj_b = projected[n_a : n_a + n_b]

    plt.figure(figsize=(8, 8))
    plt.scatter(proj_a[:, 0], proj_a[:, 1], marker="o", alpha=0.7, label=name_a)
    plt.scatter(proj_b[:, 0], proj_b[:, 1], marker="o", alpha=0.7, label=name_b)

    if reference is not None:
        proj_ref = projected[n_a + n_b :]
        plt.scatter(
            proj_ref[:, 0], proj_ref[:, 1], marker="x", alpha=0.7, label=name_reference
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
