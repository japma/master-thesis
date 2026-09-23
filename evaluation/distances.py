"""Distances between two sets of features: `samples` scored against `reference`."""

from collections.abc import Callable

import torch

KERNEL_BLOCK = 4096

# CMMD
CMMD_SIGMA = 10.0
CMMD_SCALE = 1000.0

# KID
KID_DEGREE = 3
KID_COEF0 = 1.0

# Precision/recall
PRC_NEIGHBOURHOOD = 3

Kernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def kernel_sum(a: torch.Tensor, b: torch.Tensor, kernel: Kernel) -> float:
    return sum(kernel(block, b).sum().item() for block in a.split(KERNEL_BLOCK))


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sq = x.pow(2).sum(1, keepdim=True) - 2 * x @ y.T + y.pow(2).sum(1)
    return torch.exp(-sq / (2 * CMMD_SIGMA**2))


def polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x @ y.T / x.shape[1] + KID_COEF0) ** KID_DEGREE


def frechet_distance(samples: torch.Tensor, reference: torch.Tensor) -> float:
    s, r = samples.double(), reference.double()
    mu_s, mu_r = s.mean(dim=0), r.mean(dim=0)
    sigma_s, sigma_r = torch.cov(s.T), torch.cov(r.T)
    sqrt_trace = torch.linalg.eigvals(sigma_s @ sigma_r).sqrt().real.sum()
    distance = (mu_s - mu_r).square().sum() + sigma_s.trace() + sigma_r.trace()
    return float(distance - 2 * sqrt_trace)


def kernel_inception_distance(samples: torch.Tensor, reference: torch.Tensor) -> float:
    s, r = samples.double(), reference.double()
    m, n = s.shape[0], r.shape[0]
    diag_s = ((s * s).sum(1) / s.shape[1] + KID_COEF0) ** KID_DEGREE
    diag_r = ((r * r).sum(1) / r.shape[1] + KID_COEF0) ** KID_DEGREE
    k_ss = (kernel_sum(s, s, polynomial_kernel) - diag_s.sum().item()) / (m * (m - 1))
    k_rr = (kernel_sum(r, r, polynomial_kernel) - diag_r.sum().item()) / (n * (n - 1))
    k_sr = kernel_sum(s, r, polynomial_kernel) / (m * n)
    return k_ss + k_rr - 2 * k_sr


def cmmd(samples: torch.Tensor, reference: torch.Tensor) -> float:
    s, r = samples.double(), reference.double()
    k_ss = kernel_sum(s, s, gaussian_kernel) / s.shape[0] ** 2
    k_rr = kernel_sum(r, r, gaussian_kernel) / r.shape[0] ** 2
    k_sr = kernel_sum(s, r, gaussian_kernel) / (s.shape[0] * r.shape[0])
    return CMMD_SCALE * (k_ss + k_rr - 2 * k_sr)


def manifold_coverage(points: torch.Tensor, support: torch.Tensor) -> float:
    p, s = points.double(), support.double()
    radii = torch.cat(
        [
            torch.cdist(block, s).kthvalue(PRC_NEIGHBOURHOOD + 1, dim=1).values
            for block in s.split(KERNEL_BLOCK)
        ]
    )
    covered = torch.cat(
        [(torch.cdist(block, s) <= radii).any(dim=1) for block in p.split(KERNEL_BLOCK)]
    )
    return covered.double().mean().item()


def precision(samples: torch.Tensor, reference: torch.Tensor) -> float:
    return manifold_coverage(samples, reference)


def recall(samples: torch.Tensor, reference: torch.Tensor) -> float:
    return manifold_coverage(reference, samples)
