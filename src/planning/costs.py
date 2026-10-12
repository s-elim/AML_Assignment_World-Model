"""Latent goal costs between imagined final-frame tokens (S, P, D) and goal tokens (1, P, D)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1(z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all tokens and channels (the V-JEPA 2-AC planning energy)."""
    return (z - g).abs().flatten(1).mean(1)


def mse(z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Mean squared error (the DINO-WM planning objective)."""
    return (z - g).pow(2).flatten(1).mean(1)


def l2(z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between the flattened token grids."""
    return (z - g).flatten(1).norm(dim=1)


def cosine(z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """One minus the cosine similarity of the flattened token grids."""
    return 1.0 - F.cosine_similarity(z.flatten(1), g.flatten(1).expand_as(z.flatten(1)), dim=1)


class DiagMahalanobis:
    """Mahalanobis distance with a diagonal per-channel covariance estimated from training tokens."""

    def __init__(self, var: torch.Tensor):
        self.inv = 1.0 / var.clamp_min(1e-6)  # (D,)

    def __call__(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return ((z - g).pow(2) * self.inv).flatten(1).mean(1).sqrt()


COSTS = {"l1": l1, "mse": mse, "l2": l2, "cosine": cosine}


def build_cost(name: str, channel_var: torch.Tensor | None = None):
    if name == "mahalanobis":
        if channel_var is None:
            raise ValueError("mahalanobis needs the per-channel variance of training tokens")
        return DiagMahalanobis(channel_var)
    return COSTS[name]
