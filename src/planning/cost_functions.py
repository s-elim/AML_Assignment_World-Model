import torch
import torch.nn.functional as F


def cosine_distance(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance in representation space:
    D_cost(u, v) = 1 - (u^T v) / (||u||_2 ||v||_2)
    Input shapes: (B, D) or (B, N, D)
    Returns: (B,) distance values
    """
    if z_pred.ndim == 3:
        z_pred = z_pred.flatten(1)
        z_goal = z_goal.flatten(1)

    cos_sim = F.cosine_similarity(z_pred, z_goal, dim=-1)
    return 1.0 - cos_sim


def l1_distance(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    Mean L1 distance between predicted and goal representations.
    Input shapes: (B, D) or (B, N, D)
    Returns: (B,)
    """
    diff = torch.abs(z_pred - z_goal)
    if diff.ndim > 1:
        return torch.mean(diff.flatten(1), dim=-1)
    return diff


def l2_distance(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    Mean squared L2 distance between predicted and goal representations.
    Input shapes: (B, D) or (B, N, D)
    Returns: (B,)
    """
    diff = (z_pred - z_goal) ** 2
    if diff.ndim > 1:
        return torch.sum(diff.flatten(1), dim=-1)
    return diff
