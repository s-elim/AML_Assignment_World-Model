from typing import Dict, List, Any
import numpy as np
import torch


def compute_latent_mse(z_pred: torch.Tensor, z_true: torch.Tensor) -> float:
    """
    Compute Mean Squared Error between predicted latent token representations and ground-truth tokens.
    """
    diff = z_pred - z_true
    return float(torch.mean(diff ** 2).item())


def compute_multi_horizon_mse(
    z_preds: Dict[int, torch.Tensor],
    z_trues: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """
    Compute latent MSE across multiple rollout prediction horizons k in {1, 3, 5, 10}.
    """
    results = {}
    for k in sorted(z_preds.keys()):
        if k in z_trues:
            results[k] = compute_latent_mse(z_preds[k], z_trues[k])
    return results


def compute_goal_distance(pose_current: np.ndarray, pose_goal: np.ndarray) -> float:
    """
    Compute Euclidean distance between current pose and target goal in centimeters (cm).
    """
    dist_meters = float(np.linalg.norm(pose_current[:3] - pose_goal[:3]))
    return dist_meters * 100.0


def aggregate_multi_seed_metrics(
    seed_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics over multiple random seeds, reporting mean and sample standard deviation.
    """
    if not seed_results:
        return {}

    all_keys = [k for k in seed_results[0].keys() if isinstance(seed_results[0][k], (int, float))]
    aggregated = {}

    for k in all_keys:
        values = [res[k] for res in seed_results if k in res]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregated[k] = {
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return aggregated
