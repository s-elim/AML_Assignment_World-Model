import numpy as np
import torch
import pytest

from src.utils.seed import set_seed
from src.utils.metrics import (
    compute_latent_mse,
    compute_multi_horizon_mse,
    compute_goal_distance,
    aggregate_multi_seed_metrics,
)


def test_deterministic_seeding():
    set_seed(42)
    t1 = torch.randn(5, 5)
    np1 = np.random.randn(5)

    set_seed(42)
    t2 = torch.randn(5, 5)
    np2 = np.random.randn(5)

    assert torch.allclose(t1, t2)
    assert np.allclose(np1, np2)


def test_compute_latent_mse():
    z_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z_true = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
    mse = compute_latent_mse(z_pred, z_true)
    assert pytest.approx(mse, 1e-4) == 0.5


def test_compute_multi_horizon_mse():
    z_preds = {
        1: torch.tensor([[1.0, 2.0]]),
        5: torch.tensor([[2.0, 4.0]]),
    }
    z_trues = {
        1: torch.tensor([[1.0, 2.0]]),
        5: torch.tensor([[0.0, 0.0]]),
    }
    mses = compute_multi_horizon_mse(z_preds, z_trues)
    assert pytest.approx(mses[1], 1e-4) == 0.0
    assert pytest.approx(mses[5], 1e-4) == 10.0


def test_compute_goal_distance():
    p1 = np.array([0.1, 0.2, 0.3])
    p2 = np.array([0.1, 0.2, 0.0])
    dist_cm = compute_goal_distance(p1, p2)
    assert pytest.approx(dist_cm, 1e-3) == 30.0  # 0.3m = 30cm


def test_aggregate_multi_seed_metrics():
    seed_results = [
        {"success_rate": 80.0, "latency_ms": 40.0},
        {"success_rate": 84.0, "latency_ms": 42.0},
        {"success_rate": 82.0, "latency_ms": 38.0},
    ]
    summary = aggregate_multi_seed_metrics(seed_results)
    assert pytest.approx(summary["success_rate"]["mean"], 1e-2) == 82.0
    assert pytest.approx(summary["latency_ms"]["mean"], 1e-2) == 40.0
    assert "std" in summary["success_rate"]
