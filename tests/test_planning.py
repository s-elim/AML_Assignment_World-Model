"""CEM must find the minimiser of a known cost; latent costs must be zero at the goal."""

import torch

from src.planning.cem import CEM, CEMConfig
from src.planning.costs import build_cost


def test_cem_converges_on_quadratic() -> None:
    target = torch.tensor([0.3, -0.5, 0.1, 0.0, 0.2, -0.1, 0.8])
    cem = CEM(CEMConfig(horizon=4, samples=256, iterations=15), action_dim=7, seed=0)
    plan = cem.plan(lambda a: (a - target).pow(2).sum((1, 2)))
    assert (plan - target).abs().max() < 0.1


def test_cem_respects_bounds_and_smoothing() -> None:
    cem = CEM(CEMConfig(horizon=6, samples=64, iterations=3, smooth=0.8), action_dim=7, seed=0)
    plan = cem.plan(lambda a: -a.sum((1, 2)))
    assert plan.max() <= 1.0 and plan.min() >= -1.0


def test_costs_vanish_at_goal() -> None:
    g = torch.randn(1, 64, 32)
    for name in ["l1", "mse", "l2", "cosine"]:
        assert build_cost(name)(g.clone(), g).abs().max() < 1e-5
    assert build_cost("mahalanobis", torch.ones(32))(g.clone(), g).abs().max() < 1e-5
