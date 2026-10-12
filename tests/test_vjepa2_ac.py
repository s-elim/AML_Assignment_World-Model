"""The cached rollout must equal the released planner's full re-forward, and the training loss must be finite."""

import torch

from src.models.vjepa2_ac import VJEPA2AC


def _model() -> VJEPA2AC:
    torch.manual_seed(0)
    m = VJEPA2AC(embed_dim=64, pred_dim=128, depth=2).eval()
    with torch.no_grad():
        m.integrator.W.normal_(0, 0.1)
    return m


def test_cached_rollout_matches_full_forward() -> None:
    m = _model()
    S, H = 3, 5
    z0 = torch.nn.functional.layer_norm(torch.randn(S, m.P, 64), (64,))
    s0, a = torch.randn(S, 7), torch.randn(S, H, 7)
    fast, ref = m.rollout(z0, s0, a), m.rollout_full(z0, s0, a)
    assert fast.shape == (S, H, m.P, 64)
    assert torch.allclose(fast, ref, atol=1e-4), (fast - ref).abs().max()


def test_loss_is_finite_and_differentiable() -> None:
    m = _model().train()
    z = torch.nn.functional.layer_norm(torch.randn(2, 4, m.P, 64), (64,))
    out = m.loss(z, torch.randn(2, 3, 7), torch.randn(2, 4, 7))
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert m.predictor.action_encoder.weight.grad is not None
