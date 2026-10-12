"""DINO-WM: frame-causal attention and rollout shapes."""

import torch

from src.models.dino_wm import DinoWM


def test_frame_causal_mask_blocks_future() -> None:
    torch.manual_seed(0)
    m = DinoWM(visual_dim=32, patches=4, depth=2, heads=2, mlp=64).eval()
    x = torch.randn(1, 3, 4, m.dim)
    y = x.clone()
    y[:, 2] += 1.0  # change only the last frame
    with torch.no_grad():
        a, b = m._predict(x), m._predict(y)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-5)
    assert not torch.allclose(a[:, 2], b[:, 2])


def test_rollout_shape_and_loss() -> None:
    m = DinoWM(visual_dim=32, patches=4, depth=2, heads=2, mlp=64)
    out = m.loss(torch.randn(2, 4, 4, 32), torch.randn(2, 4, 7), torch.randn(2, 4, 35))
    assert torch.isfinite(out["loss"])
    with torch.no_grad():
        z = m.rollout(torch.randn(3, 2, 4, 32), torch.randn(3, 2, 7), torch.randn(3, 1, 35), torch.randn(3, 5, 7))
    assert z.shape == (3, 5, 4, 32)
