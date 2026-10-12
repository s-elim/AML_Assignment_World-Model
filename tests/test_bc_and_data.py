"""Behaviour-cloning head and macro-step data utilities."""

import numpy as np
import torch

from src.data.libero_data import Demo, StateNorm, clip_index, macro_action
from src.models.bc_policy import BCPolicy, random_shift


def test_bc_nll_decreases_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    m = BCPolicy("dinov2_frozen", token_dim=32)
    obs = {"image": torch.randn(16, 64, 32), "state": torch.randn(16, 7), "task": torch.randint(0, 4, (16,))}
    act = torch.rand(16, 7) * 2 - 1
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = float(m.nll(obs, act))
    for _ in range(50):
        opt.zero_grad()
        m.nll(obs, act).backward()
        opt.step()
    assert float(m.nll(obs, act)) < first
    assert m.act(obs).abs().max() <= 1.0


def test_random_shift_keeps_shape_and_dtype() -> None:
    x = torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8)
    y = random_shift(x)
    assert y.shape == x.shape and y.dtype == torch.uint8


def test_macro_actions_and_clips() -> None:
    a = np.arange(40, dtype=np.float32).reshape(20, 2).repeat(4, 1)[:, :7]
    assert np.allclose(macro_action(a, 0, 5), a[:5].mean(0))
    d = Demo("t", 0, 0, "train", np.zeros((20, 7)), a, np.zeros((20, 79)), "")
    idx = clip_index([d], T=4, k=5, split="train")
    assert idx[0] == (0, 0) and idx[-1] == (0, 20 - 3 * 5 - 1)
    norm = StateNorm.fit([Demo("t", 0, 0, "train", np.random.RandomState(0).randn(50, 7), a, np.zeros(1), "")])
    assert np.allclose(norm(norm.mean), 0.0)
