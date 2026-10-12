"""TD-MPC2: two-hot encoding is invertible and one update plus one MPPI call run end to end."""

import torch

from src.models.tdmpc2 import MPPI, TDMPC2, RunningScale, two_hot, two_hot_inv, tdmpc2_update


def test_two_hot_roundtrip() -> None:
    x = torch.tensor([[0.0], [0.5], [1.0], [3.0], [-2.0]])
    logits = torch.log(two_hot(x) + 1e-12)
    assert torch.allclose(two_hot_inv(logits), x, atol=1e-3)


def test_update_and_plan() -> None:
    torch.manual_seed(0)
    m = TDMPC2(num_tasks=4, frames=3)
    B, H = 8, 3
    batch = {
        "pix": torch.randint(0, 255, (B, H + 1, 9, 64, 64), dtype=torch.uint8),
        "state": torch.randn(B, H + 1, 7), "action": torch.rand(B, H, 7) * 2 - 1,
        "reward": torch.zeros(B, H, 1), "done": torch.zeros(B, H, 1), "valid": torch.ones(B, H, 1),
        "task": torch.randint(0, 4, (B,)),
    }
    opt = torch.optim.Adam([p for n, p in m.named_parameters() if not n.startswith(("pi", "target_Qs"))], lr=3e-4)
    pi_opt = torch.optim.Adam(m.pi.parameters(), lr=3e-4)
    out = tdmpc2_update(m, opt, pi_opt, RunningScale(), batch, gamma=0.95)
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
    m.eval()
    with torch.no_grad():
        z = m.encoder(batch["pix"][:1, 0], batch["state"][:1, 0])
        a = MPPI(m, 0.95, samples=64, elites=8, pi_trajs=8, iterations=2).plan(z, torch.tensor([1]))
    assert a.shape == (7,) and a.abs().max() <= 1.0
