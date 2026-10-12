"""Cross-entropy method over macro-action sequences, vectorised over samples.

The update follows the released V-JEPA 2-AC planner (``notebooks/utils/mpc_utils.py``): a diagonal Gaussian
per step of the horizon, initialised at zero mean and full-range standard deviation, top-k elites, and
momentum on both the mean and the standard deviation. Two options extend it: AR(1) temporal smoothing of the
sampling noise (``smooth`` = beta, 0 disables it; colored-noise CEM in the sense of Pinneri et al., 2020) and
an optional warm start from the previous plan shifted by one step. The executed plan is the final mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class CEMConfig:
    horizon: int = 10
    samples: int = 256
    iterations: int = 10
    elite_frac: float = 0.05
    momentum_mean: float = 0.15
    momentum_std: float = 0.15
    init_std: float | tuple = 1.0  # scalar, or one value per action dimension (data-informed prior)
    min_std: float = 0.02
    smooth: float = 0.0
    warm_start: bool = False
    low: float = -1.0
    high: float = 1.0

    @property
    def topk(self) -> int:
        return max(2, int(round(self.samples * self.elite_frac)))


class CEM:
    def __init__(self, cfg: CEMConfig, action_dim: int = 7, device: str = "cpu", seed: int = 0):
        self.cfg, self.action_dim, self.device = cfg, action_dim, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.prev_mean: torch.Tensor | None = None
        self.trace: list[dict] = []

    def reset(self) -> None:
        self.prev_mean = None

    def _noise(self, S: int, H: int) -> torch.Tensor:
        eps = torch.randn(S, H, self.action_dim, generator=self.gen, device=self.device)
        b = self.cfg.smooth
        if b > 0:
            out = [eps[:, 0]]
            for t in range(1, H):
                out.append(b * out[-1] + (1 - b * b) ** 0.5 * eps[:, t])
            eps = torch.stack(out, 1)
        return eps

    @torch.no_grad()
    def plan(self, cost_fn: Callable[[torch.Tensor], torch.Tensor], record: bool = False) -> torch.Tensor:
        """cost_fn maps actions (S, H, A) to costs (S,). Returns the planned sequence (H, A)."""
        c = self.cfg
        H, A = c.horizon, self.action_dim
        mean = torch.zeros(H, A, device=self.device)
        if c.warm_start and self.prev_mean is not None:
            mean = torch.cat([self.prev_mean[1:], torch.zeros(1, A, device=self.device)])
        std = torch.as_tensor(c.init_std, dtype=torch.float32, device=self.device).expand(H, A).clone()
        self.trace = []
        for it in range(c.iterations):
            acts = (mean + std * self._noise(c.samples, H)).clamp(c.low, c.high)
            costs = cost_fn(acts)
            idx = costs.topk(c.topk, largest=False).indices
            elite = acts[idx]
            mean = (1 - c.momentum_mean) * elite.mean(0) + c.momentum_mean * mean
            std = ((1 - c.momentum_std) * elite.std(0) + c.momentum_std * std).clamp_min(c.min_std)
            if record:
                self.trace.append({"mean": mean.clone().cpu(), "std": std.clone().cpu(),
                                   "best_cost": float(costs[idx[0]]), "elite_cost": float(costs[idx].mean())})
        self.prev_mean = mean
        return mean
