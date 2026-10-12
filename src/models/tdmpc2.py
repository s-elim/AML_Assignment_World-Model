"""TD-MPC2 (Hansen, Su and Wang, ICLR 2024), re-implemented at the paper's 5M-parameter configuration.

Kept from the paper: SimNorm latent (simplex dimension 8), LayerNorm + Mish MLPs of width 512, latent 512,
a learnable task embedding of width 96 concatenated to every head (multi-task model), discrete two-hot
regression over 101 symlog bins for reward and value, an ensemble of 5 Q-functions with 1% dropout and
Polyak-averaged targets (tau = 0.01), a Gaussian policy prior trained to maximise Q with an entropy bonus and
a running Q-scale, the joint loss 20 * consistency + 0.1 * reward + 0.1 * value with temporal weight
rho = 0.5 over a training horizon of 3, and MPPI planning (512 samples, 64 elites, 24 policy trajectories,
6 iterations, horizon 3, temperature 0.5) that warm-starts from the previous plan. The pixel encoder is the
paper's 4-layer CNN on 64x64 frames with a 3-frame stack.

Adapted: the paper trains online; here the model trains offline on the same 160 demonstrations as every
other model, with LIBERO's sparse success reward (1 on the macro step that reaches success, episodes
terminate there) and the 7-D proprioceptive state fused into the latent.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BINS, VMIN, VMAX = 101, -10.0, 10.0


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1 + x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


BIN_SIZE = (VMAX - VMIN) / (NUM_BINS - 1)


def two_hot(x: torch.Tensor) -> torch.Tensor:
    """(B, 1) scalar targets -> (B, NUM_BINS) two-hot in symlog space."""
    x = symlog(x).clamp(VMIN, VMAX).squeeze(1)
    idx = torch.floor((x - VMIN) / BIN_SIZE).long().clamp(0, NUM_BINS - 1)
    off = ((x - VMIN) / BIN_SIZE - idx.float()).unsqueeze(1)
    out = torch.zeros(x.size(0), NUM_BINS, device=x.device)
    out.scatter_(1, idx.unsqueeze(1), 1 - off)
    out.scatter_(1, (idx.unsqueeze(1) + 1) % NUM_BINS, off)
    return out


def two_hot_inv(logits: torch.Tensor) -> torch.Tensor:
    bins = torch.linspace(VMIN, VMAX, NUM_BINS, device=logits.device)
    return symexp((F.softmax(logits, -1) * bins).sum(-1, keepdim=True))


def soft_ce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(two_hot(target) * F.log_softmax(pred, -1)).sum(-1, keepdim=True)


class SimNorm(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        return F.softmax(x.view(*shp[:-1], -1, self.dim), -1).view(shp)


class NormedLinear(nn.Sequential):
    def __init__(self, i: int, o: int, dropout: float = 0.0, act: nn.Module | None = None):
        layers = [nn.Linear(i, o)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers += [nn.LayerNorm(o), act if act is not None else nn.Mish()]
        super().__init__(*layers)


def mlp(i: int, hidden: list[int], o: int, act: nn.Module | None = None, dropout: float = 0.0) -> nn.Sequential:
    dims = [i] + hidden
    layers = [NormedLinear(dims[j], dims[j + 1], dropout=dropout * (j == 0)) for j in range(len(hidden))]
    layers.append(NormedLinear(dims[-1], o, act=act) if act is not None else nn.Linear(dims[-1], o))
    return nn.Sequential(*layers)


class ShiftAug(nn.Module):
    """Random shift by up to ``pad`` pixels (DrQ), applied only in training."""

    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        n, _, h, w = x.shape
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device)[:h]
        ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base = torch.cat([ar, ar.transpose(1, 0)], 2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, frames: int = 3, latent: int = 512, state_dim: int = 7):
        super().__init__()
        self.aug = ShiftAug()
        self.conv = nn.Sequential(
            nn.Conv2d(3 * frames, 32, 7, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), nn.Flatten(),
        )
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.Mish())
        self.fuse = NormedLinear(512 + 64, latent, act=SimNorm())

    def forward(self, pix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """pix uint8 (B, 3*frames, 64, 64), state (B, 7) normalised -> (B, latent)."""
        x = self.aug(pix.float()) / 255.0 - 0.5
        return self.fuse(torch.cat([self.conv(x), self.state(state)], -1))


class TDMPC2(nn.Module):
    def __init__(self, num_tasks: int = 4, action_dim: int = 7, latent: int = 512, hidden: int = 512,
                 task_dim: int = 96, num_q: int = 5, frames: int = 3):
        super().__init__()
        self.A, self.latent, self.num_q = action_dim, latent, num_q
        self.task_emb = nn.Embedding(num_tasks, task_dim, max_norm=1)
        self.encoder = Encoder(frames, latent)
        za, zt = latent + action_dim + task_dim, latent + task_dim
        self.dynamics = mlp(za, [hidden, hidden], latent, act=SimNorm())
        self.reward = mlp(za, [hidden, hidden], NUM_BINS)
        self.pi = mlp(zt, [hidden, hidden], 2 * action_dim)
        self.Qs = nn.ModuleList(mlp(za, [hidden, hidden], NUM_BINS, dropout=0.01) for _ in range(num_q))
        for q in self.Qs:  # zero-init the value heads, as in the paper
            nn.init.zeros_(q[-1].weight)
            nn.init.zeros_(q[-1].bias)
        nn.init.zeros_(self.reward[-1].weight)
        self.target_Qs = copy.deepcopy(self.Qs).requires_grad_(False)
        self.log_std_min, self.log_std_max = -10.0, 2.0

    def task(self, x: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.task_emb(tid).expand(*x.shape[:-1], -1)], -1)

    def next(self, z, a, tid):
        return self.dynamics(self.task(torch.cat([z, a], -1), tid))

    def rew(self, z, a, tid):
        return self.reward(self.task(torch.cat([z, a], -1), tid))

    def Q(self, z, a, tid, mode: str = "min", target: bool = False):
        x = self.task(torch.cat([z, a], -1), tid)
        if mode == "all":
            return torch.stack([q(x) for q in self.Qs])
        qs = self.target_Qs if target else self.Qs
        i, j = torch.randperm(self.num_q)[:2].tolist()
        q1, q2 = two_hot_inv(qs[i](x)), two_hot_inv(qs[j](x))
        return torch.min(q1, q2) if mode == "min" else (q1 + q2) / 2

    def policy(self, z, tid):
        mu, log_std = self.pi(self.task(z, tid)).chunk(2, -1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mu)
        pre = mu + eps * log_std.exp()
        log_prob = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        a = torch.tanh(pre)
        log_prob = log_prob - torch.log(F.relu(1 - a.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return torch.tanh(mu), a, log_prob

    @torch.no_grad()
    def soft_update(self, tau: float = 0.01) -> None:
        for p, tp in zip(self.Qs.parameters(), self.target_Qs.parameters()):
            tp.lerp_(p, tau)


class RunningScale:
    """Running 5th-95th percentile range of Q, used to normalise the policy objective (TD-MPC2)."""

    def __init__(self, tau: float = 0.01):
        self.value, self.tau = None, tau

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().flatten()
        r = torch.quantile(x, 0.95) - torch.quantile(x, 0.05)
        r = r.clamp_min(1.0)
        self.value = r if self.value is None else self.value.lerp(r, self.tau)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / (self.value if self.value is not None else 1.0)


def tdmpc2_update(model: TDMPC2, opt, pi_opt, scale: RunningScale, batch: dict, gamma: float,
                  rho: float = 0.5, entropy_coef: float = 1e-4) -> dict[str, float]:
    """One TD-MPC2 update on a batch of sequences of length H+1 with validity masks."""
    pix, st, act, rew, done, valid, tid = (batch[k] for k in ["pix", "state", "action", "reward", "done", "valid", "task"])
    B, H1 = pix.shape[:2]
    H = H1 - 1
    with torch.no_grad():
        next_z = model.encoder(pix[:, 1:].flatten(0, 1), st[:, 1:].flatten(0, 1)).view(B, H, -1)
        _, pi_a, _ = model.policy(next_z, tid[:, None])
        td = rew + gamma * (1 - done) * model.Q(next_z, pi_a, tid[:, None], "min", target=True)
    z = model.encoder(pix[:, 0], st[:, 0])
    zs, cons = [z], 0.0
    for t in range(H):
        z = model.next(z, act[:, t], tid)
        w = valid[:, t] * rho**t
        cons = cons + (F.mse_loss(z, next_z[:, t], reduction="none").mean(-1, keepdim=True) * w).mean()
        zs.append(z)
    zs_t = torch.stack(zs[:-1], 1)
    qs = model.Q(zs_t, act, tid[:, None], "all")  # (num_q, B, H, bins)
    rp = model.rew(zs_t, act, tid[:, None])
    rl, vl = 0.0, 0.0
    for t in range(H):
        w = valid[:, t] * rho**t
        rl = rl + (soft_ce(rp[:, t], rew[:, t]) * w).mean()
        for q in range(model.num_q):
            vl = vl + (soft_ce(qs[q, :, t], td[:, t]) * w).mean()
    cons, rl, vl = cons / H, rl / H, vl / (H * model.num_q)
    loss = 20.0 * cons + 0.1 * rl + 0.1 * vl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
    opt.step()
    # Policy prior on detached latents.
    zd = zs_t.detach()
    _, a, logp = model.policy(zd, tid[:, None])
    q = model.Q(zd, a, tid[:, None], "avg")
    scale.update(q[:, 0])
    rho_w = (rho ** torch.arange(H, device=zd.device)).view(1, H, 1)
    pi_loss = ((entropy_coef * logp - scale(q)) * rho_w * valid).mean()
    pi_opt.zero_grad(set_to_none=True)
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.pi.parameters(), 20.0)
    pi_opt.step()
    model.soft_update()
    return {"loss": float(loss), "consistency": float(cons), "reward": float(rl), "value": float(vl), "pi": float(pi_loss)}


class MPPI:
    """TD-MPC2's MPPI planner over the learned latent model."""

    def __init__(self, model: TDMPC2, gamma: float, horizon: int = 3, iterations: int = 6, samples: int = 512,
                 elites: int = 64, pi_trajs: int = 24, temperature: float = 0.5, min_std: float = 0.05,
                 max_std: float = 2.0, seed: int = 0):
        self.m, self.gamma = model, gamma
        self.H, self.I, self.N, self.K, self.Np = horizon, iterations, samples, elites, pi_trajs
        self.temp, self.min_std, self.max_std = temperature, min_std, max_std
        self.prev = None
        torch.manual_seed(seed)

    def reset(self) -> None:
        self.prev = None

    @torch.no_grad()
    def value(self, z, acts, tid):
        G, disc = 0.0, 1.0
        for t in range(self.H):
            G = G + disc * two_hot_inv(self.m.rew(z, acts[:, t], tid))
            z = self.m.next(z, acts[:, t], tid)
            disc *= self.gamma
        _, a, _ = self.m.policy(z, tid)
        return G + disc * self.m.Q(z, a, tid, "avg")

    @torch.no_grad()
    def plan(self, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """z (1, latent), tid (1,) -> first action (A,)."""
        A, dev = self.m.A, z.device
        pi_acts = torch.empty(self.Np, self.H, A, device=dev)
        _z = z.expand(self.Np, -1)
        for t in range(self.H):
            _, pi_acts[:, t], _ = self.m.policy(_z, tid)
            _z = self.m.next(_z, pi_acts[:, t], tid)
        mean = torch.zeros(self.H, A, device=dev)
        if self.prev is not None:
            mean[:-1] = self.prev[1:]
        std = torch.full((self.H, A), self.max_std, device=dev)
        zN = z.expand(self.N, -1)
        for _ in range(self.I):
            r = (mean + std * torch.randn(self.N - self.Np, self.H, A, device=dev)).clamp(-1, 1)
            acts = torch.cat([pi_acts, r])
            val = self.value(zN, acts, tid).nan_to_num(0)
            idx = val[:, 0].topk(self.K).indices
            ev, ea = val[idx], acts[idx]
            score = torch.exp(self.temp * (ev - ev.max()))
            score = score / score.sum()
            mean = (score[:, :, None] * ea).sum(0)
            std = ((score[:, :, None] * (ea - mean).pow(2)).sum(0)).sqrt().clamp(self.min_std, self.max_std)
        j = torch.multinomial(score[:, 0], 1).item()
        self.prev = mean
        return ea[j, 0].clamp(-1, 1)
