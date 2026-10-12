"""DINO-WM (Zhou et al., ICML 2025) at reduced scale.

Kept from the paper and its released code: frozen DINOv2 ViT-S/14 patch tokens; 10-D action and
proprioception embeddings tiled over the patches and concatenated to every patch token; a ViT predictor with
learned positional embeddings and a frame-causal mask over ``num_hist`` frames; teacher-forced next-frame
prediction with an MSE loss on the visual and proprioceptive parts of the embedding (detached targets); and an
action input that concatenates the ``frameskip`` primitive actions of one macro step.

Reduced to the size of the V-JEPA 2-AC predictor: 6 blocks, 6 heads of width 64, MLP width 1536 and 8x8 tokens
per frame (release: 16 heads, MLP 2048, 16x16 tokens).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, inner * 3, bias=False)
        self.out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, N, -1))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoWM(nn.Module):
    def __init__(
        self,
        visual_dim: int = 384,
        action_dim: int = 7,
        state_dim: int = 7,
        frameskip: int = 5,
        num_hist: int = 3,
        patches: int = 64,
        emb: int = 10,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        mlp: int = 1536,
    ):
        super().__init__()
        self.visual_dim, self.emb, self.num_hist, self.P, self.frameskip = visual_dim, emb, num_hist, patches, frameskip
        self.dim = visual_dim + 2 * emb
        self.action_encoder = nn.Linear(action_dim * frameskip, emb)
        self.proprio_encoder = nn.Linear(state_dim, emb)
        self.pos = nn.Parameter(torch.randn(1, num_hist * patches, self.dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.ModuleList([_Attention(self.dim, heads, dim_head), _FeedForward(self.dim, mlp)]) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(self.dim)
        frame = torch.arange(num_hist).repeat_interleave(patches)
        self.register_buffer("mask", frame[:, None] >= frame[None, :], persistent=False)

    def embed(self, z: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z (B, T, P, Dv), s (B, T, 7), a (B, T, 7*frameskip) -> (B, T, P, Dv + 2*emb)."""
        P = z.size(2)
        ps = self.proprio_encoder(s).unsqueeze(2).expand(-1, -1, P, -1)
        pa = self.action_encoder(a).unsqueeze(2).expand(-1, -1, P, -1)
        return torch.cat([z, ps, pa], -1)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T<=num_hist, P, dim) -> predicted next-frame embeddings (B, T, P, dim)."""
        B, T, P, D = x.shape
        h = x.flatten(1, 2) + self.pos[:, : T * P]
        mask = self.mask[: T * P, : T * P]
        for attn, ff in self.layers:
            h = h + attn(h, mask)
            h = h + ff(h)
        return self.norm(h).view(B, T, P, D)

    def loss(self, z: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """z (B, num_hist+1, P, Dv), s (B, num_hist+1, 7), a (B, num_hist+1, 7*frameskip)."""
        x = self.embed(z, s, a)
        pred = self._predict(x[:, :-1])
        tgt = x[:, 1:].detach()
        dv = self.visual_dim
        z_loss = F.mse_loss(pred[..., :dv], tgt[..., :dv])
        p_loss = F.mse_loss(pred[..., dv : dv + self.emb], tgt[..., dv : dv + self.emb])
        return {"loss": z_loss + p_loss, "visual": z_loss, "proprio": p_loss}

    @torch.no_grad()
    def rollout(self, z_hist: torch.Tensor, s_hist: torch.Tensor, a_hist: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imagine H frames autoregressively over a sliding window of ``num_hist`` frames.

        z_hist (S, h, P, Dv), s_hist (S, h, 7) and a_hist (S, h-1, 7*fs) hold the observed history (h >= 1);
        actions (S, H, 7) are macro actions, expanded to the frameskip concatenation by a zero-order hold.
        Returns the visual part of the imagined frames, (S, H, P, Dv).
        """
        S, H = actions.shape[:2]
        acts = actions.repeat(1, 1, self.frameskip)  # zero-order hold -> (S, H, 7*fs)
        a_all = torch.cat([a_hist, acts], 1)
        x = self.embed(z_hist, s_hist, a_all[:, : z_hist.size(1)])
        outs = []
        for t in range(H):
            window = x[:, -self.num_hist :]
            nxt = self._predict(window)[:, -1]  # (S, P, dim): visual + proprio + action slots
            outs.append(nxt[..., : self.visual_dim])
            idx = z_hist.size(1) + t  # action slot of the new frame
            a_next = a_all[:, idx] if idx < a_all.size(1) else torch.zeros_like(acts[:, 0])
            pa = self.action_encoder(a_next).unsqueeze(1).expand(-1, self.P, -1)
            nxt = torch.cat([nxt[..., : self.visual_dim + self.emb], pa], -1)
            x = torch.cat([x, nxt.unsqueeze(1)], 1)
        return torch.stack(outs, 1)
