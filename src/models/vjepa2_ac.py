"""Action-conditioned V-JEPA 2 (V-JEPA 2-AC; Assran et al., 2025) at reduced scale.

The predictor is the released ``VisionTransformerPredictorAC``, vendored unmodified: each frame contributes an
action token, a proprioceptive-state token and its patch tokens, attention is block-causal over frames, and
positions use 3-D RoPE. Training follows ``app/vjepa_droid/train.py``: an L1 teacher-forcing loss over the
clip plus an L1 loss on a two-step autoregressive rollout, against layer-normalised targets from the frozen
encoder.

Differences from the release, forced by the compute budget or by LIBERO: 6 predictor blocks of width 384
(release: 24 blocks of width 1024 on ViT-g features), an 8x8 token grid from a frozen ViT-L, and actions that
are LIBERO OSC commands rather than measured pose deltas. The state token of an imagined frame is therefore
advanced by a linear map fitted on the training set (``StateIntegrator``) instead of by pose composition.

``rollout`` caches keys and values across steps. Block-causal attention makes earlier frames independent of
later ones, so the cache is exact (checked against the full re-forward in ``tests/test_vjepa2_ac.py``) and an
H-step rollout processes O(H) frames instead of O(H^2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.third_party.vjepa2.models.ac_predictor import vit_ac_predictor
from src.third_party.vjepa2.models.utils.modules import rotate_queries_or_keys

COND_TOKENS = 2  # action token + state token per frame


class StateIntegrator(nn.Module):
    """Linear map from (normalised state, macro action) to the normalised state change over one macro step."""

    def __init__(self, state_dim: int = 7, action_dim: int = 7):
        super().__init__()
        self.register_buffer("W", torch.zeros(state_dim + action_dim + 1, state_dim))

    @torch.no_grad()
    def fit(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> float:
        x = torch.cat([s, a, torch.ones(len(s), 1)], 1).double()
        y = (s_next - s).double()
        sol = torch.linalg.lstsq(x, y).solution
        self.W.copy_(sol.float())
        resid = y - x @ sol
        return float(1.0 - resid.pow(2).sum() / (y - y.mean(0)).pow(2).sum())  # R^2

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a, torch.ones_like(s[..., :1])], -1)
        return s + x @ self.W


class VJEPA2AC(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        pred_dim: int = 384,
        depth: int = 6,
        grid: int = 8,
        max_frames: int = 32,
        action_dim: int = 7,
        auto_steps: int = 2,
    ):
        super().__init__()
        self.grid, self.P, self.embed_dim, self.auto_steps = grid, grid * grid, embed_dim, auto_steps
        self.predictor = vit_ac_predictor(
            img_size=(grid * 16, grid * 16), patch_size=16, num_frames=max_frames, tubelet_size=1,
            embed_dim=embed_dim, predictor_embed_dim=pred_dim, depth=depth, num_heads=pred_dim // 64,
            action_embed_dim=action_dim, use_rope=True, is_frame_causal=True, use_silu=False, wide_silu=True,
        )
        self.integrator = StateIntegrator(action_dim, action_dim)

    # ------------------------------------------------------------------ training
    def _step(self, z: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """z (B, T*P, D) context tokens, a (B, T, 7), s (B, T, 7) -> next-frame tokens (B, T*P, D)."""
        out = self.predictor(z, a, s)
        return F.layer_norm(out, (out.size(-1),))

    def loss(self, z: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> dict[str, torch.Tensor]:
        """z (B, T, P, D) encoded clip, a (B, T-1, 7) actions between frames, s (B, T, 7) states."""
        B, T, P, D = z.shape
        h = z.flatten(1, 2)
        z_tf = self._step(h[:, : (T - 1) * P], a, s[:, :-1])
        jloss = (z_tf - h[:, P:]).abs().mean()
        ctx = torch.cat([h[:, :P], z_tf[:, :P]], 1)
        for n in range(1, self.auto_steps):
            nxt = self._step(ctx, a[:, : n + 1], s[:, : n + 1])[:, -P:]
            ctx = torch.cat([ctx, nxt], 1)
        z_ar = ctx[:, P:]
        sloss = (z_ar - h[:, P : P + z_ar.size(1)]).abs().mean()
        return {"loss": jloss + sloss, "teacher_forcing": jloss, "rollout": sloss}

    # ------------------------------------------------------------------ planning
    def _rope_tables(self, attn, t: int, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin tables (P+2, head_dim) for one frame block at frame index t.

        Reproduces ``rotate_queries_or_keys`` exactly (same op order and dtype, including the release's
        duplicated-frequency layout): time on the first ``d_dim`` channels of every token, row and column on the
        next two segments of the patch tokens, and identity (cos 1, sin 0) on the remaining channels.
        """
        hd, dd, hh, ww = attn.head_dim, attn.d_dim, attn.h_dim, attn.w_dim
        ids = torch.arange(self.P, device=device)

        def seg(pos: torch.Tensor, D: int):
            omega = torch.arange(D // 2, dtype=dtype, device=device)
            omega /= D / 2.0
            omega = 1.0 / 10000**omega
            freq = torch.einsum("..., f -> ... f", pos.to(dtype), omega)
            return freq.cos().repeat(1, 2), freq.sin().repeat(1, 2)

        def table(parts, n):
            cos = [c for c, _ in parts]
            sin = [s_ for _, s_ in parts]
            used = sum(c.shape[-1] for c in cos)
            if used < hd:
                cos.append(torch.ones(n, hd - used, dtype=dtype, device=device))
                sin.append(torch.zeros(n, hd - used, dtype=dtype, device=device))
            return torch.cat(cos, -1), torch.cat(sin, -1)

        tc = torch.full((COND_TOKENS,), float(t), device=device)
        cond = table([seg(tc, dd)], COND_TOKENS)
        pt = torch.full((self.P,), float(t), device=device)
        ph = (ids // self.grid).float() * (attn.grid_size / self.grid)
        pw = (ids % self.grid).float() * (attn.grid_size / self.grid)
        patch = table([seg(pt, dd), seg(ph, hh), seg(pw, ww)], self.P)
        return torch.cat([cond[0], patch[0]]), torch.cat([cond[1], patch[1]])

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        y = x.unflatten(-1, (-1, 2))
        y1, y2 = y.unbind(dim=-1)
        y = torch.stack((-y2, y1), dim=-1).flatten(-2)
        return (x * cos) + (y * sin)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imagine H frames. z0 (S, P, D), s0 (S, 7) normalised, actions (S, H, 7) -> (S, H, P, D).

        Keys and values of earlier frames are written once into preallocated buffers (block-causal attention
        makes them final), so each step processes only the P+2 tokens of the new frame.
        """
        pred = self.predictor
        S, H = actions.shape[:2]
        N = self.P + COND_TOKENS
        blocks = pred.predictor_blocks
        a0 = blocks[0].attn
        K = V = None
        tables: list[tuple[torch.Tensor, torch.Tensor]] = []
        z, s, outs = z0, s0, []
        for t in range(H):
            x = torch.cat(
                [pred.action_encoder(actions[:, t]).unsqueeze(1), pred.state_encoder(s).unsqueeze(1),
                 pred.predictor_embed(z)], 1
            )
            for i, blk in enumerate(blocks):
                attn = blk.attn
                qkv = attn.qkv(blk.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # (S, heads, N, hd)
                if K is None:
                    K = [torch.empty(S, attn.num_heads, H * N, attn.head_dim, dtype=k.dtype, device=k.device)
                         for _ in blocks]
                    V = [torch.empty_like(K[0]) for _ in blocks]
                if len(tables) == t:
                    tables.append(self._rope_tables(a0, t, q.dtype, q.device))
                cos, sin = tables[t]
                q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
                K[i][:, :, t * N : (t + 1) * N] = k
                V[i][:, :, t * N : (t + 1) * N] = v
                y = F.scaled_dot_product_attention(q, K[i][:, :, : (t + 1) * N], V[i][:, :, : (t + 1) * N])
                x = x + attn.proj(y.transpose(1, 2).reshape(S, N, -1))
                x = x + blk.mlp(blk.norm2(x))
            z = pred.predictor_proj(pred.predictor_norm(x[:, COND_TOKENS:]))
            z = F.layer_norm(z, (z.size(-1),))
            outs.append(z)
            s = self.integrator(s, actions[:, t])
        return torch.stack(outs, 1)

    @torch.no_grad()
    def rollout_full(self, z0: torch.Tensor, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Reference implementation: re-run the whole clip each step, as the released CEM planner does."""
        S, H = actions.shape[:2]
        ctx, s, states, outs = z0, s0, [s0], []
        for t in range(H):
            st = torch.stack(states, 1)
            nxt = self._step(ctx, actions[:, : t + 1], st)[:, -self.P :]
            outs.append(nxt)
            ctx = torch.cat([ctx, nxt], 1)
            s = self.integrator(s, actions[:, t])
            states.append(s)
        return torch.stack(outs, 1)
