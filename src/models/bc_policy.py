"""Behaviour cloning baselines: observation -> macro action, no predictive model.

The policy head outputs a diagonal Gaussian over the 7-D macro action and is trained by negative
log-likelihood of the demonstrated macro actions; the executed action is the mean. A learned task embedding
specifies which of the four tasks to perform. Backbones:

* ``resnet18``  ResNet-18 (ImageNet initialisation), trained end to end; the main BC baseline.
* ``dinov2_frozen`` / ``vjepa2_frozen``  cached frozen 8x8 token grids from the world models' encoders,
  attention-pooled by a learned query (representation-freezing ablation).
* ``dinov2_finetune``  DINOv2 ViT-S/14 trained end to end from its pretrained weights.

Optionally a second ResNet-18 encodes the wrist camera (dual-view ablation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import IMAGENET_MEAN, IMAGENET_STD


def _imnorm(x: torch.Tensor, size: int | None = None) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2).float() / 255.0 if x.dtype == torch.uint8 else x
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def random_shift(x: torch.Tensor, pad: int = 6) -> torch.Tensor:
    """Random-crop augmentation on uint8 (B, H, W, 3) images (pad and crop back)."""
    B, H, W, _ = x.shape
    xp = F.pad(x.permute(0, 3, 1, 2).float(), (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (oy[:, None] + torch.arange(H, device=x.device)[None]).clamp(max=H + 2 * pad - 1)
    cols = (ox[:, None] + torch.arange(W, device=x.device)[None]).clamp(max=W + 2 * pad - 1)
    out = xp[torch.arange(B)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]
    return out.permute(0, 2, 3, 1).round().to(torch.uint8)


class ResNetTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        self.net = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0)
        self.dim = self.net.num_features

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(_imnorm(img))


class DinoTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        self.net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224,
                                     num_classes=0, dynamic_img_size=True)
        self.dim = 384

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = self.net.forward_features(_imnorm(img, 224))[:, self.net.num_prefix_tokens :]
        return h.mean(1)


class TokenPool(nn.Module):
    """Learned-query attention pooling over cached frozen tokens (B, 64, D) -> (B, D')."""

    def __init__(self, in_dim: int, dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.dim = dim

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        x = self.proj(tok.float()) + self.pos
        return self.attn(self.q.expand(len(x), -1, -1), x, x)[0][:, 0]


class BCPolicy(nn.Module):
    def __init__(self, backbone: str = "resnet18", wrist: bool = False, num_tasks: int = 4, action_dim: int = 7,
                 state_dim: int = 7, token_dim: int | None = None):
        super().__init__()
        self.backbone, self.wrist = backbone, wrist
        if backbone == "resnet18":
            self.trunk = ResNetTrunk()
        elif backbone == "dinov2_finetune":
            self.trunk = DinoTrunk()
        elif backbone in ("dinov2_frozen", "vjepa2_frozen"):
            self.trunk = TokenPool(token_dim)
        else:
            raise ValueError(backbone)
        feat = self.trunk.dim
        if wrist:
            self.wrist_trunk = ResNetTrunk()
            feat += self.wrist_trunk.dim
        self.task_emb = nn.Embedding(num_tasks, 32)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(feat + 32 + 64, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                  nn.Linear(512, 2 * action_dim))

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: 'image' (uint8 images or cached tokens), 'state' (B, 7) normalised, 'task' (B,), opt. 'wrist'."""
        f = [self.trunk(obs["image"])]
        if self.wrist:
            f.append(self.wrist_trunk(obs["wrist"]))
        f += [self.task_emb(obs["task"]), self.state_mlp(obs["state"])]
        mu, log_std = self.head(torch.cat(f, -1)).chunk(2, -1)
        return mu, log_std.clamp(-5.0, 1.0)

    def nll(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        mu, log_std = self(obs)
        return (0.5 * ((action - mu) / log_std.exp()).pow(2) + log_std + 0.9189385).sum(-1).mean()

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self(obs)[0].clamp(-1, 1)
