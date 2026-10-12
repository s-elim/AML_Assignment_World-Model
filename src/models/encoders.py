"""Frozen pretrained visual encoders shared by the two latent world models.

Both encoders map a 128x128 RGB frame to an 8x8 grid of patch tokens, so the two world models see the same
token budget and differ only in representation and predictor:

* V-JEPA 2 ViT-L/16 (Assran et al., 2025; released ``vitl.pt`` target encoder, 304M parameters). Frames are
  upsampled to 256x256 and duplicated into one two-frame tubelet, as the V-JEPA 2-AC training code encodes
  single frames, giving 16x16 tokens of width 1024.
* DINOv2 ViT-S/14 (Oquab et al., 2024; ``vit_small_patch14_dinov2.lvd142m``, 21.6M parameters), the DINO-WM
  encoder. Frames are upsampled to 224x224, giving 16x16 patch tokens of width 384 (CLS token dropped).

Both 16x16 grids are average-pooled 2x2 to 8x8 to keep CEM planning affordable, and every token is
layer-normalised without affine parameters, as in V-JEPA 2-AC (``normalize_reps``).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = os.path.join(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "hub", "checkpoints")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRID = 8


def _normalise(frames: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 (B, H, W, 3) or float (B, 3, H, W) in [0, 1] -> ImageNet-normalised (B, 3, size, size)."""
    if frames.dtype == torch.uint8:
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
    frames = F.interpolate(frames, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean) / std


def _pool_grid(tokens: torch.Tensor, side: int) -> torch.Tensor:
    """(B, side*side, D) -> (B, GRID*GRID, D) by average pooling."""
    b, n, d = tokens.shape
    assert n == side * side, (n, side)
    x = tokens.transpose(1, 2).reshape(b, d, side, side)
    x = F.adaptive_avg_pool2d(x, GRID)
    return x.flatten(2).transpose(1, 2)


class VJEPA2Encoder(nn.Module):
    name = "vjepa2"
    dim = 1024
    input_size = 256

    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        from src.third_party.vjepa2.models import vision_transformer as vit

        self.net = vit.vit_large(
            patch_size=16, img_size=(256, 256), num_frames=64, tubelet_size=2, use_sdpa=True,
            use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
        )
        path = checkpoint or os.path.join(CKPT_DIR, "vjepa2_vitl_target_encoder.pt")
        if not os.path.exists(path):
            path = self._extract_target_encoder(path)
        self.net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        self.eval().requires_grad_(False)

    @staticmethod
    def _extract_target_encoder(out: str) -> str:
        """Pull the EMA target encoder out of the released 5.1 GB ``vitl.pt`` training checkpoint."""
        src = os.path.join(CKPT_DIR, "vitl.pt")
        if not os.path.exists(src):
            torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/vjepa2/vitl.pt", src)
        sd = torch.load(src, map_location="cpu", weights_only=False)["target_encoder"]
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        torch.save(sd, out)
        return out

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = _normalise(frames, self.input_size)
        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B, 3, 2, H, W): one tubelet per frame
        h = self.net(x)  # (B, 256, 1024)
        h = _pool_grid(h, self.input_size // 16)
        return F.layer_norm(h, (h.size(-1),))


class DINOv2Encoder(nn.Module):
    name = "dinov2"
    dim = 384
    input_size = 224

    def __init__(self):
        super().__init__()
        import timm

        self.net = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0, dynamic_img_size=True
        )
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = _normalise(frames, self.input_size)
        h = self.net.forward_features(x)[:, self.net.num_prefix_tokens :]  # (B, 256, 384)
        h = _pool_grid(h, self.input_size // 14)
        return F.layer_norm(h, (h.size(-1),))


def build_encoder(name: str) -> nn.Module:
    if name == "vjepa2":
        return VJEPA2Encoder()
    if name == "dinov2":
        return DINOv2Encoder()
    raise ValueError(name)
