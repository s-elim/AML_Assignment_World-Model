from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.vjepa_ac_interface import PatchEmbed


class DINOWMModel(nn.Module):
    """
    DINO-WM Latent World Model (Zhou et al., ICML 2025).
    Predicts next-frame latent patch representations in DINOv2 feature space.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 128,
        action_dim: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        # DINOv2-style feature extractor
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Transition MLP predictor: takes (z_t, a_t) and outputs residual delta_z
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.transition_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.max() > 1.5:
            obs = obs / 255.0

        x = self.patch_embed(obs) + self.pos_embed
        z = self.backbone(x)
        z = F.layer_norm(z, (z.size(-1),))
        return z

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, N, D = latent.shape
        # Broadcast action across all patch tokens
        a_emb = self.action_proj(action).unsqueeze(1).repeat(1, N, 1)  # (B, N, D)
        inp = torch.cat([latent, a_emb], dim=-1)
        delta_z = self.transition_net(inp)
        next_z = latent + delta_z
        next_z = F.layer_norm(next_z, (next_z.size(-1),))
        return next_z

    def compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs_seq.shape
        flat_obs = obs_seq.view(B * T, C, H, W)
        with torch.no_grad():
            all_z = self.encode(flat_obs).view(B, T, -1, self.embed_dim)

        total_loss = torch.tensor(0.0, device=obs_seq.device)
        for t in range(T - 1):
            curr_z = all_z[:, t]
            action_t = act_seq[:, t]
            target_z = all_z[:, t + 1].detach()

            pred_next_z = self.predict(curr_z, action_t)
            step_loss = F.mse_loss(pred_next_z, target_z)
            total_loss = total_loss + step_loss

        total_loss = total_loss / max(1, T - 1)
        return {"loss": total_loss}
