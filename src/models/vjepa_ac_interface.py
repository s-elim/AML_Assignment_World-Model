from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.norm(x)
        return x


class VJEPA2ACModel(nn.Module):
    """
    Action-Conditioned V-JEPA (V-JEPA 2-AC) Latent World Model.
    Learns predictive dynamics directly in self-supervised patch token representation space
    without decoding back to raw pixels.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 128,
        predictor_embed_dim: int = 128,
        predictor_depth: int = 2,
        action_dim: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.action_dim = action_dim

        # Visual patch encoder (representing frozen or pretrained ViT backbone)
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Action conditioner and causal predictor
        self.action_encoder = nn.Linear(action_dim, predictor_embed_dim)
        self.predictor_in_proj = nn.Linear(embed_dim, predictor_embed_dim)

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=4,
            dim_feedforward=predictor_embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=predictor_depth)
        self.predictor_out_proj = nn.Linear(predictor_embed_dim, embed_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode visual observation into patch token representations.
        :param obs: (B, C, H, W) or (C, H, W)
        :return: (B, N, D)
        """
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        # Normalize uint8 to [0, 1] if needed
        if obs.max() > 1.5:
            obs = obs / 255.0

        x = self.patch_embed(obs) + self.pos_embed
        z = self.encoder(x)
        z = F.layer_norm(z, (z.size(-1),))
        return z

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent representation given current representation and action.
        :param latent: (B, N, D)
        :param action: (B, d_a)
        :return: (B, N, D)
        """
        B, N, D = latent.shape
        z_proj = self.predictor_in_proj(latent)  # (B, N, D_pred)
        a_emb = self.action_encoder(action).unsqueeze(1)  # (B, 1, D_pred)

        # Interleave action token as condition token
        seq = torch.cat([a_emb, z_proj], dim=1)  # (B, N+1, D_pred)
        out = self.predictor(seq)

        # Split out patch tokens and project back
        next_z = out[:, 1:, :]
        next_z = self.predictor_out_proj(next_z)
        next_z = F.layer_norm(next_z, (next_z.size(-1),))
        return next_z

    def compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute V-JEPA L2 latent prediction loss across sequence.
        :param obs_seq: (B, T, C, H, W)
        :param act_seq: (B, T, d_a)
        """
        B, T, C, H, W = obs_seq.shape
        # Flatten time into batch to encode all frames
        flat_obs = obs_seq.view(B * T, C, H, W)
        with torch.no_grad():
            all_z = self.encode(flat_obs).view(B, T, -1, self.embed_dim)

        total_loss = torch.tensor(0.0, device=obs_seq.device)
        # Predict one-step ahead
        for t in range(T - 1):
            curr_z = all_z[:, t]
            action_t = act_seq[:, t]
            target_z = all_z[:, t + 1].detach()

            pred_next_z = self.predict(curr_z, action_t)
            step_loss = F.mse_loss(pred_next_z, target_z)
            total_loss = total_loss + step_loss

        total_loss = total_loss / max(1, T - 1)
        return {"loss": total_loss}
