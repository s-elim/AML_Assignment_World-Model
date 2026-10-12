from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicyModel(nn.Module):
    """
    Direct Behavioral Cloning Policy Baseline (Chi et al., RSS 2023).
    Maps raw visual observations directly to low-level continuous actions without
    latent dynamics modeling or forward planning.
    """

    def __init__(
        self,
        img_size: int = 224,
        embed_dim: int = 64,
        action_dim: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Normalized action space [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.max() > 1.5:
            obs = obs / 255.0

        feat = self.encoder(obs)
        action = self.policy_head(feat)
        # Scale to max delta xyz (0.05) and gripper range
        scaled_action = action.clone()
        scaled_action[:, :3] = scaled_action[:, :3] * 0.05
        return scaled_action

    def compute_loss(self, obs: torch.Tensor, target_action: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_action = self.forward(obs)
        loss = F.mse_loss(pred_action, target_action)
        return {"loss": loss}
