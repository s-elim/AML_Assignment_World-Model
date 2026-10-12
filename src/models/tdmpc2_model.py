from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDMPC2Model(nn.Module):
    """
    TD-MPC2 Task-Driven World Model (Hansen et al., ICLR 2024).
    Maps high-dimensional visual observations to a compact latent state,
    learning action-conditioned dynamics and reward prediction for continuous control.
    """

    def __init__(
        self,
        img_size: int = 224,
        latent_dim: int = 64,
        action_dim: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Compact convolutional visual encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Latent dynamics model: z_{t+1} = T(z_t, a_t)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Q-value prediction head
        self.q_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.max() > 1.5:
            obs = obs / 255.0
        z = self.encoder(obs)
        return z

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([latent, action], dim=-1)
        next_z = self.dynamics(inp)
        return next_z

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([latent, action], dim=-1)
        return self.reward_head(inp)

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        reward_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs_seq.shape
        flat_obs = obs_seq.view(B * T, C, H, W)
        all_z = self.encode(flat_obs).view(B, T, self.latent_dim)

        dyn_loss = torch.tensor(0.0, device=obs_seq.device)
        reward_loss = torch.tensor(0.0, device=obs_seq.device)

        for t in range(T - 1):
            curr_z = all_z[:, t]
            action_t = act_seq[:, t]
            target_z = all_z[:, t + 1].detach()

            pred_z = self.predict(curr_z, action_t)
            dyn_loss = dyn_loss + F.mse_loss(pred_z, target_z)

            if reward_seq is not None:
                target_r = reward_seq[:, t].unsqueeze(-1)
                pred_r = self.predict_reward(curr_z, action_t)
                reward_loss = reward_loss + F.mse_loss(pred_r, target_r)

        dyn_loss = dyn_loss / max(1, T - 1)
        if reward_seq is not None:
            reward_loss = reward_loss / max(1, T - 1)
        total_loss = dyn_loss + 0.5 * reward_loss

        return {"loss": total_loss, "dyn_loss": dyn_loss, "reward_loss": reward_loss}
