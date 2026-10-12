import os
import sys
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

from src.models.vjepa_ac_interface import VJEPA2ACModel
from src.models.dino_wm import DINOWMModel
from src.models.tdmpc2_model import TDMPC2Model
from src.models.bc_policy import BCPolicyModel
from src.utils.seed import set_seed
from src.utils.metrics import compute_latent_mse


class TrajectoryDataset(Dataset):
    """Dataset of robot demonstration trajectories sampling temporal sub-windows."""

    def __init__(self, npz_path: str, seq_len: int = 4):
        data = np.load(npz_path, allow_pickle=True)
        self.obs = data["observations"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.seq_len = seq_len

        # Pre-index valid windows with padding fallback
        self.indices = []
        for ep_idx in range(len(self.actions)):
            ep_len = len(self.actions[ep_idx])
            if ep_len >= seq_len:
                for t in range(ep_len - seq_len + 1):
                    self.indices.append((ep_idx, t, False))
            else:
                self.indices.append((ep_idx, 0, True))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_t, is_padded = self.indices[idx]
        
        if not is_padded:
            end_t = start_t + self.seq_len
            obs_window = self.obs[ep_idx][start_t:end_t]
            act_window = self.actions[ep_idx][start_t:end_t]
            rew_window = self.rewards[ep_idx][start_t:end_t]
        else:
            raw_obs = self.obs[ep_idx]
            raw_act = self.actions[ep_idx]
            raw_rew = self.rewards[ep_idx]
            pad_len = self.seq_len - len(raw_act)
            
            # Repeat last element
            obs_pad = np.repeat(raw_obs[-1:], pad_len, axis=0)
            obs_window = np.concatenate([raw_obs[:len(raw_act)], obs_pad], axis=0)
            
            act_pad = np.zeros((pad_len, raw_act.shape[-1]), dtype=np.float32)
            act_window = np.concatenate([raw_act, act_pad], axis=0)
            
            rew_pad = np.repeat(raw_rew[-1:], pad_len, axis=0)
            rew_window = np.concatenate([raw_rew, rew_pad], axis=0)

        obs_tensor = torch.from_numpy(obs_window).permute(0, 3, 1, 2).float() / 255.0
        act_tensor = torch.from_numpy(act_window).float()
        rew_tensor = torch.from_numpy(rew_window).float()

        return {
            "observations": obs_tensor,
            "actions": act_tensor,
            "rewards": rew_tensor,
        }

        # Permute images to (T, 3, H, W) and convert to float [0, 1]
        obs_tensor = torch.from_numpy(obs_window).permute(0, 3, 1, 2).float() / 255.0
        act_tensor = torch.from_numpy(act_window).float()
        rew_tensor = torch.from_numpy(rew_window).float()

        return {
            "observations": obs_tensor,
            "actions": act_tensor,
            "rewards": rew_tensor,
        }


def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    m_type = config.get("model_type", "vjepa2_ac")
    if m_type == "vjepa2_ac":
        return VJEPA2ACModel(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 128),
            predictor_embed_dim=config.get("predictor_embed_dim", 128),
            predictor_depth=config.get("predictor_depth", 4),
            action_dim=config.get("action_dim", 4),
        )
    elif m_type == "dino_wm":
        return DINOWMModel(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 128),
            action_dim=config.get("action_dim", 4),
        )
    elif m_type == "tdmpc2":
        return TDMPC2Model(
            img_size=config.get("img_size", 224),
            latent_dim=config.get("latent_dim", 64),
            action_dim=config.get("action_dim", 4),
        )
    elif m_type == "bc_policy":
        return BCPolicyModel(
            img_size=config.get("img_size", 224),
            embed_dim=config.get("embed_dim", 64),
            action_dim=config.get("action_dim", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {m_type}")


def train_model(
    config_path: str,
    data_dir: str,
    output_dir: str = "results",
    device: str = "cpu",
    seed: int = 42,
) -> str:
    set_seed(seed)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    task_name = Path(data_dir).name
    model_type = config.get("model_type", "vjepa2_ac")
    ckpt_dir = Path(output_dir) / "checkpoints"
    log_dir = Path(output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seq_len = config.get("sequence_length", 8)
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 10)
    lr = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.01))

    train_dataset = TrajectoryDataset(os.path.join(data_dir, "train.npz"), seq_len=seq_len)
    val_dataset = TrajectoryDataset(os.path.join(data_dir, "val.npz"), seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_ckpt_path = str(ckpt_dir / f"{model_type}_{task_name}_best.pt")
    history = {"train_loss": [], "val_loss": []}

    print(f"--- Training {model_type} on {task_name} ({epochs} epochs) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            obs = batch["observations"].to(device)  # (B, T, C, H, W)
            acts = batch["actions"].to(device)      # (B, T, 4)
            rews = batch["rewards"].to(device)      # (B, T)

            optimizer.zero_grad()
            if model_type == "bc_policy":
                loss_dict = model.compute_loss(obs[:, 0], acts[:, 0])
            elif model_type == "tdmpc2":
                loss_dict = model.compute_loss(obs, acts, rews)
            else:
                loss_dict = model.compute_loss(obs, acts)

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observations"].to(device)
                acts = batch["actions"].to(device)
                rews = batch["rewards"].to(device)

                if model_type == "bc_policy":
                    val_dict = model.compute_loss(obs[:, 0], acts[:, 0])
                elif model_type == "tdmpc2":
                    val_dict = model.compute_loss(obs, acts, rews)
                else:
                    val_dict = model.compute_loss(obs, acts)

                val_losses.append(val_dict["loss"].item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train: {mean_train:.5f} | Val: {mean_val:.5f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": mean_val,
                },
                best_ckpt_path,
            )

    log_path = log_dir / f"{model_type}_{task_name}_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action-conditioned world model.")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory (e.g. data/reach)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
