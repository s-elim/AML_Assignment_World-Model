import torch
import pytest

from src.models.vjepa_ac_interface import VJEPA2ACModel
from src.models.dino_wm import DINOWMModel
from src.models.tdmpc2_model import TDMPC2Model
from src.models.bc_policy import BCPolicyModel


def test_vjepa2_ac_shapes_and_forward():
    model = VJEPA2ACModel(
        img_size=224,
        patch_size=16,
        embed_dim=128,
        predictor_embed_dim=128,
        predictor_depth=2,
        action_dim=4,
    )
    obs = torch.randn(2, 3, 224, 224)
    action = torch.randn(2, 4)

    # Encode
    z = model.encode(obs)
    assert z.ndim == 3  # (B, N, D)
    assert z.shape[0] == 2
    assert z.shape[1] == (224 // 16) ** 2
    assert z.shape[2] == 128

    # Predict
    z_next = model.predict(z, action)
    assert z_next.shape == z.shape

    # Loss
    obs_seq = torch.randn(2, 5, 3, 224, 224)
    act_seq = torch.randn(2, 5, 4)
    loss_dict = model.compute_loss(obs_seq, act_seq)
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() >= 0.0


def test_dino_wm_shapes_and_forward():
    model = DINOWMModel(
        img_size=224,
        patch_size=16,
        embed_dim=128,
        action_dim=4,
    )
    obs = torch.randn(2, 3, 224, 224)
    action = torch.randn(2, 4)

    z = model.encode(obs)
    assert z.ndim == 3
    assert z.shape[0] == 2
    assert z.shape[2] == 128

    z_next = model.predict(z, action)
    assert z_next.shape == z.shape

    obs_seq = torch.randn(2, 5, 3, 224, 224)
    act_seq = torch.randn(2, 5, 4)
    loss_dict = model.compute_loss(obs_seq, act_seq)
    assert "loss" in loss_dict


def test_tdmpc2_shapes_and_forward():
    model = TDMPC2Model(
        img_size=224,
        latent_dim=64,
        action_dim=4,
    )
    obs = torch.randn(2, 3, 224, 224)
    action = torch.randn(2, 4)

    z = model.encode(obs)
    assert z.shape == (2, 64)

    z_next = model.predict(z, action)
    assert z_next.shape == (2, 64)

    reward_pred = model.predict_reward(z, action)
    assert reward_pred.shape == (2, 1)

    obs_seq = torch.randn(2, 5, 3, 224, 224)
    act_seq = torch.randn(2, 5, 4)
    reward_seq = torch.randn(2, 5)
    loss_dict = model.compute_loss(obs_seq, act_seq, reward_seq)
    assert "loss" in loss_dict


def test_bc_policy_forward():
    model = BCPolicyModel(
        img_size=224,
        embed_dim=64,
        action_dim=4,
    )
    obs = torch.randn(2, 3, 224, 224)
    target_action = torch.randn(2, 4)

    pred_action = model(obs)
    assert pred_action.shape == (2, 4)

    loss_dict = model.compute_loss(obs, target_action)
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() >= 0.0
