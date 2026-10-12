import torch
import pytest

from src.planning.cost_functions import cosine_distance, l1_distance, l2_distance
from src.planning.cem import CEMPlanner
from src.planning.mpc_controller import LatentMPCController


def test_cost_functions():
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    z2 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)

    cos_dist = cosine_distance(z1, z2)
    assert cos_dist.shape == (2,)
    assert pytest.approx(cos_dist[0].item(), 1e-4) == 0.0
    assert pytest.approx(cos_dist[1].item(), 1e-4) == 2.0

    l1_dist = l1_distance(z1, z2)
    assert pytest.approx(l1_dist[0].item(), 1e-4) == 0.0
    assert pytest.approx(l1_dist[1].item(), 1e-4) == 1.0

    l2_dist = l2_distance(z1, z2)
    assert pytest.approx(l2_dist[0].item(), 1e-4) == 0.0
    assert pytest.approx(l2_dist[1].item(), 1e-4) == 4.0


def test_cem_planner():
    planner = CEMPlanner(
        horizon=3,
        num_samples=50,
        num_iterations=5,
        elite_fraction=0.2,
        action_dim=4,
        action_low=-0.05,
        action_high=0.05,
    )

    # Synthetic linear dynamics: z_{t+1} = z_t + action[:2]
    # Goal: reach z_goal = [0.1, 0.1] from [0.0, 0.0]
    z_init = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    z_goal = torch.tensor([[0.1, 0.1]], dtype=torch.float32)

    def mock_dynamics(latents, actions):
        # latents: (S, 2), actions: (S, H, 4)
        curr = latents
        for t in range(actions.shape[1]):
            curr = curr + actions[:, t, :2]
        return curr

    best_action_seq, best_cost = planner.plan(
        initial_latent=z_init,
        goal_latent=z_goal,
        dynamics_fn=mock_dynamics,
        cost_fn=l2_distance,
    )

    assert best_action_seq.shape == (3, 4)
    assert isinstance(best_cost, float)
    # The actions should move in the positive direction toward [0.1, 0.1]
    total_delta = best_action_seq[:, :2].sum(dim=0)
    assert total_delta[0] > 0.02
    assert total_delta[1] > 0.02


def test_latent_mpc_controller_step():
    class MockWorldModel:
        def encode(self, obs):
            # obs: (H, W, 3) or (B, H, W, 3)
            return torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        def predict(self, latent, action):
            # latent: (B, D), action: (B, da)
            return latent + action[:, :2]

    controller = LatentMPCController(
        world_model=MockWorldModel(),
        horizon=2,
        num_samples=20,
        num_iterations=3,
        action_dim=4,
    )

    obs = torch.zeros((224, 224, 3))
    goal_obs = torch.ones((224, 224, 3))

    action = controller.get_action(obs, goal_obs)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (4,)
