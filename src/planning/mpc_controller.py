from typing import Any, Optional
import torch
import numpy as np

from src.planning.cem import CEMPlanner
from src.planning.cost_functions import cosine_distance


class LatentMPCController:
    """
    Receding Horizon Latent Model Predictive Controller.
    Computes visual representations of current and goal observations, optimizes multi-step
    action rollouts in representation space via CEM, and executes the first action step.
    """

    def __init__(
        self,
        world_model: Any,
        horizon: int = 5,
        num_samples: int = 200,
        num_iterations: int = 10,
        elite_fraction: float = 0.1,
        momentum: float = 0.2,
        action_dim: int = 4,
        action_low: Any = None,
        action_high: Any = None,
        cost_metric: str = "cosine",
        device: str = "cpu",
    ):
        self.world_model = world_model
        self.device = device

        if action_low is None:
            action_low = torch.tensor([-0.05, -0.05, -0.05, -1.0], device=device)
        if action_high is None:
            action_high = torch.tensor([0.05, 0.05, 0.05, 1.0], device=device)

        self.planner = CEMPlanner(
            horizon=horizon,
            num_samples=num_samples,
            num_iterations=num_iterations,
            elite_fraction=elite_fraction,
            momentum=momentum,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        self.cost_metric = cost_metric

    def _dynamics_rollout(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Unroll latent state dynamics forward over planning horizon.
        :param latents: (S, D) or (S, N, D)
        :param actions: (S, H, d_a)
        :return: terminal latent representations (S, D) or (S, N, D)
        """
        curr = latents
        H = actions.shape[1]
        for t in range(H):
            act_step = actions[:, t, :]
            curr = self.world_model.predict(curr, act_step)
        return curr

    def get_action(
        self,
        current_obs: Any,
        goal_obs: Any,
    ) -> torch.Tensor:
        """
        Plan and return the first action step for closed-loop execution.
        :param current_obs: observation tensor or numpy array
        :param goal_obs: goal observation tensor or numpy array
        :return: action tensor of shape (action_dim,)
        """
        if isinstance(current_obs, np.ndarray):
            current_obs = torch.from_numpy(current_obs).float()
        if isinstance(goal_obs, np.ndarray):
            goal_obs = torch.from_numpy(goal_obs).float()

        current_obs = current_obs.to(self.device)
        goal_obs = goal_obs.to(self.device)

        with torch.no_grad():
            z_init = self.world_model.encode(current_obs)
            z_goal = self.world_model.encode(goal_obs)

            best_action_seq, _ = self.planner.plan(
                initial_latent=z_init,
                goal_latent=z_goal,
                dynamics_fn=self._dynamics_rollout,
                cost_fn=cosine_distance,
            )

        return best_action_seq[0]
