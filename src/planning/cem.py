from typing import Callable, Tuple, Any
import torch
from src.planning.cost_functions import cosine_distance


class CEMPlanner:
    """
    Vectorized Cross-Entropy Method (CEM) for trajectory optimization in latent space.
    Optimizes candidate action sequences to minimize latent distance to goal representations.
    """

    def __init__(
        self,
        horizon: int = 5,
        num_samples: int = 200,
        num_iterations: int = 10,
        elite_fraction: float = 0.1,
        momentum: float = 0.2,
        action_dim: int = 4,
        action_low: Any = -0.05,
        action_high: Any = 0.05,
        device: str = "cpu",
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.elite_fraction = elite_fraction
        self.num_elites = max(2, int(num_samples * elite_fraction))
        self.momentum = momentum
        self.action_dim = action_dim
        self.device = device

        if not isinstance(action_low, torch.Tensor):
            self.action_low = torch.full((action_dim,), float(action_low), device=device)
        else:
            self.action_low = action_low.to(device)

        if not isinstance(action_high, torch.Tensor):
            self.action_high = torch.full((action_dim,), float(action_high), device=device)
        else:
            self.action_high = action_high.to(device)

    def plan(
        self,
        initial_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine_distance,
    ) -> Tuple[torch.Tensor, float]:
        """
        Execute CEM optimization over action trajectories.
        :param initial_latent: (1, ...) or (D,) latent state of current observation
        :param goal_latent: (1, ...) or (D,) latent state of target goal observation
        :param dynamics_fn: callable taking (S, ...) latents and (S, H, d_a) actions, returning terminal latents (S, ...)
        :param cost_fn: callable taking (S, ...) and (S, ...), returning (S,) costs
        :return: (best_action_seq, best_cost)
        """
        device = initial_latent.device

        # Initialize Gaussian action distribution
        mean = torch.zeros((self.horizon, self.action_dim), device=device)
        std = torch.ones((self.horizon, self.action_dim), device=device) * (
            (self.action_high - self.action_low) / 2.0
        )

        # Expand initial latent and goal latent across samples
        if initial_latent.ndim == 1:
            initial_latent = initial_latent.unsqueeze(0)
        if goal_latent.ndim == 1:
            goal_latent = goal_latent.unsqueeze(0)

        # Broadcast initial latent and goal latent to (S, ...)
        repeat_shape = [self.num_samples] + [1] * (initial_latent.ndim - 1)
        latents_batch = initial_latent.repeat(*repeat_shape)
        goal_batch = goal_latent.repeat(*repeat_shape)

        best_action_seq = None
        best_cost = float("inf")

        for iteration in range(self.num_iterations):
            # 1. Sample candidate action trajectories
            # Shape: (S, H, d_a)
            noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=device)
            actions = mean.unsqueeze(0) + noise * std.unsqueeze(0)
            actions = torch.clamp(actions, min=self.action_low, max=self.action_high)

            # 2. Simulate latent trajectories using dynamics model
            with torch.no_grad():
                terminal_latents = dynamics_fn(latents_batch, actions)

                # 3. Compute costs against goal latent
                costs = cost_fn(terminal_latents, goal_batch)

            # 4. Select elite samples
            elite_indices = torch.topk(costs, k=self.num_elites, largest=False).indices
            elite_actions = actions[elite_indices]  # (K, H, d_a)

            # Track global best
            min_cost_idx = elite_indices[0]
            current_best_cost = costs[min_cost_idx].item()
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_action_seq = actions[min_cost_idx].clone()

            # 5. Update mean and std with momentum
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0).clamp(min=1e-4)

            mean = self.momentum * mean + (1.0 - self.momentum) * new_mean
            std = self.momentum * std + (1.0 - self.momentum) * new_std

        if best_action_seq is None:
            best_action_seq = mean

        return best_action_seq, best_cost
