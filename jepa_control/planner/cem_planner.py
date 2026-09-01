import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def latent_l1_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes mean L1 distance across feature dimensions."""
    return torch.mean(torch.abs(a.flatten(1) - b.flatten(1)), dim=-1)

class CEMPlanner:
    """
    Cross-Entropy Method (CEM) planner for latent-space trajectory optimization.
    Optimizes a sequence of actions such that the Action-Conditioned World Model's
    predicted latent rollout matches the target latent goal.
    """

    def __init__(
        self,
        rollout: int = 1,
        samples: int = 200,
        topk: int = 10,
        cem_steps: int = 50,
        momentum_mean: float = 0.15,
        momentum_std: float = 0.75,
        maxnorm: float = 0.1,
        action_dim: int = 7,
        abs_gripper: bool = True,
    ):
        self.rollout = rollout
        self.samples = samples
        self.topk = topk
        self.cem_steps = cem_steps
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.maxnorm = maxnorm
        self.action_dim = action_dim
        self.abs_gripper = abs_gripper

    def plan(
        self,
        context_latent: torch.Tensor,
        current_pose: torch.Tensor,
        goal_latent: torch.Tensor,
        world_model_fn,
    ) -> torch.Tensor:
        """
        Args:
            context_latent: [1, HW, D] current observation latent tokens
            current_pose:   [1, 7] current robot state/pose
            goal_latent:    [1, HW, D] target latent goal from Dreamer Predictor
            world_model_fn: callable f(latent, action, state) -> next_latent
        Returns:
            best_action: [1, 7] or [rollout, 7] optimized action command
        """
        device = context_latent.device
        dtype = context_latent.dtype

        # Ensure 4D latents [B, T=1, HW, D]
        if context_latent.ndim == 3:
            context_latent = context_latent.unsqueeze(1)
        if goal_latent.ndim == 3:
            goal_latent = goal_latent.unsqueeze(1)
        if current_pose.ndim == 2:
            current_pose = current_pose.unsqueeze(1)

        # Repeat context for all samples: [S, 1, HW, D] and [S, 1, 7]
        ctx_latent_s = context_latent.repeat(self.samples, 1, 1, 1)
        goal_latent_s = goal_latent.repeat(self.samples, 1, 1, 1)
        pose_s = current_pose.repeat(self.samples, 1, 1)

        # Initialize distribution
        mean = torch.zeros((self.rollout, self.action_dim), device=device, dtype=torch.float32)
        std = torch.ones((self.rollout, self.action_dim), device=device, dtype=torch.float32) * self.maxnorm

        for step in range(self.cem_steps):
            # Sample action candidates: [S, Rollout, ActionDim]
            action_samples = torch.randn((self.samples, self.rollout, self.action_dim), device=device, dtype=torch.float32) * std + mean
            
            # Clip Cartesian bounds
            action_samples[:, :, :-1] = torch.clamp(action_samples[:, :, :-1], min=-self.maxnorm, max=self.maxnorm)
            if self.abs_gripper:
                action_samples[:, :, -1:] = torch.clamp(action_samples[:, :, -1:], min=0.0, max=1.0)

            # Rollout inside the World Model
            curr_z = ctx_latent_s
            for h in range(self.rollout):
                a_h = action_samples[:, h : h + 1, :] # [S, 1, ActionDim]
                curr_z = world_model_fn(curr_z, a_h, pose_s) # [S, 1, HW, D]

            # Score distance to goal
            costs = torch.mean(torch.abs(curr_z.flatten(1) - goal_latent_s.flatten(1)), dim=-1) # [S]

            # Select elite samples
            _, elite_idx = torch.topk(costs, k=self.topk, largest=False)
            elite_actions = action_samples[elite_idx] # [K, Rollout, ActionDim]

            # Update distribution parameters with momentum
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0) + 1e-6

            mean = self.momentum_mean * mean + (1.0 - self.momentum_mean) * new_mean
            std = self.momentum_std * std + (1.0 - self.momentum_std) * new_std

        # Return mean of the elite distribution
        best_action = mean[0] # First step action [7]
        return best_action
