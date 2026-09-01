import os
from collections import deque
from typing import Optional, Tuple
import h5py
import numpy as np
import torch
import torch.nn.functional as F

class ReferenceEpisodeLoader:
    """Loads reference demonstration frames from an HDF5 episode."""

    def __init__(
        self,
        h5_path: str,
        image_key: str = "observations/images/camera_front",
        data_fps: int = 30,
        target_fps: int = 5,
    ):
        self.h5_path = os.path.abspath(h5_path)
        self.image_key = image_key
        self.frame_skip = max(1, data_fps // target_fps)
        self.current_idx = 0

        with h5py.File(self.h5_path, "r") as f:
            if image_key not in f:
                # Try common fallback keys
                fallbacks = [
                    "observations/images/camera_front",
                    "observations/images/right_shoulder_rgb",
                    "observations/images/front_rgb",
                ]
                for key in fallbacks:
                    if key in f:
                        self.image_key = key
                        break
            self.images = np.asarray(f[self.image_key])
            self.length = len(self.images)

        if self.length <= 1:
            raise ValueError(f"Reference episode {h5_path} too short: len={self.length}")

    def get_reference_pair(self, advance: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (current_ref_frame, future_ref_frame)."""
        curr_idx = min(self.current_idx, self.length - 1)
        future_idx = min(curr_idx + self.frame_skip, self.length - 1)

        curr_frame = self.images[curr_idx]
        future_frame = self.images[future_idx]

        if advance and self.current_idx < self.length - 1:
            self.current_idx = min(self.current_idx + self.frame_skip, self.length - 1)

        return curr_frame, future_frame

    def reset(self):
        self.current_idx = 0


class AdaptiveGoalTracker:
    """
    Monitors the latent L1 distance between the robot's current latent state
    and the intended subgoal. Advances demonstration frames only when D_k < epsilon.
    """

    def __init__(self, l1_threshold: float = 1.0, queue_horizon: int = 4):
        self.l1_threshold = l1_threshold
        self.obs_buffer = deque(maxlen=queue_horizon)
        self.prev_goal: Optional[torch.Tensor] = None

    def should_advance(self, current_latent: torch.Tensor) -> Tuple[bool, float]:
        """
        Calculates L1 distance between latest observation and active goal.
        Returns:
            (advance_flag, min_l1_dist)
        """
        current_rep = current_latent.detach()
        self.obs_buffer.append(current_rep)

        if self.prev_goal is None or len(self.obs_buffer) == 0:
            return False, 0.0

        # Calculate L1 distance to goal across recent buffer
        dists = [
            F.l1_loss(self.prev_goal.flatten(1), rep.flatten(1)).item()
            for rep in self.obs_buffer
        ]
        min_dist = float(min(dists))

        advance = min_dist < self.l1_threshold
        if advance:
            self.obs_buffer.clear()

        return advance, min_dist

    def set_active_goal(self, goal: torch.Tensor):
        self.prev_goal = goal.detach()

    def reset(self):
        self.obs_buffer.clear()
        self.prev_goal = None
