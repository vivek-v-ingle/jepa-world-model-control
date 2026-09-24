import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure Meta FAIR jepa-wms repo is on sys.path
HUB_DIR = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_jepa-wms_main"
if HUB_DIR.exists() and str(HUB_DIR) not in sys.path:
    sys.path.insert(0, str(HUB_DIR))

logger = logging.getLogger(__name__)

class DinoWMRunner:
    """
    Closed-loop Visual Manipulation Runner using Meta FAIR's JEPA World Models (dino_wm_droid).
    
    Backbone: DINOv2 ViT-S/14 frozen visual encoder (produces spatially localized 16x16 patch tokens).
    World Model: Action-conditioned ViTPredictor trained on real DROID 7-DoF robot manipulation dataset.
    Planning: Model Predictive Path Planning via Cross-Entropy Method (CEMPlanner) or MPPI.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.config = config
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
        logger.info(f"Initializing DinoWMRunner with device: {self.device}")
        
        # 1. Load Meta FAIR dino_wm_droid model and preprocessor via torch.hub
        try:
            from evals.simu_env_planning.planning.planning.planner import CEMPlanner, MPPIPlanner
            from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistL1MPCObjective
        except ImportError as e:
            logger.error(f"Failed to import jepa-wms planning modules: {e}")
            raise

        logger.info("Loading Meta FAIR dino_wm_droid checkpoint...")
        self.model, self.preprocessor = torch.hub.load(
            "facebookresearch/jepa-wms",
            "dino_wm_droid",
            device=str(self.device),
        )
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 2. Configure CEM Planner matching official Meta FAIR DROID benchmark
        dino_cfg = config.get("dino_wm", {})
        planner_cfg = config.get("planner", {})
        mpc_cfg = planner_cfg.get("mpc", {})
        
        self.iterations = dino_cfg.get("cem_steps", 15)
        self.num_samples = dino_cfg.get("samples", 300)
        self.horizon = dino_cfg.get("rollout", 3)
        self.num_elites = dino_cfg.get("topk", 10)
        self.var_scale = dino_cfg.get("var_scale", 0.1)
        self.l1_threshold = dino_cfg.get("l1_threshold", 0.70)

        self.planner = CEMPlanner(
            unroll=self.model.unroll,
            iterations=self.iterations,
            num_samples=self.num_samples,
            horizon=self.horizon,
            action_dim=7,
            num_elites=self.num_elites,
            var_scale=self.var_scale,
            max_norms=[0.1, 0.75],
            max_norm_dims=[[0, 1, 2, 3, 4, 5], [6]],
            num_act_stepped=1,
        )

        self.step_count = 0
        self.active_goal_latent = None
        self.current_latent = None
        logger.info(f"DinoWMRunner initialized (CEM: iter={self.iterations}, samples={self.num_samples}, horizon={self.horizon}, var_scale={self.var_scale})")

    def preprocess_image(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        Converts BGR OpenCV / camera frame (e.g. 1280x720) into normalized tensor:
        Target shape: [B=1, T=1, C=3, H=224, W=224] on device.
        DROID world model was trained with [-1, 1] normalization: (x / 255.0 - 0.5) / 0.5.
        """
        if img_bgr is None:
            raise ValueError("Provided image is None")

        # Convert BGR to RGB
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr

        # Resize to 224x224
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        
        # [H, W, C] -> [1, 1, C, H, W]
        tensor = torch.from_numpy(img_resized).float().to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
        
        # Apply official DROID dataset normalization
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def encode_frame(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Takes [1, 1, 3, 224, 224] -> returns [1, 1, 1, 16, 16, 384] representation."""
        with torch.no_grad():
            return self.model.encode(img_tensor)

    def step(
        self,
        current_obs_rgb: np.ndarray,
        current_robot_pose: np.ndarray,
        ref_curr_rgb: np.ndarray,
        ref_target_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, torch.Tensor, float, bool, str]:
        """
        Executes a single closed-loop perception and visual planning step.
        Returns:
            (action_7d, active_goal_latent, latent_l1_dist, advance, reason)
        """
        self.step_count += 1
        
        with torch.no_grad():
            # 1. Preprocess and encode current observation and target reference goal
            t_curr = self.preprocess_image(current_obs_rgb)
            t_goal = self.preprocess_image(ref_target_rgb)

            z_curr = self.encode_frame(t_curr) # [1, 1, 1, 16, 16, 384]
            z_goal = self.encode_frame(t_goal) # [1, 1, 1, 16, 16, 384]
            self.current_latent = z_curr
            self.active_goal_latent = z_goal

            # 2. Compute latent L1 distance to reference goal
            # DINOv2 representations yield sharp spatial gradients on physical targets
            target_enc = z_goal[:, 0] # [1, 1, 16, 16, 384]
            dist = torch.mean(torch.abs(z_curr[:, 0] - target_enc)).item()

            # 3. Formulate MPC Objective (L2 representation distance matching DROID training)
            from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
            objective = ReprTargetDistMPCObjective(cfg={}, target_enc=target_enc, sum_all_diffs=False)
            self.planner.set_objective(objective)

            # 4. Plan optimal action trajectory with CEM
            res = self.planner.plan(z_curr)
            planned_action = res.actions[0].cpu().numpy().copy()

            advance = dist <= self.l1_threshold
            reason = f"L1={dist:.4f} (thresh={self.l1_threshold:.2f})"

            return planned_action, z_goal, dist, advance, reason
