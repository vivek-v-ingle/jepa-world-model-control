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
        self.last_predicted_latent: Optional[torch.Tensor] = None
        self.last_prediction_error: Optional[float] = None
        self.last_retarget_info: Dict[str, Any] = {}

        # 3. Initialize DINOv2 Latent Patch Attention Retargeter
        from jepa_control.planner.latent_patch_attention import DinoLatentPatchAttention
        nominal_cfg = dino_cfg.get("nominal_bottle_patch", "auto")
        if isinstance(nominal_cfg, (list, tuple)):
            nominal_patch = tuple(nominal_cfg)
        else:
            nominal_patch = "auto"
        patch_temp = float(dino_cfg.get("patch_temp", 0.05))
        use_centering = dino_cfg.get("feature_centering", True)

        self.retargeter = DinoLatentPatchAttention(
            nominal_bottle_patch=nominal_patch,
            temperature=patch_temp,
            use_feature_centering=use_centering,
            device=self.device,
        )

        logger.info(
            f"DinoWMRunner initialized (CEM: iter={self.iterations}, samples={self.num_samples}, "
            f"horizon={self.horizon}, var_scale={self.var_scale} | Retargeter: nominal={nominal_patch})"
        )

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

    def predict_next_latent(self, z_curr: torch.Tensor, action_7d: np.ndarray) -> torch.Tensor:
        """
        Autoregressively predicts next latent state using ViTPredictor unroll:
        \\hat{z}_{t+1} = WorldModel(z_t, a_t)
        Returns:
            \\hat{z}_{t+1}: [1, 1, 1, 16, 16, 384]
        """
        with torch.no_grad():
            act_tensor = torch.from_numpy(action_7d).float().to(self.device).view(1, 1, 7)
            pred_seq = self.model.unroll(z_curr, act_tensor)
            return pred_seq[1:2]

    def step(
        self,
        current_obs_rgb: np.ndarray,
        current_robot_pose: np.ndarray,
        ref_curr_rgb: np.ndarray,
        ref_target_rgb: np.ndarray,
        retarget_goal: bool = True,
        return_info: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Executes a single closed-loop perception, prediction verification, and visual planning step.
        Returns:
            (action_7d, active_goal_latent, latent_l1_dist, advance, reason [, retarget_info])
        """
        self.step_count += 1
        
        with torch.no_grad():
            # 1. Preprocess and encode current observation and target reference goal
            t_curr = self.preprocess_image(current_obs_rgb)
            t_goal = self.preprocess_image(ref_target_rgb)

            z_curr = self.encode_frame(t_curr)  # [1, 1, 1, 16, 16, 384]
            z_goal = self.encode_frame(t_goal)  # [1, 1, 1, 16, 16, 384]
            self.current_latent = z_curr

            # 2. Evaluate World Model 1-Step Prediction Error: ||\\hat{z}_t - z_t||_1
            pred_error = None
            if self.last_predicted_latent is not None:
                pred_error = torch.mean(torch.abs(self.last_predicted_latent - z_curr)).item()
                self.last_prediction_error = pred_error
                logger.info(
                    f"🔮 [WORLD MODEL] Prediction Error (Step {self.step_count - 1} -> {self.step_count}): "
                    f"||\\hat{{z}}_t - z_t||_1 = {pred_error:.6f}"
                )

            # 3. Ensure reference bottle token is anchored to demonstration
            if self.retargeter.ref_bottle_token is None:
                t_ref_0 = self.preprocess_image(ref_curr_rgb)
                z_ref_0 = self.encode_frame(t_ref_0)
                self.retargeter.initialize_reference(z_ref_0, z_live=z_curr)

            # 4. Spatially retarget goal latent via DINOv2 Latent Patch Attention
            if retarget_goal:
                z_target, retarget_info = self.retargeter.retarget_goal_latent(z_goal, z_curr)
            else:
                z_target = z_goal
                sim_map, (live_r, live_c), max_sim = self.retargeter.compute_similarity_map(z_curr)
                retarget_info = {
                    "ref_bottle_patch": self.retargeter.ref_patch_coords,
                    "live_bottle_patch": (live_r, live_c),
                    "patch_delta": (live_r - self.retargeter.ref_patch_coords[0], live_c - self.retargeter.ref_patch_coords[1]),
                    "cosine_similarity": float(max_sim),
                    "sim_map": sim_map.detach().cpu().numpy(),
                }

            retarget_info["prediction_error"] = pred_error
            self.last_retarget_info = retarget_info
            self.active_goal_latent = z_target

            # 5. Compute latent L1 distance to retargeted goal
            target_enc = z_target[:, 0]  # [1, 1, 16, 16, 384]
            dist = torch.mean(torch.abs(z_curr[:, 0] - target_enc)).item()

            # 6. Formulate MPC Objective (L2 representation distance matching DROID training)
            from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
            objective = ReprTargetDistMPCObjective(cfg={}, target_enc=target_enc, sum_all_diffs=False)
            self.planner.set_objective(objective)

            # 7. Plan optimal action trajectory with CEM
            res = self.planner.plan(z_curr)
            planned_action = res.actions[0].cpu().numpy().copy()

            # 8. Record World Model 1-step prediction \\hat{z}_{t+1} for next step verification
            self.last_predicted_latent = self.predict_next_latent(z_curr, planned_action)

            advance = dist <= self.l1_threshold
            err_str = f" | WMErr={pred_error:.4f}" if pred_error is not None else ""
            reason = f"L1={dist:.4f} (thresh={self.l1_threshold:.2f}){err_str}"

            if return_info:
                return planned_action, z_target, dist, advance, reason, retarget_info
            return planned_action, z_target, dist, advance, reason
