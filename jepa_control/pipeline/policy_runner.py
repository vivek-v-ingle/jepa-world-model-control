import os
import copy
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from jepa_control.models.vision_transformer import VisionTransformer, vit_giant_xformers_rope
from jepa_control.models.ac_predictor import VisionTransformerPredictorAC
from jepa_control.models.dreamer_predictor import DreamerPredictor
from jepa_control.planner.cem_planner import CEMPlanner
from jepa_control.planner.adaptive_goal import AdaptiveGoalTracker, ReferenceEpisodeLoader
from jepa_control.perception.transforms import ImagePreprocessor
from jepa_control.robot.base_robot import BaseRobot

logger = logging.getLogger(__name__)

class JEPAPolicyRunner:
    """
    Complete end-to-end execution pipeline for JEPA World Model Control:
    1. Preprocesses visual observations (live camera or mock frames).
    2. Encodes observations into latent representations using V-JEPA 2.1 backbone.
    3. Infers target-compatible latent subgoals using the Dreamer Predictor.
    4. Optimizes 7-DoF robot action commands using the CEM latent planner.
    5. Dispatches actions to the robot driver and adaptively tracks progress.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.config = config
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
        dtype_name = config.get("meta", {}).get("dtype", "bfloat16").lower()
        self.dtype = torch.bfloat16 if dtype_name == "bfloat16" and torch.cuda.is_available() else torch.float32

        # Initialize Image Preprocessor
        crop_size = config.get("data", {}).get("crop_size", 256)
        self.preprocessor = ImagePreprocessor(crop_size=crop_size)

        # Build & Load Models
        self._init_models()

        # Initialize Planner & Goal Tracker
        planner_cfg = config.get("planner", {})
        mpc_cfg = planner_cfg.get("mpc", {})
        self.planner = CEMPlanner(
            rollout=mpc_cfg.get("rollout", 1),
            samples=mpc_cfg.get("samples", 200),
            topk=mpc_cfg.get("topk", 10),
            cem_steps=mpc_cfg.get("cem_steps", 50),
            momentum_mean=mpc_cfg.get("momentum_mean", 0.15),
            momentum_std=mpc_cfg.get("momentum_std", 0.75),
            maxnorm=mpc_cfg.get("maxnorm", 0.1),
            action_dim=7,
            abs_gripper=planner_cfg.get("abs_gripper", True),
        )

        self.goal_tracker = AdaptiveGoalTracker(
            l1_threshold=planner_cfg.get("l1_threshold", 1.0),
            queue_horizon=planner_cfg.get("queue_horizon", 4),
        )

    def _init_models(self):
        meta_cfg = self.config.get("meta", {})
        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})

        crop_size = data_cfg.get("crop_size", 256)
        patch_size = data_cfg.get("patch_size", 16)
        tubelet_size = data_cfg.get("tubelet_size", 2)
        embed_dim = 1408
        pred_embed_dim = model_cfg.get("pred_embed_dim", 1024)
        pred_depth = model_cfg.get("pred_depth", 24)

        # 1. Vision Encoder (V-JEPA ViT-Giant)
        self.encoder = vit_giant_xformers_rope(
            img_size=(crop_size, crop_size),
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            num_frames=16,
            uniform_power=True,
        )

        # 2. Action-Conditioned Dynamics Predictor
        self.predictor = VisionTransformerPredictorAC(
            img_size=(crop_size, crop_size),
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            num_frames=16,
            embed_dim=embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=16,
            action_embed_dim=7,
            use_rope=True,
            uniform_power=True,
        )

        # 3. Dreamer Predictor
        self.dreamer_predictor = DreamerPredictor(
            embed_dim=embed_dim,
            num_heads=16,
            patch_h=crop_size // patch_size,
            patch_w=crop_size // patch_size,
            up_dim=64,
            num_self_attn_blocks=16,
        )

        # Load weights
        pretrain_ckpt = meta_cfg.get("pretrain_checkpoint")
        dreamer_ckpt = meta_cfg.get("dreamer_predictor_checkpoint")

        if pretrain_ckpt and os.path.exists(pretrain_ckpt):
            logger.info(f"Loading pretrain checkpoint: {pretrain_ckpt}")
            ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
            
            # Load encoder
            enc_dict = ckpt.get("target_encoder", ckpt.get("encoder", {}))
            clean_enc = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_dict.items()}
            self.encoder.load_state_dict(clean_enc, strict=False)

            # Load predictor
            pred_dict = ckpt.get("predictor", {})
            clean_pred = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pred_dict.items()}
            self.predictor.load_state_dict(clean_pred, strict=False)

        if dreamer_ckpt and os.path.exists(dreamer_ckpt):
            logger.info(f"Loading dreamer checkpoint: {dreamer_ckpt}")
            d_ckpt = torch.load(dreamer_ckpt, map_location="cpu", weights_only=False)
            d_dict = d_ckpt.get("dreamer_predictor", d_ckpt)
            clean_d = {k.replace("module.", "").replace("backbone.", ""): v for k, v in d_dict.items()}
            self.dreamer_predictor.load_state_dict(clean_d, strict=False)

        # Move to device & freeze evaluation
        for m in [self.encoder, self.predictor, self.dreamer_predictor]:
            m.to(self.dtype).to(self.device).eval()
            for p in m.parameters():
                p.requires_grad = False

    def encode_frame(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Takes [1, C, 2, H, W] -> outputs latent tokens [1, HW, D]."""
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device, dtype=self.dtype)
            h = self.encoder(img_tensor)
            h = F.layer_norm(h, (h.size(-1),))
            return h

    def step(
        self,
        current_obs_rgb: np.ndarray,
        current_robot_pose: np.ndarray,
        ref_curr_rgb: np.ndarray,
        ref_target_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, torch.Tensor, float]:
        """
        Executes a single step of perception, goal inference, and CEM planning.
        Returns:
            (action_7d, active_goal, latent_l1_dist)
        """
        with torch.no_grad():
            # 1. Encode observations
            t_curr = self.preprocessor(current_obs_rgb)
            t_ref_curr = self.preprocessor(ref_curr_rgb)
            t_ref_target = self.preprocessor(ref_target_rgb)

            z_curr = self.encode_frame(t_curr)
            z_ref_curr = self.encode_frame(t_ref_curr)
            z_ref_target = self.encode_frame(t_ref_target)

            # 2. Check adaptive goal advancement
            advance, dist = self.goal_tracker.should_advance(z_curr)

            # 3. Infer latent goal if new segment
            goal_latent = self.dreamer_predictor(
                xt=z_curr,
                yt=z_ref_curr,
                yt_plus_1=z_ref_target,
            )
            self.goal_tracker.set_active_goal(goal_latent)

            # 4. Plan action via CEM
            pose_tensor = torch.from_numpy(current_robot_pose[:7]).unsqueeze(0).to(self.device, dtype=self.dtype)

            def world_model_fn(z_in, a_in, s_in):
                z_in = z_in.to(self.dtype)
                a_in = a_in.to(self.dtype)
                s_in = s_in.to(self.dtype)
                z_next = self.predictor(z_in, a_in, s_in)
                return F.layer_norm(z_next, (z_next.size(-1),))

            action_tensor = self.planner.plan(
                context_latent=z_curr,
                current_pose=pose_tensor,
                goal_latent=goal_latent,
                world_model_fn=world_model_fn,
            )

            action_np = action_tensor.detach().float().cpu().numpy()
            return action_np, goal_latent, dist
