import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DinoLatentPatchAttention:
    """
    Self-Supervised Goal Retargeting & Object Localization in DINOv2 Latent Space.
    
    Uses Cross-Patch Feature Cosine Similarity and Latent Attention between the
    live camera observation and reference demonstration representations to spatially
    retarget the goal latent z* without any OpenCV color masks or hand-crafted detectors.
    """

    def __init__(
        self,
        nominal_bottle_patch: Tuple[int, int] = (8, 9),
        patch_grid_size: int = 16,
        embed_dim: int = 384,
        temperature: float = 0.05,
        use_feature_centering: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.nominal_patch = nominal_bottle_patch
        self.grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size  # 256
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.use_centering = use_feature_centering
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        # Reference bottle token (extracted on first reference frame)
        self.ref_bottle_token: Optional[torch.Tensor] = None
        self.ref_latent: Optional[torch.Tensor] = None
        self.ref_patch_coords: Tuple[int, int] = nominal_bottle_patch
        self.bg_token: Optional[torch.Tensor] = None

        logger.info(
            f"DinoLatentPatchAttention initialized (grid={self.grid_size}x{self.grid_size}, "
            f"nominal_patch={self.nominal_patch}, temp={self.temperature})"
        )

    def auto_detect_bottle_patch(self, z_obs: torch.Tensor) -> Tuple[int, int]:
        """
        Automatically detects the bottle's patch coordinates (r, c) on the tabletop
        using semantic feature contrast against the black cloth background.
        """
        if z_obs.ndim == 6:
            z_sq = z_obs[0, 0, 0]
        elif z_obs.ndim == 4:
            z_sq = z_obs[0]
        else:
            z_sq = z_obs

        # Table cloth background anchor (row 12, col 9)
        bg = z_sq[12, 9]
        bg_norm = F.normalize(bg.unsqueeze(0), dim=-1)

        flat = z_sq.view(self.num_patches, self.embed_dim)
        flat_norm = F.normalize(flat, dim=-1)

        # High difference indicates foreground object on the table
        bg_sim = torch.matmul(flat_norm, bg_norm.T).view(self.grid_size, self.grid_size)
        diff = 1.0 - bg_sim

        # Restrict search strictly to tabletop workspace (rows 6 to 11, cols 7 to 13)
        # Note: Columns 0 to 6 are the background white wall and must be masked out
        mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool, device=z_obs.device)
        mask[6:11, 7:13] = True
        diff[~mask] = -1.0

        peak_idx = diff.argmax().item()
        r, c = peak_idx // self.grid_size, peak_idx % self.grid_size
        return (r, c)

    def initialize_reference(
        self,
        z_ref: torch.Tensor,
        nominal_coords: Optional[Any] = None,
        z_live: Optional[torch.Tensor] = None,
    ):
        """
        Extracts reference semantic tokens from the reference demonstration latent.
        z_ref: [1, 1, 1, 16, 16, 384] or [16, 16, 384]
        """
        if z_ref.ndim == 6:
            z_ref = z_ref[0, 0, 0]  # [16, 16, 384]
        elif z_ref.ndim == 4:
            z_ref = z_ref[0]

        self.ref_latent = z_ref.clone()
        target_coords = nominal_coords or self.nominal_patch

        if target_coords == "auto" or target_coords == ("a", "u", "t", "o") or target_coords is None:
            # Auto-detect bottle on reference demonstration
            coords = self.auto_detect_bottle_patch(z_ref)
            logger.info(f"🔍 [AUTO-DETECT] Tabletop bottle automatically detected at patch (r={coords[0]}, c={coords[1]})")
        else:
            coords = target_coords

        self.ref_patch_coords = coords

        # Background token from lower-middle black cloth (e.g. r=12, c=9)
        self.bg_token = z_ref[12, 9].clone()

        # Extract bottle token at designated patch
        r, c = coords
        ref_flat = z_ref.view(self.num_patches, self.embed_dim)
        if self.use_centering:
            ref_flat = ref_flat - ref_flat.mean(dim=0, keepdim=True)
        ref_norm = F.normalize(ref_flat, dim=-1)
        self.ref_bottle_token = ref_norm[r * self.grid_size + c].clone()

        logger.info(
            f"DINOv2 Reference Initialized: Bottle token at (r={r}, c={c}) in 16x16 grid | "
            f"Norm: {torch.norm(self.ref_bottle_token).item():.4f}"
        )

    def compute_similarity_map(self, z_curr: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], float]:
        """
        Computes 16x16 cross-patch cosine similarity heatmap between the reference bottle
        and all spatial patches in the current observation latent, masked strictly to tabletop workspace.
        Returns:
            (sim_map [16, 16], (peak_r, peak_c), max_similarity)
        """
        if self.ref_bottle_token is None:
            self.initialize_reference(z_curr)

        if z_curr.ndim == 6:
            z_curr_sq = z_curr[0, 0, 0]
        elif z_curr.ndim == 4:
            z_curr_sq = z_curr[0]
        else:
            z_curr_sq = z_curr

        curr_flat = z_curr_sq.view(self.num_patches, self.embed_dim)
        if self.use_centering:
            curr_flat = curr_flat - curr_flat.mean(dim=0, keepdim=True)
        curr_norm = F.normalize(curr_flat, dim=-1)

        # Cross-patch cosine similarity with the reference bottle token
        sim_flat = torch.matmul(curr_norm, self.ref_bottle_token)  # [256]
        sim_map = sim_flat.view(self.grid_size, self.grid_size)

        # Tabletop workspace mask to prevent latching onto wall (cols 0-6) or upper arm
        table_mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool, device=z_curr.device)
        table_mask[5:11, 7:14] = True
        sim_masked = sim_map.clone()
        sim_masked[~table_mask] = -1.0

        peak_idx = torch.argmax(sim_masked).item()
        peak_r = peak_idx // self.grid_size
        peak_c = peak_idx % self.grid_size
        max_sim = sim_map[peak_r, peak_c].item()

        return sim_map, (peak_r, peak_c), max_sim

    def retarget_goal_latent(
        self,
        z_subgoal: torch.Tensor,
        z_curr: torch.Tensor,
        blend_factor: float = 0.85,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Spatially retargets the reference goal latent z*_subgoal to the bottle's new location.
        
        Args:
            z_subgoal: Target goal representation from demonstration [1, 1, 1, 16, 16, 384]
            z_curr: Current live observation representation [1, 1, 1, 16, 16, 384]
            blend_factor: Weight given to spatially shifted subgoal vs background retention
            
        Returns:
            (retargeted_z_subgoal, telemetry_info)
        """
        if self.ref_bottle_token is None:
            self.initialize_reference(z_subgoal)

        sim_map, (live_r, live_c), max_sim = self.compute_similarity_map(z_curr)
        ref_r, ref_c = self.ref_patch_coords

        delta_r = live_r - ref_r
        delta_c = live_c - ref_c

        # Preserve original dimensions [1, 1, 1, 16, 16, 384]
        orig_shape = z_subgoal.shape
        z_sub = z_subgoal.clone()
        if z_sub.ndim == 6:
            z_sub_grid = z_sub[0, 0, 0]  # [16, 16, 384]
        else:
            z_sub_grid = z_sub

        H, W, D = z_sub_grid.shape
        retargeted_grid = z_sub_grid.clone()

        # 1. Spatial Patch Grid Translation:
        # Move the interaction tokens by (delta_r, delta_c)
        if delta_r != 0 or delta_c != 0:
            shifted = torch.zeros_like(z_sub_grid)
            # Fill with black cloth background token if available
            bg = self.bg_token if self.bg_token is not None else z_sub_grid[0, 0]
            shifted[:] = bg

            # Source and destination slices in 16x16 grid
            src_r_start = max(0, -delta_r)
            src_r_end = min(H, H - delta_r)
            src_c_start = max(0, -delta_c)
            src_c_end = min(W, W - delta_c)

            dst_r_start = max(0, delta_r)
            dst_r_end = min(H, H + delta_r)
            dst_c_start = max(0, delta_c)
            dst_c_end = min(W, W + delta_c)

            shifted[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = z_sub_grid[
                src_r_start:src_r_end, src_c_start:src_c_end
            ]
            retargeted_grid = shifted

        # 2. Cross-Patch Attention (Query-Key-Value in Latent Space):
        # Softmax-weighted feature transport between live scene and reference demonstration
        curr_flat = z_curr[0, 0, 0].view(self.num_patches, D)
        ref_source = self.ref_latent if self.ref_latent is not None else z_sub_grid
        ref_flat = ref_source.view(self.num_patches, D)
        val_flat = retargeted_grid.view(self.num_patches, D)

        curr_norm = F.normalize(curr_flat, dim=-1)
        ref_norm = F.normalize(ref_flat, dim=-1)

        # Cross-patch attention matrix [256, 256]
        attn = F.softmax(torch.matmul(curr_norm, ref_norm.T) / self.temperature, dim=-1)
        z_attn_transported = torch.matmul(attn, val_flat).view(H, W, D)

        # Blend shifted grid with cross-attention transport for smooth spatial tokens
        final_grid = blend_factor * retargeted_grid + (1.0 - blend_factor) * z_attn_transported

        # Restore original tensor shape [1, 1, 1, 16, 16, 384]
        retargeted_z = final_grid.view(orig_shape)

        info = {
            "ref_bottle_patch": (ref_r, ref_c),
            "live_bottle_patch": (live_r, live_c),
            "patch_delta": (delta_r, delta_c),
            "cosine_similarity": float(max_sim),
            "sim_map": sim_map.detach().cpu().numpy(),
        }

        return retargeted_z, info
