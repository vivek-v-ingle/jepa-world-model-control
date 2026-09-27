#!/usr/bin/env python3
"""
Evaluate Cross-Viewpoint JEPA Transfer:
Input: Side-View Demonstration Frames (data/reference/side_demo_frame*.jpg)
Live Observation: Top-Down ZED Camera (data/reference/live_cam_check.jpg)

Computes:
1. Latent L1 distance between Side reference frames and Top live observation.
2. DINOv2 Cross-View Semantic Feature Cosine Similarity.
3. CEM planner action response under cross-view conditioning.
Generates proof artifacts (visual heatmap and summary statistics).
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from jepa_control.pipeline.dino_wm_runner import DinoWMRunner

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINO-WM on {device}...")

    config_path = ROOT / "config" / "deploy_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    runner = DinoWMRunner(config, device=device)

    # Load image frames
    top_img_path = ROOT / "data" / "reference" / "live_cam_check.jpg"
    side_f0_path = ROOT / "data" / "reference" / "side_demo_frame0.jpg"
    side_f250_path = ROOT / "data" / "reference" / "side_demo_frame250.jpg"
    side_f400_path = ROOT / "data" / "reference" / "side_demo_frame400.jpg"

    top_bgr = cv2.imread(str(top_img_path))
    side_f0_bgr = cv2.imread(str(side_f0_path))
    side_f250_bgr = cv2.imread(str(side_f250_path))
    side_f400_bgr = cv2.imread(str(side_f400_path))

    top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    side_f0_rgb = cv2.cvtColor(side_f0_bgr, cv2.COLOR_BGR2RGB)
    side_f250_rgb = cv2.cvtColor(side_f250_bgr, cv2.COLOR_BGR2RGB)
    side_f400_rgb = cv2.cvtColor(side_f400_bgr, cv2.COLOR_BGR2RGB)

    # 1. Encode into DINOv2 latent space
    top_t = runner.preprocess_image(top_rgb)
    side_f0_t = runner.preprocess_image(side_f0_rgb)
    side_f250_t = runner.preprocess_image(side_f250_rgb)
    side_f400_t = runner.preprocess_image(side_f400_rgb)

    top_latent = runner.encode_frame(top_t).view(256, 384)
    side_f0_latent = runner.encode_frame(side_f0_t).view(256, 384)
    side_f250_latent = runner.encode_frame(side_f250_t).view(256, 384)
    side_f400_latent = runner.encode_frame(side_f400_t).view(256, 384)

    # Calculate L1 distances
    dist_top_vs_side0 = torch.mean(torch.abs(top_latent - side_f0_latent)).item()
    dist_top_vs_side250 = torch.mean(torch.abs(top_latent - side_f250_latent)).item()
    dist_top_vs_side400 = torch.mean(torch.abs(top_latent - side_f400_latent)).item()

    print("\n" + "=" * 65)
    print("CROSS-VIEWPOINT LATENT DISTANCE ANALYSIS (TOP vs SIDE)")
    print("=" * 65)
    print(f"Top Live vs Side Demo Frame 0   (Home Pose):    L1 = {dist_top_vs_side0:.4f}")
    print(f"Top Live vs Side Demo Frame 250 (Grasp Pose):   L1 = {dist_top_vs_side250:.4f}")
    print(f"Top Live vs Side Demo Frame 400 (Lift & Carry): L1 = {dist_top_vs_side400:.4f}")
    print(f"Contrast with Same-View Rollout Baseline:       L1 = ~0.34 - 0.44 (converging to 0.089)")

    # 2. Test CEM Action Planning under Cross-View Condition
    fake_pose = np.array([-673.0, -424.0, 174.0, 180.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action_delta, goal_latent, dist, advance, reason = runner.step(
        current_obs_rgb=top_rgb,
        current_robot_pose=fake_pose,
        ref_curr_rgb=side_f0_rgb,
        ref_target_rgb=side_f250_rgb,  # Target is the grasp frame in the side view
    )

    print("\n" + "=" * 65)
    print("CEM PLANNER BEHAVIOR (Goal = Side Grasp, Obs = Top View)")
    print("=" * 65)
    print(f"Planned Action Delta: {[round(float(x), 4) for x in action_delta]}")
    print(f"Computed Latent Distance: {dist:.4f}")

    # 3. Compute DINOv2 Semantic Patch Cross-Similarity
    # Find bottle in side view: in side_demo_frame250, bottle is in center-right
    # Spatial grid is 16x16 = 256 patches.
    # Normalize features along channel dimension:
    top_feat_norm = F.normalize(top_latent, dim=-1)       # [256, 384]
    side_feat_norm = F.normalize(side_f250_latent, dim=-1) # [256, 384]

    # Cosine similarity matrix between all side patches and all top patches: [256, 256]
    cos_sim = torch.matmul(side_feat_norm, top_feat_norm.T)

    # Max similarity across all side patches to each top patch:
    max_top_sim, _ = torch.max(cos_sim, dim=0) # [256]
    sim_grid = max_top_sim.view(16, 16).cpu().numpy()

    # Normalize heatmap to 0-255
    heatmap = ((sim_grid - sim_grid.min()) / (sim_grid.max() - sim_grid.min() + 1e-6) * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (top_bgr.shape[1], top_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(top_bgr, 0.6, heatmap_color, 0.4, 0)

    # Save visual comparison proof
    proof_dir = ROOT / "data" / "reference"
    out_comparison_path = proof_dir / "cross_view_proof.jpg"

    # Create 3-panel figure: [Side Demo Grasp] | [Top Live Cam] | [DINOv2 Attention Heatmap]
    h, w = 360, 640
    p1 = cv2.resize(side_f250_bgr, (w, h))
    p2 = cv2.resize(top_bgr, (w, h))
    p3 = cv2.resize(overlay, (w, h))

    # Add text labels
    cv2.putText(p1, "INPUT 1: SIDE VIEW DEMO (Grasp)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(p2, "INPUT 2: TOP LIVE CAM (Obs)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(p3, "DINOv2 Cross-View Attention", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack([p1, p2, p3])
    cv2.imwrite(str(out_comparison_path), combined)
    print(f"\nProof image saved to: {out_comparison_path}")

if __name__ == "__main__":
    main()
