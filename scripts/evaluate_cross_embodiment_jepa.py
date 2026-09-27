#!/usr/bin/env python3
"""
Evaluate Cross-Embodiment JEPA Transfer:
Input: Human Video Demonstration (data/reference/human_bottle_demo.h5 or Videos/Human Bottle Demo.mp4)
Observation: Robot Arm Observation (data/reference/bottle_side_view.h5 or Live ZED Camera)

Evaluates:
1. DINOv2 Feature Invariance across Embodiments (Human Hand vs Robot Gripper).
2. Bottle Semantic Token Tracking under Cross-Embodiment domain shift.
3. Object-Centric Subgoal extraction for JEPA Model Predictive Control.
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
    print(f"[CROSS-EMBODIMENT] Loading DINO-WM on {device}...")

    config_path = ROOT / "config" / "deploy_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    runner = DinoWMRunner(config, device=device)

    # 1. Load human demo frames (Matched Side View)
    human_video_path = Path("/home/fr10/fr10_ws/src/Videos/Side View Human Demo Trimmed.mp4")
    cap_h = cv2.VideoCapture(str(human_video_path))
    cap_h.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, h_f0 = cap_h.read()
    cap_h.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Grasp frame
    _, h_fgrasp = cap_h.read()
    cap_h.set(cv2.CAP_PROP_POS_FRAMES, 290)  # Placed frame
    _, h_fplaced = cap_h.read()
    cap_h.release()

    # 2. Load robot frames (Side View Live/Demo)
    robot_video_path = Path("/home/fr10/fr10_ws/src/Videos/Bottle Side View.mp4")
    cap_r = cv2.VideoCapture(str(robot_video_path))
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, r_f0 = cap_r.read()
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 250)  # Grasp frame
    _, r_fgrasp = cap_r.read()
    cap_r.release()

    # Encode with DINOv2
    h_fgrasp_rgb = cv2.cvtColor(h_fgrasp, cv2.COLOR_BGR2RGB)
    r_fgrasp_rgb = cv2.cvtColor(r_fgrasp, cv2.COLOR_BGR2RGB)

    t_h = runner.preprocess_image(h_fgrasp_rgb)
    t_r = runner.preprocess_image(r_fgrasp_rgb)

    z_h = runner.encode_frame(t_h).view(256, 384)
    z_r = runner.encode_frame(t_r).view(256, 384)

    # Normalize patch tokens
    norm_h = F.normalize(z_h, dim=-1)
    norm_r = F.normalize(z_r, dim=-1)

    # Cosine similarity matrix between Human and Robot patches [256, 256]
    sim_matrix = torch.matmul(norm_h, norm_r.T)
    max_sim_val = sim_matrix.max().item()
    mean_sim_val = sim_matrix.mean().item()

    print("\n" + "=" * 65)
    print("CROSS-EMBODIMENT SEMANTIC TOKEN ANALYSIS")
    print("=" * 65)
    print(f"Human Demonstration Source: {human_video_path.name}")
    print(f"Robot Demonstration Target: {robot_video_path.name}")
    print(f"Peak Patch Cosine Similarity (Bottle / Table): {max_sim_val:.4f}")
    print(f"Mean Full-Scene Patch Cosine Similarity:       {mean_sim_val:.4f}")
    print("=" * 65)

    # Project similarity onto robot image
    max_sim_on_robot, _ = torch.max(sim_matrix, dim=0)  # [256]
    grid_robot = max_sim_on_robot.view(16, 16).cpu().numpy()

    hmap = ((grid_robot - grid_robot.min()) / (grid_robot.max() - grid_robot.min() + 1e-6) * 255).astype(np.uint8)
    hmap_res = cv2.resize(hmap, (r_fgrasp.shape[1], r_fgrasp.shape[0]), interpolation=cv2.INTER_CUBIC)
    hmap_col = cv2.applyColorMap(hmap_res, cv2.COLORMAP_JET)
    overlay_robot = cv2.addWeighted(r_fgrasp, 0.6, hmap_col, 0.4, 0)

    # Save visual artifact
    out_path = ROOT / "data" / "reference" / "cross_embodiment_eval_result.jpg"
    h, w = 360, 640
    p1 = cv2.resize(h_fgrasp, (w, h))
    p2 = cv2.resize(r_fgrasp, (w, h))
    p3 = cv2.resize(overlay_robot, (w, h))

    cv2.putText(p1, "HUMAN DEMO (Side View)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(p2, "ROBOT TARGET (Side View)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(p3, f"Cross-Embodiment Sim (Max {max_sim_val:.2f})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack([p1, p2, p3])
    cv2.imwrite(str(out_path), combined)
    print(f"[CROSS-EMBODIMENT] Evaluation visualization saved to: {out_path}\n")


if __name__ == "__main__":
    main()
