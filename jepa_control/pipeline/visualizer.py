import os
import cv2
import numpy as np
from typing import List, Optional

class PolicyVisualizer:
    """
    Real-time multi-panel HUD visualizer for JEPA imitation learning policies.
    Renders live observation, reference goal, progress metrics, and planned actions.
    Supports both desktop GUI display and headless video logging.
    """

    def __init__(
        self,
        window_name: str = "JEPA World Model Control - Fairino FR10",
        save_video_path: Optional[str] = None,
        show_gui: bool = False,
        fps: int = 5,
        panel_size: int = 360,
    ):
        self.window_name = window_name
        self.save_video_path = save_video_path
        self.show_gui = show_gui
        self.fps = fps
        self.panel_size = panel_size
        self.video_writer = None

    def render(
        self,
        current_obs_rgb: np.ndarray,
        reference_goal_rgb: np.ndarray,
        action_7d: List[float],
        latent_l1_dist: float,
        l1_threshold: float,
        current_tcp_pose: np.ndarray,
        step_idx: int,
        prediction_error: Optional[float] = None,
        retarget_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Builds a 3-panel dashboard:
        [ Live Observation ] | [ Reference Subgoal ] | [ Telemetry & Action ]
        """
        S = self.panel_size

        # 1. Prepare Live Camera Panel
        p1 = cv2.resize(current_obs_rgb, (S, S))
        p1_bgr = cv2.cvtColor(p1, cv2.COLOR_RGB2BGR)
        cv2.putText(p1_bgr, "LIVE OBSERVATION (o_t)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tcp_text = f"TCP: X={current_tcp_pose[0]:.0f} Y={current_tcp_pose[1]:.0f} Z={current_tcp_pose[2]:.0f}"
        cv2.putText(p1_bgr, tcp_text, (10, S - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 2. Prepare Reference Subgoal Panel
        p2 = cv2.resize(reference_goal_rgb, (S, S))
        p2_bgr = cv2.cvtColor(p2, cv2.COLOR_RGB2BGR)
        cv2.putText(p2_bgr, "REFERENCE GOAL (y_t+1)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(p2_bgr, f"Step: {step_idx}", (10, S - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 3. Prepare Telemetry Panel
        p3_bgr = np.zeros((S, S, 3), dtype=np.uint8) + 25  # Dark sleek background
        cv2.putText(p3_bgr, "JEPA TELEMETRY", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

        # Latent distance bar
        progress = max(0.0, min(1.0, 1.0 - (latent_l1_dist / max(l1_threshold, 1e-4))))
        bar_color = (0, 255, 0) if latent_l1_dist < l1_threshold else (0, 165, 255)
        cv2.putText(p3_bgr, f"Latent L1 Dist: {latent_l1_dist:.4f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.rectangle(p3_bgr, (10, 56), (S - 10, 72), (55, 55, 55), -1)
        cv2.rectangle(p3_bgr, (10, 56), (10 + int((S - 20) * progress), 72), bar_color, -1)

        # World Model 1-Step Prediction Error
        if prediction_error is not None:
            err_color = (0, 255, 0) if prediction_error < 0.15 else ((0, 255, 255) if prediction_error < 0.35 else (0, 140, 255))
            cv2.putText(p3_bgr, f"WM Pred Error: {prediction_error:.4f}", (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, err_color, 1)
        else:
            cv2.putText(p3_bgr, "WM Pred Error: Initializing", (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        # DINOv2 Latent Patch Attention Metrics
        if retarget_info:
            sim_val = retarget_info.get("cosine_similarity", 0.0)
            dr, dc = retarget_info.get("patch_delta", (0, 0))
            cv2.putText(
                p3_bgr,
                f"Patch Attn: Sim={sim_val:.3f} | d=({dr:+d},{dc:+d})",
                (10, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 220, 100),
                1,
            )

        cv2.line(p3_bgr, (10, 122), (S - 10, 122), (60, 60, 60), 1)

        # Planned 7-DoF action
        cv2.putText(p3_bgr, "Planned Action (CEM):", (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        labels = ["dX (mm)", "dY (mm)", "dZ (mm)", "dRx", "dRy", "dRz", "Gripper"]
        for idx, (lbl, val) in enumerate(zip(labels, action_7d)):
            val_str = f"{val * 1000:+.1f}" if idx < 3 else f"{val:+.3f}"
            y_pos = 164 + idx * 20
            cv2.putText(p3_bgr, f"{lbl:>8}: {val_str}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)

        # Combine panels side by side [S, 3*S, 3]
        combined = np.hstack([p1_bgr, p2_bgr, p3_bgr])

        # Save to video if configured
        if self.save_video_path:
            if self.video_writer is None:
                h, w, _ = combined.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(self.save_video_path, fourcc, self.fps, (w, h))
            self.video_writer.write(combined)

        # Display if desktop GUI enabled
        if self.show_gui:
            try:
                cv2.imshow(self.window_name, combined)
                cv2.waitKey(1)
            except Exception:
                pass

        return combined

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.show_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
