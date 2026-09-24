"""
Screwdriver visual detector and tabletop grounding module.
Uses the overhead ZED camera stream to detect the live position of the screwdriver
and computes Cartesian approach offsets relative to the reference demonstration.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Nominal demonstration coordinates (from reference frame 0 in fr10_pick_and_place.h5)
NOMINAL_U0 = 696.4  # Pixel X in 1280x720 ZED frame (demo screwdriver handle centroid)
NOMINAL_V0 = 296.5  # Pixel Y in 1280x720 ZED frame (demo screwdriver handle centroid)
NOMINAL_ROBOT_X0 = -241.55  # mm in robot base frame
NOMINAL_ROBOT_Y0 = -797.98  # mm in robot base frame
NOMINAL_RZ0 = 0.58

# Live calibrated ground truth anchor coordinates
LIVE_U0 = 671.2
LIVE_V0 = 209.0
LIVE_ROBOT_X0 = -212.0
LIVE_ROBOT_Y0 = -970.0
LIVE_RZ0 = 10.0
DEFAULT_APPROACH_Z = 120.0   # mm above tabletop


def detect_screwdriver(
    image_rgb: np.ndarray,
    table_roi: Tuple[int, int, int, int] = (420, 120, 750, 500),
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect the yellow/orange handle of the screwdriver on the table.
    
    Args:
        image_rgb: [H=720, W=1280, 3] RGB image from ZED camera.
        table_roi: (min_x, min_y, max_x, max_y) bounding box of the pickable tabletop region.
        
    Returns:
        (center_u, center_v, width, height) in image pixel coordinates, or None if not found.
    """
    if image_rgb is None or image_rgb.size == 0:
        return None

    min_x, min_y, max_x, max_y = table_roi
    h, w, _ = image_rgb.shape

    # Clamp ROI to image dimensions
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(w, max_x)
    max_y = min(h, max_y)

    roi = image_rgb[min_y:max_y, min_x:max_x]
    
    # 1. HSV Saturation segmentation:
    # Table surface is neutral brushed aluminum (saturation < 30).
    # Screwdriver handle has distinct vibrant yellow/orange saturation (sat > 50).
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    candidate_mask = ((sat > 50) & (val > 50)).astype(np.uint8)

    # Clean noise with morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    
    best_candidate = None
    best_area = 0

    # Look for candidates matching typical handle footprint (30 to 3000 pixels)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        
        if 30 < area < 3000 and area > best_area:
            best_area = area
            local_cx, local_cy = centroids[i]
            best_candidate = (
                float(local_cx + min_x),
                float(local_cy + min_y),
                float(cw),
                float(ch),
            )

    return best_candidate


def compute_grounded_pick_pose(
    image_rgb: np.ndarray,
    nominal_x: float = NOMINAL_ROBOT_X0,
    nominal_y: float = NOMINAL_ROBOT_Y0,
    z_height: float = DEFAULT_APPROACH_Z,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """
    Computes the 6D Cartesian pre-approach waypoint based on live visual detection.
    
    Returns:
        (target_pose_6d, (center_u, center_v))
    """
    det = detect_screwdriver(image_rgb)
    
    if det is None:
        logger.warning("[GROUNDING] Screwdriver not visually segmented. Using nominal pick pose.")
        return np.array([nominal_x, nominal_y, z_height, -175.05, -1.46, NOMINAL_RZ0], dtype=np.float32), None

    cu, cv, cw, ch = det
    logger.info(f"[GROUNDING] Screwdriver detected at U={cu:.1f}, V={cv:.1f} (size={cw:.0f}x{ch:.0f} px)")

    # Compute pixel delta from verified live anchor position
    du = cu - LIVE_U0
    dv = cv - LIVE_V0

    # Calibrated affine mapping from tabletop pixel displacements to robot base coordinates:
    delta_x = -0.34 * dv + 0.85 * du
    delta_y = 1.96 * dv + 0.45 * du
    delta_rz = 0.10 * du

    grounded_x = float(LIVE_ROBOT_X0 + delta_x)
    grounded_y = float(LIVE_ROBOT_Y0 + delta_y)
    grounded_rz = float(LIVE_RZ0 + delta_rz)

    # Safe physical bounds within reachable tabletop envelope
    grounded_x = float(np.clip(grounded_x, -350.0, -200.0))
    grounded_y = float(np.clip(grounded_y, -1000.0, -700.0))
    grounded_rz = float(np.clip(grounded_rz, -15.0, 25.0))

    logger.info(
        f"[GROUNDING] Visual offset from anchor: dU={du:.1f}px, dV={dv:.1f}px -> "
        f"Grounded Approach Waypoint: X={grounded_x:.1f}, Y={grounded_y:.1f}, Z={z_height:.1f}mm, Rz={grounded_rz:.1f}°"
    )

    pose = np.array([grounded_x, grounded_y, z_height, -175.05, -1.46, grounded_rz], dtype=np.float32)
    return pose, (cu, cv)

