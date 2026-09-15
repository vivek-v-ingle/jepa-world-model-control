"""
3D Object Detector for ZED 2 Camera & Fairino FR10 Manipulator.

Uses ZED 2 stereo depth sensing (sl.MEASURE.XYZRGBA) to extract 3D Cartesian
coordinates (X, Y, Z in mm) of tabletop objects and map them to the robot base frame.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False


class ZED3DObjectDetector:
    """
    Detects tabletop objects using ZED 2 RGB-D point cloud measure,
    extracts 3D Cartesian coordinates (X_cam, Y_cam, Z_cam) in mm,
    and maps coordinates into the Fairino FR10 base frame.
    """

    def __init__(
        self,
        resolution: str = "HD720",
        depth_mode: str = "NEURAL",
        min_depth_mm: float = 300.0,
        max_depth_mm: float = 2000.0,
    ):
        if not ZED_SDK_AVAILABLE:
            raise RuntimeError("pyzed.sl is not installed. Please install ZED SDK.")

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()

        res_dict = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        depth_dict = {
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.ULTRA),
        }

        self.init_params.camera_resolution = res_dict.get(resolution, sl.RESOLUTION.HD720)
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = depth_dict.get(depth_mode, sl.DEPTH_MODE.PERFORMANCE)
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER

        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"[ZED3D] Failed to open ZED camera: {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.image_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()

        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm

        # Default camera-to-robot base translation/rotation calibration fallback
        # In overhead setup: X_robot maps to Y_cam/Z_cam offset, Y_robot maps to X_cam
        logger.info("[ZED3D] ZED 3D Object Detector initialized successfully.")

    def capture_frame_and_cloud(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Grabs RGB frame and 3D XYZRGBA point cloud mat from ZED 2.
        Returns:
            (success, rgb_image_np [H,W,3], point_cloud_np [H,W,4])
        """
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)

            rgba_np = self.image_mat.get_data()
            rgb_np = cv2.cvtColor(rgba_np, cv2.COLOR_BGRA2RGB)
            cloud_np = self.point_cloud_mat.get_data() # [H, W, 4] -> X, Y, Z, color

            return True, rgb_np, cloud_np

        return False, None, None

    def detect_tabletop_object_3d(
        self,
        bgr_image: np.ndarray,
        cloud_np: np.ndarray,
        crop_table_roi: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Detects the primary object on the tabletop and returns its 3D Cartesian coordinates.

        Returns dict:
            {
                "bbox": [x, y, w, h],
                "centroid_px": (u, v),
                "cam_3d_mm": (X_cam, Y_cam, Z_cam),
                "robot_3d_mm": (X_robot, Y_robot, Z_robot),
                "confidence": float,
            }
        """
        h, w, _ = bgr_image.shape

        # Convert to HSV for robust object foreground detection
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Focus ROI on tabletop surface (exclude background walls/edges)
        mask = np.zeros((h, w), dtype=np.uint8)
        if crop_table_roi:
            # Table ROI box in 720p frame
            mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = 255
        else:
            mask[:] = 255

        # Adaptive thresholding to isolate tabletop object
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )
        thresh = cv2.bitwise_and(thresh, mask)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("[ZED3D] No object contours found in table ROI.")
            return None

        # Sort contours by area, pick prominent tabletop object (filtering tiny noise)
        valid_contours = [c for c in contours if 300 < cv2.contourArea(c) < 50000]
        if not valid_contours:
            logger.warning("[ZED3D] No valid-sized object contours found.")
            return None

        largest_contour = max(valid_contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_contour)
        u_center = bx + bw // 2
        v_center = by + bh // 2

        # Extract 3D camera coordinates (X_cam, Y_cam, Z_cam) from point cloud at centroid
        xyz_point = cloud_np[v_center, u_center, :3]
        x_cam, y_cam, z_cam = xyz_point[0], xyz_point[1], xyz_point[2]

        # Handle invalid depth pixels (NaN/Inf) by median neighborhood sampling
        if not np.all(np.isfinite(xyz_point)) or z_cam <= 0:
            patch = cloud_np[max(0, v_center - 5):min(h, v_center + 5), max(0, u_center - 5):min(w, u_center + 5), :3]
            patch_valid = patch[np.isfinite(patch[:, :, 2]) & (patch[:, :, 2] > 0)]
            if len(patch_valid) > 0:
                median_xyz = np.median(patch_valid, axis=0)
                x_cam, y_cam, z_cam = median_xyz[0], median_xyz[1], median_xyz[2]
            else:
                logger.warning("[ZED3D] Invalid 3D depth at object centroid.")
                return None

        # Map Camera 3D coordinates (mm) to Fairino Robot Base Frame coordinates (mm)
        # Overhead camera transform:
        # X_robot = -Y_cam + offset_x
        # Y_robot = -X_cam + offset_y
        # Z_robot = table surface height (~250-270 mm)
        x_robot = float(-y_cam - 150.0)
        y_robot = float(-x_cam - 50.0)
        z_robot = float(270.0) # Safe grasp approach height above table

        res = {
            "bbox": [bx, by, bw, bh],
            "centroid_px": (u_center, v_center),
            "cam_3d_mm": (float(x_cam), float(y_cam), float(z_cam)),
            "robot_3d_mm": (x_robot, y_robot, z_robot),
            "contour": largest_contour,
        }

        logger.info(
            f"[ZED3D] Object Detected @ Px({u_center}, {v_center}) | "
            f"Cam3D: [{x_cam:.1f}, {y_cam:.1f}, {z_cam:.1f}] mm | "
            f"Robot3D: [{x_robot:.1f}, {y_robot:.1f}, {z_robot:.1f}] mm"
        )
        return res

    def release(self):
        self.zed.close()
        logger.info("[ZED3D] ZED Camera closed.")
