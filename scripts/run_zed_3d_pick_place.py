#!/usr/bin/env python3
"""
Direct 3D Perception & Deterministic Pick-and-Place Execution Script
for ZED 2 Depth Camera + Fairino FR10 Manipulator + JODELL RG Gripper.
"""

import sys
import time
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_control.perception.zed_3d_detector import ZED3DObjectDetector
from jepa_control.robot.fairino_driver import FairinoDriver

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ZED3D_PickPlace")


def main():
    parser = argparse.ArgumentParser(description="ZED 3D Direct Object Detection & Pick-and-Place")
    parser.add_argument("--ip", type=str, default="192.168.57.2", help="Fairino robot controller IP")
    parser.add_argument("--live", action="store_true", help="Connect to physical Fairino robot")
    parser.add_argument("--visualize", action="store_true", default=True, help="Display OpenCV 3D object detection window")
    parser.add_argument("--goal_x", type=float, default=-400.0, help="Goal box dropoff X coordinate (mm)")
    parser.add_argument("--goal_y", type=float, default=250.0, help="Goal box dropoff Y coordinate (mm)")
    args = parser.parse_args()

    use_mock = not args.live

    logger.info("=" * 60)
    logger.info(f"Starting ZED 3D Direct Pick & Place (Mode: {'LIVE ROBOT' if args.live else 'MOCK ROBOT'})")
    logger.info("=" * 60)

    # 1. Initialize Robot Driver
    driver = FairinoDriver(
        controller_ip=args.ip,
        mock=use_mock,
        min_z_mm=245.0,  # Enforce table floor safety
        max_cartesian_step_mm=50.0,
    )
    if not driver.connect():
        logger.error("Failed to connect to Fairino robot driver.")
        sys.exit(1)

    # 2. Initialize ZED 3D Perception
    detector = None
    if not use_mock:
        try:
            detector = ZED3DObjectDetector(resolution="HD720", depth_mode="NEURAL")
        except Exception as exc:
            logger.warning(f"Could not initialize ZED 3D detector ({exc}). Using mock detection.")

    # 3. Detect Target Object 3D Location
    target_3d = None
    if detector is not None:
        logger.info("[VISION] Grabbing 3D point cloud & searching for target object...")
        for _ in range(10): # Try up to 10 grabs for stable depth point cloud
            ret, rgb_img, cloud_np = detector.capture_frame_and_cloud()
            if ret and rgb_img is not None and cloud_np is not None:
                bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                det_res = detector.detect_tabletop_object_3d(bgr, cloud_np)
                if det_res is not None:
                    target_3d = det_res["robot_3d_mm"]
                    logger.info(f"✅ Target Object 3D Location Found: {target_3d} (Robot Base Frame)")

                    if args.visualize:
                        cv2.rectangle(bgr, (det_res["bbox"][0], det_res["bbox"][1]), 
                                      (det_res["bbox"][0] + det_res["bbox"][2], det_res["bbox"][1] + det_res["bbox"][3]), (0, 255, 0), 2)
                        cv2.circle(bgr, det_res["centroid_px"], 5, (0, 0, 255), -1)
                        cv2.putText(bgr, f"Target 3D: {target_3d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("ZED 3D Object Perception", bgr)
                        cv2.waitKey(2000)
                    break
            time.sleep(0.1)

    if target_3d is None:
        logger.warning("Target 3D location not detected by camera. Using tabletop default estimate.")
        target_3d = (-500.0, -100.0, 270.0)

    obj_x, obj_y, _ = target_3d

    # 4. Execute 4-Phase Pick & Place Sequence
    try:
        # Step A: Ensure Gripper OPEN
        logger.info("\n--- Phase 0: Opening Gripper ---")
        driver.set_gripper(0.0)
        time.sleep(1.0)

        # Step B: Hover / Approach over Object
        logger.info(f"\n--- Phase 1: Approaching Hover Pose over Object ({obj_x:.1f}, {obj_y:.1f}, 450.0) ---")
        hover_pose = np.array([obj_x, obj_y, 450.0, 174.28, 3.93, -11.69, 0.0], dtype=np.float32)
        curr_pose = driver.get_tcp_pose()
        delta_hover = hover_pose[:3] - curr_pose[:3]
        action_hover = [delta_hover[0] / 120.0, delta_hover[1] / 120.0, -delta_hover[2] / 220.0, 0.0, 0.0, 0.0, 0.0]
        driver.step_action(action_hover)
        time.sleep(1.5)

        # Step C: Descend & Clamp
        logger.info(f"\n--- Phase 2: Descending to Grasp Height ({obj_x:.1f}, {obj_y:.1f}, 270.0) ---")
        grasp_pose = np.array([obj_x, obj_y, 270.0, 174.28, 3.93, -11.69, 1.0], dtype=np.float32)
        curr_pose = driver.get_tcp_pose()
        delta_grasp = grasp_pose[:3] - curr_pose[:3]
        action_grasp = [delta_grasp[0] / 120.0, delta_grasp[1] / 120.0, -delta_grasp[2] / 220.0, 0.0, 0.0, 0.0, 1.0]
        driver.step_action(action_grasp)
        time.sleep(1.5)

        logger.info("Closing JODELL RG Gripper on Object...")
        driver.set_gripper(1.0) # Close gripper
        time.sleep(2.0)

        # Step D: Lift Object
        logger.info("\n--- Phase 3: Lifting Object (Z=480.0 mm) ---")
        lift_action = [0.0, 0.0, -210.0 / 220.0, 0.0, 0.0, 0.0, 1.0]
        driver.step_action(lift_action)
        time.sleep(1.5)

        # Step E: Transfer to Goal Box & Release
        logger.info(f"\n--- Phase 4: Transferring to Goal Dropoff ({args.goal_x}, {args.goal_y}) ---")
        curr_pose = driver.get_tcp_pose()
        goal_pose = np.array([args.goal_x, args.goal_y, 300.0, 174.28, 3.93, -11.69, 0.0], dtype=np.float32)
        delta_goal = goal_pose[:3] - curr_pose[:3]
        action_goal = [delta_goal[0] / 120.0, delta_goal[1] / 120.0, -delta_goal[2] / 220.0, 0.0, 0.0, 0.0, 0.0]
        driver.step_action(action_goal)
        time.sleep(2.0)

        logger.info("Opening Gripper to Release Object...")
        driver.set_gripper(0.0)
        time.sleep(1.0)

        logger.info("\n============================================================")
        logger.info("✅ Direct 3D Pick-and-Place Task Completed Successfully!")
        logger.info("============================================================")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    finally:
        if detector is not None:
            detector.release()
        driver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
