#!/usr/bin/env python3
"""
Safely moves the Fairino FR10 arm back to the standard tabletop ready pose:
[-460.0, -230.0, 500.0, 174.28, 3.93, -11.68, 0.0]
and opens the JODELL RG gripper.
"""
import sys
import time
import logging
from pathlib import Path
import numpy as np

import argparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_control.robot.fairino_driver import FairinoDriver

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ResetHome")

DEFAULT_HOME_POSE = np.array([-673.08, -424.72, 174.42, 179.90, 0.86, -1.87, 0.0], dtype=np.float32)
ABOVE_PICK_POSE = np.array([-245.69, -899.88, 140.91, -178.83, -3.37, 81.05, 0.0], dtype=np.float32)

POINTS_FILE = Path("/home/fr10/fr10_ws/src/Initial commands for fairino/pick_place_points.json")
if POINTS_FILE.exists():
    try:
        import json
        _pts = json.loads(POINTS_FILE.read_text(encoding="utf-8"))
        if "home" in _pts:
            DEFAULT_HOME_POSE[:6] = np.array(_pts["home"][:6], dtype=np.float32)
        if "pick" in _pts:
            _pick = np.array(_pts["pick"][:6], dtype=np.float32)
            _app_z = float(_pts.get("approach_z_mm", 60.0))
            ABOVE_PICK_POSE[:6] = _pick
            ABOVE_PICK_POSE[2] = _pick[2] + _app_z
    except Exception as _e:
        logger.warning(f"Could not load points from {POINTS_FILE}: {_e}")

def main():
    parser = argparse.ArgumentParser(description="Reset Fairino FR10 to Tabletop Ready Pose or Above Pick Pose")
    parser.add_argument("--z", type=float, default=None, help="Target Z height in mm")
    parser.add_argument("--pick", action="store_true", help="Reset directly above the pick location (pre-approach)")
    parser.add_argument("--grounding", action="store_true", help="Perform live visual object grounding")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without physical robot")
    args = parser.parse_args()

    if args.pick:
        target_name = "ABOVE TARGET OBJECT (Pre-Approach)"
        ready_pose = ABOVE_PICK_POSE.copy()
        if args.grounding and not args.mock:
            try:
                from jepa_control.perception.camera import get_camera_stream
                from jepa_control.perception.screwdriver_detector import compute_grounded_pick_pose
                cam = get_camera_stream(camera_type="zed")
                ret, frame = cam.read()
                cam.release()
                if ret and frame is not None:
                    g_pose, coords = compute_grounded_pick_pose(
                        frame,
                        nominal_x=float(ABOVE_PICK_POSE[0]),
                        nominal_y=float(ABOVE_PICK_POSE[1]),
                        z_height=float(ABOVE_PICK_POSE[2]),
                    )
                    if coords is not None:
                        ready_pose[:6] = g_pose[:6]
                        logger.info(f"🎯 Visually grounded object at pixel {coords} -> Pick Pose: {[round(float(x), 2) for x in ready_pose[:6]]}")
                    else:
                        logger.info(f"ℹ️ Visual detector found no object; using taught waypoint: {[round(float(x), 2) for x in ready_pose[:6]]}")
            except Exception as exc:
                logger.warning(f"Visual grounding fallback to nominal: {exc}")
        if args.z is not None:
            ready_pose[2] = args.z
    else:
        target_name = "HOME READY POSE (Table Center)"
        ready_pose = DEFAULT_HOME_POSE.copy()
        if args.z is not None:
            ready_pose[2] = args.z

    logger.info("=" * 60)
    logger.info(f"Moving Fairino FR10 to {target_name} ({'MOCK' if args.mock else 'LIVE'})")
    logger.info(f"Target Pose: {ready_pose}")
    logger.info("=" * 60)

    driver = FairinoDriver(controller_ip="192.168.57.2", mock=args.mock)
    if not driver.connect():
        logger.error("Failed to connect to Fairino robot.")
        sys.exit(1)

    curr_pose = driver.get_tcp_pose()
    logger.info(f"Current TCP Pose: {[round(float(x), 2) for x in curr_pose]}")

    # 1. Open gripper
    driver.set_gripper(0.0)
    time.sleep(0.5)

    # 2. Smoothly move to ready pose
    logger.info("Executing MoveL to ready pose at safe speed (15%)...")
    success = driver._move_linear(ready_pose[:6], speed=15.0)
    time.sleep(1.0)
    driver.wait_for_motion_completion(timeout_sec=25.0)

    final_pose = driver.get_tcp_pose()
    logger.info(f"Final TCP Pose: {[round(float(x), 2) for x in final_pose]}")

    driver.close()
    if success:
        logger.info("✅ Robot successfully reset to ready pose.")
    else:
        logger.error("❌ Reset motion failed safety checks.")

if __name__ == "__main__":
    main()
