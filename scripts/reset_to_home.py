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

DEFAULT_READY_POSE = np.array([-460.78, -212.45, 257.16, 174.50, 3.79, -11.64, 0.0], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Reset Fairino FR10 to Tabletop Ready Pose")
    parser.add_argument("--z", type=float, default=257.16, help="Target Z height in mm (default: 257.16)")
    args = parser.parse_args()

    ready_pose = DEFAULT_READY_POSE.copy()
    ready_pose[2] = args.z

    logger.info("=" * 60)
    logger.info("Moving Fairino FR10 to Tabletop Ready Pose")
    logger.info(f"Target Pose: {ready_pose}")
    logger.info("=" * 60)

    driver = FairinoDriver(controller_ip="192.168.57.2", mock=False)
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
    driver.wait_for_motion_completion(timeout_sec=10.0)

    final_pose = driver.get_tcp_pose()
    logger.info(f"Final TCP Pose: {[round(float(x), 2) for x in final_pose]}")

    driver.close()
    if success:
        logger.info("✅ Robot successfully reset to ready pose.")
    else:
        logger.error("❌ Reset motion failed safety checks.")

if __name__ == "__main__":
    main()
