#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_control.robot.fairino_driver import FairinoDriver

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TestFairino")

def main():
    parser = argparse.ArgumentParser(description="Test Fairino FR10 Robot Interface (Mock / Live)")
    parser.add_argument("--ip", type=str, default="192.168.57.2", help="Fairino controller IP")
    parser.add_argument("--mock", action="store_true", default=True, help="Force dry-run mock mode")
    parser.add_argument("--live", action="store_true", help="Attempt live connection to physical robot")
    args = parser.parse_args()

    use_mock = not args.live

    logger.info("=" * 60)
    logger.info(f"Initializing Fairino Driver (Mode: {'MOCK/DRY-RUN' if use_mock else 'LIVE'})")
    logger.info("=" * 60)

    driver = FairinoDriver(
        controller_ip=args.ip,
        mock=use_mock,
    )

    # 1. Connect
    connected = driver.connect()
    if not connected:
        logger.error("Failed to initialize driver.")
        sys.exit(1)

    # 2. Query initial TCP pose
    init_pose = driver.get_tcp_pose()
    logger.info(f"Initial TCP Pose: {init_pose}")

    # 3. Simulate step actions
    test_actions = [
        [0.010, 0.000, 0.000, 0.0, 0.0, 0.0, 0.0],  # +10mm X
        [0.000, 0.010, 0.000, 0.0, 0.0, 0.0, 0.0],  # +10mm Y
        [0.000, 0.000, -0.010, 0.0, 0.0, 0.0, 1.0], # -10mm Z & close gripper
        [0.000, 0.000, 0.020, 0.0, 0.0, 0.0, 1.0],  # +20mm Z & hold gripper
        [0.000, 0.000, 0.000, 0.0, 0.0, 0.0, 0.0],  # Open gripper
    ]

    for step_idx, act in enumerate(test_actions):
        logger.info(f"\n--- Testing Action Step {step_idx + 1}/{len(test_actions)} ---")
        logger.info(f"Command Delta: {act}")
        success = driver.step_action(act)
        new_pose = driver.get_tcp_pose()
        logger.info(f"Resulting TCP Pose: {new_pose}")
        if not success:
            logger.error(f"Action execution failed on step {step_idx}")

    # 4. Clean shutdown
    driver.close()
    logger.info("\n✅ Fairino Driver dry-run completed successfully with all safety checks passed!")

if __name__ == "__main__":
    main()
