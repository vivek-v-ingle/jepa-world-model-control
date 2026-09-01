#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import h5py

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_control.pipeline.policy_runner import JEPAPolicyRunner
from jepa_control.planner.adaptive_goal import ReferenceEpisodeLoader
from jepa_control.robot.fairino_driver import FairinoDriver

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("OfflineRollout")

def main():
    parser = argparse.ArgumentParser(description="Run Offline JEPA Policy Execution & Trajectory Generation")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "deploy_config.yaml"), help="Config YAML path")
    parser.add_argument("--robot_config", type=str, default=str(ROOT / "config" / "fairino_robot.yaml"), help="Robot config path")
    parser.add_argument("--max_steps", type=int, default=5, help="Number of policy steps to execute")
    parser.add_argument("--live", action="store_true", help="Connect to live Fairino robot instead of mock")
    args = parser.parse_args()

    # 1. Load Configurations
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.robot_config, "r") as f:
        robot_cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Starting JEPA World Model Offline Policy Rollout")
    logger.info("=" * 60)

    # 2. Initialize Policy Runner
    runner = JEPAPolicyRunner(config)
    logger.info("JEPA models and CEM planner initialized.")

    # 3. Initialize Robot Interface
    robot_ip = robot_cfg.get("robot", {}).get("controller_ip", "192.168.57.2")
    robot = FairinoDriver(controller_ip=robot_ip, mock=not args.live)
    robot.connect()

    # 4. Load Reference Demonstration Episode
    ref_cfg = config.get("reference", {})
    ref_h5 = ref_cfg.get("reference_h5")
    image_key = ref_cfg.get("image_key", "observations/images/camera_front")
    ref_loader = ReferenceEpisodeLoader(
        h5_path=ref_h5,
        image_key=image_key,
        data_fps=ref_cfg.get("ref_data_fps", 30),
        target_fps=ref_cfg.get("ref_target_fps", 5),
    )
    logger.info(f"Loaded source demonstration: {ref_h5} ({ref_loader.length} frames)")

    # 5. Execute Policy Steps
    curr_ref, future_ref = ref_loader.get_reference_pair(advance=False)
    
    # Use initial frame as mock observation if offline
    mock_obs = curr_ref.copy()

    for step in range(args.max_steps):
        logger.info(f"\n" + "=" * 40)
        logger.info(f"Policy Step {step + 1}/{args.max_steps}")
        logger.info("=" * 40)

        current_pose = robot.get_tcp_pose()

        # Step JEPA Policy
        action_7d, goal_latent, dist = runner.step(
            current_obs_rgb=mock_obs,
            current_robot_pose=current_pose,
            ref_curr_rgb=curr_ref,
            ref_target_rgb=future_ref,
        )

        logger.info(f"Latent Goal Shape: {list(goal_latent.shape)}")
        logger.info(f"Planned Action Delta: {[round(float(x), 4) for x in action_7d]}")
        logger.info(f"Latent L1 Progress Distance: {dist:.6f}")

        # Dispatch action to robot driver
        robot.step_action(action_7d)
        new_pose = robot.get_tcp_pose()
        logger.info(f"Robot Current TCP Pose: {[round(float(x), 2) for x in new_pose]}")

        # Advance reference frame if progress achieved
        advance = dist < config.get("planner", {}).get("l1_threshold", 1.0)
        if advance:
            logger.info("Subgoal threshold reached -> Advancing reference demonstration frame.")
            curr_ref, future_ref = ref_loader.get_reference_pair(advance=True)
        else:
            logger.info("Subgoal not yet reached -> Retrying current demonstration frame.")

    robot.close()
    logger.info("\n" + "=" * 60)
    logger.info("✅ JEPA World Model offline rollout completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
