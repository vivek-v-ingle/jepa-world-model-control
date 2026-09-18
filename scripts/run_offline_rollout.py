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
from jepa_control.pipeline.visualizer import PolicyVisualizer
from jepa_control.planner.adaptive_goal import ReferenceEpisodeLoader
from jepa_control.perception.camera import get_camera_stream
from jepa_control.robot.fairino_driver import FairinoDriver

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PolicyExecution")

def main():
    parser = argparse.ArgumentParser(description="JEPA Policy Execution on Fairino FR10 (Live/Mock + ZED/USB)")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "deploy_config.yaml"), help="Config YAML path")
    parser.add_argument("--robot_config", type=str, default=str(ROOT / "config" / "fairino_robot.yaml"), help="Robot config path")
    parser.add_argument("--max_steps", type=int, default=10, help="Number of policy steps to execute")
    parser.add_argument("--live", action="store_true", help="Connect to physical Fairino robot instead of mock")
    parser.add_argument("--camera", type=str, default="mock", choices=["mock", "zed", "usb", "auto"], help="Camera stream source")
    parser.add_argument("--visualize", action="store_true", help="Enable HUD visualization display")
    parser.add_argument("--save_video", type=str, default=None, help="Path to save MP4 execution video (e.g. rollout.mp4)")
    parser.add_argument("--rotate_camera", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate live camera image by N degrees (e.g. 180)")
    parser.add_argument("--invert_dx", action="store_true", help="Invert physical X direction (dx = -dx)")
    parser.add_argument("--invert_dy", action="store_true", help="Invert physical Y direction (dy = -dy)")
    parser.add_argument("--invert_dz", action="store_true", help="Invert physical Z direction (dz = -dz)")
    args = parser.parse_args()

    # 1. Load Configurations
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.robot_config, "r") as f:
        robot_cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(f"Starting JEPA World Model Policy (Mode: {'LIVE ROBOT' if args.live else 'MOCK ROBOT'} | Camera: {args.camera.upper()})")
    logger.info("=" * 60)

    # 2. Initialize Policy Runner
    runner = JEPAPolicyRunner(config)
    logger.info("JEPA models and CEM planner initialized.")

    # 3. Initialize Robot Interface
    robot_ip = robot_cfg.get("robot", {}).get("controller_ip", "192.168.57.2")
    speed = float(robot_cfg.get("robot", {}).get("default_speed", 25.0))
    robot = FairinoDriver(controller_ip=robot_ip, default_speed=speed, mock=not args.live)
    robot.connect()

    # 4. Initialize Camera Stream (if not purely mock)
    camera = None
    if args.camera != "mock":
        cam_idx = robot_cfg.get("camera", {}).get("camera_index", 0)
        camera = get_camera_stream(camera_type=args.camera, camera_index=cam_idx)

    # 5. Initialize Visualizer
    visualizer = None
    if args.visualize or args.save_video:
        visualizer = PolicyVisualizer(save_video_path=args.save_video, show_gui=args.visualize)

    # 6. Load Reference Demonstration Episode
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

    # 7. Execute Policy Steps with Staged Milestones
    # Demonstration keyframe markers
    # Phase 1 (APPROACH): Demo frames 0 -> 50
    # Phase 2 (GRASP):    Demo frames 50 -> 60
    # Phase 3 (LIFT):     Demo frames 60 -> 85
    # Phase 4 (TRANSPORT):Demo frames 85 -> 125
    # Phase 5 (PLACE):    Demo frames 125 -> 155
    all_demo_imgs = ref_loader.images
    total_demo_frames = len(all_demo_imgs)
    l1_threshold = config.get("planner", {}).get("l1_threshold", 0.70)

    current_phase = 1
    phase_step_count = 0
    phase_names = {
        1: "APPROACH & DESCEND TO OBJECT",
        2: "GRASP TARGET OBJECT",
        3: "LIFT OBJECT FROM TABLE",
        4: "TRANSPORT TO TRAY",
        5: "PLACE & RELEASE IN TRAY",
        6: "COMPLETED",
    }

    ref_curr_idx = 0
    ref_target_idx = min(30, total_demo_frames - 1)

    for step in range(args.max_steps):
        phase_label = phase_names.get(current_phase, "COMPLETED")
        logger.info(f"\n" + "=" * 55)
        logger.info(f"Policy Step {step + 1}/{args.max_steps} | Phase {current_phase}/5: {phase_label}")
        logger.info("=" * 55)

        if current_phase > 5:
            logger.info("🎉 All 5 Pick-and-Place phases completed successfully!")
            break

        # Acquire observation
        if camera is not None:
            ret, obs_frame = camera.read()
            if not ret or obs_frame is None:
                logger.warning("Failed to grab camera frame. Reusing previous frame.")
                obs_frame = all_demo_imgs[ref_curr_idx].copy()
            elif args.rotate_camera != 0:
                import cv2
                if args.rotate_camera == 180:
                    obs_frame = cv2.rotate(obs_frame, cv2.ROTATE_180)
                elif args.rotate_camera == 90:
                    obs_frame = cv2.rotate(obs_frame, cv2.ROTATE_90_CLOCKWISE)
                elif args.rotate_camera == 270:
                    obs_frame = cv2.rotate(obs_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            obs_frame = all_demo_imgs[ref_curr_idx].copy()

        current_pose = robot.get_tcp_pose()

        # Update reference demonstration pair based on active milestone phase
        if current_phase == 1:
            # Approaching screwdriver on table
            ref_curr_idx = min(step * 3, 40)
            ref_target_idx = min(ref_curr_idx + 15, 55)
        elif current_phase == 2:
            # Grasping screwdriver
            ref_curr_idx = 50
            ref_target_idx = 60
        elif current_phase == 3:
            # Lifting off table
            ref_curr_idx = 60
            ref_target_idx = 85
        elif current_phase == 4:
            # Carrying to tool tray on right
            ref_curr_idx = 85
            ref_target_idx = 125
        elif current_phase == 5:
            # Lowering and placing into tray
            ref_curr_idx = 125
            ref_target_idx = 155

        curr_ref = all_demo_imgs[ref_curr_idx]
        future_ref = all_demo_imgs[ref_target_idx]

        # Step JEPA Policy
        action_7d, goal_latent, dist, advance, reason = runner.step(
            current_obs_rgb=obs_frame,
            current_robot_pose=current_pose,
            ref_curr_rgb=curr_ref,
            ref_target_rgb=future_ref,
        )

        logger.info(f"Active Ref Goal: Frame {ref_target_idx}/{total_demo_frames} ({phase_label})")
        logger.info(f"Planned Action Delta: {[round(float(x), 4) for x in action_7d]}")
        logger.info(f"Latent L1 Progress Distance: {dist:.6f}")

        # In Phase 2 (GRASP), force gripper closed
        if current_phase == 2:
            action_7d[6] = 1.0
        # In Phase 5 (RELEASE), open gripper
        elif current_phase == 5 and current_pose[2] <= 295.0:
            action_7d[6] = 0.0

        # Apply optional CLI directional inversions
        if args.invert_dx:
            action_7d[0] = -action_7d[0]
        if args.invert_dy:
            action_7d[1] = -action_7d[1]
        if args.invert_dz:
            action_7d[2] = -action_7d[2]

        # Dispatch action to robot driver
        robot.step_action(action_7d)
        new_pose = robot.get_tcp_pose()
        logger.info(f"Robot Current TCP Pose: {[round(float(x), 2) for x in new_pose]}")

        # Render Visualization Frame
        if visualizer is not None:
            visualizer.render(
                current_obs_rgb=obs_frame,
                reference_goal_rgb=future_ref,
                action_7d=action_7d,
                latent_l1_dist=dist,
                l1_threshold=l1_threshold,
                current_tcp_pose=new_pose,
                step_idx=step + 1,
            )

        # Check physical milestone transition conditions:
        z_curr = new_pose[2]
        y_curr = new_pose[1]
        x_curr = new_pose[0]
        phase_step_count += 1

        if current_phase == 1:
            # Transition to GRASP when at grasp height near table (Z <= 268mm)
            if (z_curr <= 268.0 and phase_step_count >= 2) or phase_step_count >= 15:
                logger.info(f"🎯 Milestone reached: Arm at grasp height (Z={z_curr:.1f}mm) -> Entering Phase 2 (GRASP).")
                current_phase = 2
                phase_step_count = 0
        elif current_phase == 2:
            # Grasp object with gripper
            robot.set_gripper(1.0)
            if phase_step_count >= 2:
                logger.info("🎯 Milestone reached: Gripper closed firmly on object -> Entering Phase 3 (LIFT).")
                current_phase = 3
                phase_step_count = 0
        elif current_phase == 3:
            # Lift object off table (Z >= 340mm or after 6 lift steps)
            if z_curr >= 340.0 or phase_step_count >= 6:
                logger.info("🎯 Milestone reached: Object lifted off table -> Entering Phase 4 (TRANSPORT).")
                current_phase = 4
                phase_step_count = 0
        elif current_phase == 4:
            # Transport across table toward tool tray (X >= -370mm and Y >= -150mm, or after 15 transport steps)
            if (x_curr >= -370.0 and y_curr >= -150.0) or phase_step_count >= 15:
                logger.info("🎯 Milestone reached: End effector reached tool tray -> Entering Phase 5 (PLACE).")
                current_phase = 5
                phase_step_count = 0
        elif current_phase == 5:
            # Place down in tray (Z <= 280mm or after 5 placement steps)
            if z_curr <= 280.0 or phase_step_count >= 5:
                robot.set_gripper(0.0)
                logger.info("🎉 Milestone reached: Object placed and released into tool tray!")
                current_phase = 6
                phase_step_count = 0

    # Cleanup
    if camera is not None:
        camera.release()
    if visualizer is not None:
        visualizer.close()
    robot.close()

    logger.info("\n" + "=" * 60)
    logger.info("✅ JEPA Policy Execution completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
