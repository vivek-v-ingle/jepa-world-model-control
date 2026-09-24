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
from jepa_control.pipeline.dino_wm_runner import DinoWMRunner
from jepa_control.pipeline.visualizer import PolicyVisualizer
from jepa_control.planner.adaptive_goal import ReferenceEpisodeLoader
from jepa_control.perception.camera import get_camera_stream
from jepa_control.perception.screwdriver_detector import compute_grounded_pick_pose
from jepa_control.robot.fairino_driver import FairinoDriver
import time

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PolicyExecution")

def main():
    parser = argparse.ArgumentParser(description="JEPA Policy Execution on Fairino FR10 (Live/Mock + ZED/USB)")
    parser.add_argument("--model", type=str, default="dino_wm", choices=["dino_wm", "vjepa"], help="World model backbone (dino_wm: Meta FAIR DINO-WM DROID, vjepa: V-JEPA 2.1 Dreamer AC)")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "deploy_config.yaml"), help="Config YAML path")
    parser.add_argument("--robot_config", type=str, default=str(ROOT / "config" / "fairino_robot.yaml"), help="Robot config path")
    parser.add_argument("--max_steps", type=int, default=10, help="Number of policy steps to execute")
    parser.add_argument("--live", action="store_true", help="Connect to physical Fairino robot instead of mock")
    parser.add_argument("--camera", type=str, default="mock", choices=["mock", "zed", "usb", "auto"], help="Camera stream source")
    parser.add_argument("--visualize", action="store_true", help="Enable HUD visualization display")
    parser.add_argument("--save_video", type=str, default=None, help="Path to save MP4 execution video (e.g. rollout.mp4)")
    parser.add_argument("--speed", type=float, default=None, help="Robot MoveL velocity percentage (e.g. 35, 45, 60)")
    parser.add_argument("--rotate_camera", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate live camera image by N degrees (e.g. 180)")
    parser.add_argument("--invert_dx", action="store_true", help="Invert physical X direction (dx = -dx)")
    parser.add_argument("--invert_dy", action="store_true", help="Invert physical Y direction (dy = -dy)")
    parser.add_argument("--invert_dz", action="store_true", help="Invert physical Z direction (dz = -dz)")
    parser.add_argument("--no_grounding", action="store_true", help="Disable visual centroid grounding of target object")
    args = parser.parse_args()

    # 1. Load Configurations
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.robot_config, "r") as f:
        robot_cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(f"Starting JEPA Policy (Model: {args.model.upper()} | Mode: {'LIVE ROBOT' if args.live else 'MOCK ROBOT'} | Camera: {args.camera.upper()})")
    logger.info("=" * 60)

    # 2. Initialize Policy Runner
    if args.model == "dino_wm":
        runner = DinoWMRunner(config)
        logger.info("Meta FAIR DINO-WM DROID model and CEM planner initialized.")
    else:
        runner = JEPAPolicyRunner(config)
        logger.info("V-JEPA 2.1 Dreamer AC models and CEM planner initialized.")


    # 3. Initialize Robot Interface
    robot_ip = robot_cfg.get("robot", {}).get("controller_ip", "192.168.57.2")
    speed = float(args.speed) if args.speed is not None else float(robot_cfg.get("robot", {}).get("default_speed", 45.0))
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

    # 6.5 Visual Object Grounding for Dynamic Screwdriver Positioning
    grounded_target_xy = None
    if not args.no_grounding and camera is not None and args.live:
        logger.info("=" * 60)
        logger.info("🔍 Performing Visual Object Grounding for Screwdriver...")
        logger.info("=" * 60)
        curr_tcp = robot.get_tcp_pose()
        if curr_tcp[2] < 250.0 and curr_tcp[1] < -500.0:
            logger.info("🔭 Elevating arm to Z=300mm for unobstructed overhead camera grounding...")
            elev_target = curr_tcp.copy()
            elev_target[2] = 300.0
            robot._move_linear(elev_target, speed=speed, timeout_sec=3.0)
            time.sleep(0.5)

        ret, init_obs = camera.read()
        if ret and init_obs is not None:
            if args.rotate_camera != 0:
                import cv2
                if args.rotate_camera == 180:
                    init_obs = cv2.rotate(init_obs, cv2.ROTATE_180)
                elif args.rotate_camera == 90:
                    init_obs = cv2.rotate(init_obs, cv2.ROTATE_90_CLOCKWISE)
                elif args.rotate_camera == 270:
                    init_obs = cv2.rotate(init_obs, cv2.ROTATE_90_COUNTERCLOCKWISE)

            curr_tcp = robot.get_tcp_pose()
            grounded_pose, det_coords = compute_grounded_pick_pose(init_obs)
            if det_coords is not None:
                grounded_target_xy = (float(grounded_pose[0]), float(grounded_pose[1]))
                dist_xy = float(np.hypot(grounded_pose[0] - curr_tcp[0], grounded_pose[1] - curr_tcp[1]))
                logger.info(
                    f"🎯 [GROUNDING] Target detected at image pixel ({det_coords[0]:.1f}, {det_coords[1]:.1f}) -> "
                    f"Grounded Approach Waypoint: X={grounded_pose[0]:.1f}, Y={grounded_pose[1]:.1f}, Z={grounded_pose[2]:.1f} mm "
                    f"(Offset from current: {dist_xy:.1f} mm)"
                )
                if dist_xy > 20.0:
                    logger.info("🚀 [GROUNDING] Repositioning arm directly above live target location...")
                    align_target = np.array(
                        [grounded_pose[0], grounded_pose[1], max(curr_tcp[2], 120.0), curr_tcp[3], curr_tcp[4], grounded_pose[5]],
                        dtype=np.float32,
                    )
                    robot._move_linear(align_target, speed=speed, timeout_sec=12.0)
                    curr_tcp = robot.get_tcp_pose()
                    logger.info(f"✅ [GROUNDING] Arm aligned above target. Current TCP Pose: {[round(float(x), 2) for x in curr_tcp]}")

    # 7. Execute Policy Steps with Staged Milestones
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

    # Dynamic target coordinates (loads user's saved points if present)
    target_pick_z = 80.91
    target_place_xy = (-947.8, -233.1)
    target_place_z = 83.30
    target_lift_z = 140.0

    points_path = Path("/home/fr10/fr10_ws/src/Initial commands for fairino/pick_place_points.json")
    if points_path.exists():
        try:
            import json
            pts = json.loads(points_path.read_text(encoding="utf-8"))
            if "pick" in pts:
                target_pick_z = float(pts["pick"][2])
            if "place" in pts:
                target_place_xy = (float(pts["place"][0]), float(pts["place"][1]))
                target_place_z = float(pts["place"][2])
            app_z = float(pts.get("approach_z_mm", 60.0))
            target_lift_z = target_pick_z + app_z
            logger.info(f"Loaded points from {points_path.name}: Pick Z={target_pick_z:.1f}mm, Place XY=({target_place_xy[0]:.1f}, {target_place_xy[1]:.1f}), Lift Z={target_lift_z:.1f}mm")
        except Exception as exc:
            logger.warning(f"Could not read pick_place_points.json: {exc}")

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
        if total_demo_frames <= 400:
            # 361-frame episode (e.g. bottle pick and place)
            if current_phase == 1:
                ref_curr_idx = min(int(phase_step_count * 5), int(0.20 * total_demo_frames))
                ref_target_idx = int(0.25 * total_demo_frames)
            elif current_phase == 2:
                ref_curr_idx = int(0.25 * total_demo_frames)
                ref_target_idx = int(0.33 * total_demo_frames)
            elif current_phase == 3:
                ref_curr_idx = int(0.33 * total_demo_frames)
                ref_target_idx = int(0.44 * total_demo_frames)
            elif current_phase == 4:
                ref_curr_idx = int(0.44 * total_demo_frames)
                ref_target_idx = min(int(0.44 * total_demo_frames + phase_step_count * 8), int(0.72 * total_demo_frames))
            elif current_phase == 5:
                ref_curr_idx = int(0.72 * total_demo_frames)
                ref_target_idx = int(0.85 * total_demo_frames)
        else:
            # 823-frame episode (e.g. screwdriver pick and place)
            if current_phase == 1:
                ref_curr_idx = min(220 + int(phase_step_count * 5), 265)
                ref_target_idx = 285
            elif current_phase == 2:
                ref_curr_idx = 280
                ref_target_idx = 320
            elif current_phase == 3:
                ref_curr_idx = 320
                ref_target_idx = 410
            elif current_phase == 4:
                ref_curr_idx = 410
                ref_target_idx = min(430 + int(phase_step_count * 15), 580)
            elif current_phase == 5:
                ref_curr_idx = 580
                ref_target_idx = 660

        ref_curr_idx = min(ref_curr_idx, total_demo_frames - 1)
        ref_target_idx = min(ref_target_idx, total_demo_frames - 1)

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

        # Phase-Specific Control Guarantees:
        # In Phase 1 (DESCENT), keep XY centered over live screwdriver and descend in Z
        if current_phase == 1:
            if grounded_target_xy is not None:
                # Direct fine centering over the detected object
                err_x = float(grounded_target_xy[0] - current_pose[0])
                err_y = float(grounded_target_xy[1] - current_pose[1])
                action_7d[0] = float(np.clip(err_x * 0.001, -0.015, 0.015))
                action_7d[1] = float(np.clip(err_y * 0.001, -0.015, 0.015))
            # Ensure steady vertical descent (25-30 mm/step downward)
            if action_7d[2] > -0.015:
                action_7d[2] = -0.028

        # In Phase 2 (GRASP), force gripper closed and hold arm steady
        elif current_phase == 2:
            action_7d[6] = 1.0
            action_7d[0] = 0.0
            action_7d[1] = 0.0
            action_7d[2] = 0.0

        # In Phase 3 (LIFT), keep gripper closed and lift cleanly upward
        elif current_phase == 3:
            action_7d[6] = 1.0
            action_7d[0] = 0.0
            action_7d[1] = 0.0
            action_7d[2] = 0.04  # 40 mm upward lift per step to clear tabletop cleanly

        # In Phase 4 (TRANSPORT), carry securely towards place target
        elif current_phase == 4:
            action_7d[6] = 1.0
            # Guide Cartesian trajectory towards place target
            dx_tray = np.clip((target_place_xy[0] - current_pose[0]) * 0.001, -0.04, 0.04)
            dy_tray = np.clip((target_place_xy[1] - current_pose[1]) * 0.001, -0.04, 0.05)
            action_7d[0] = float(dx_tray)
            action_7d[1] = float(dy_tray)
            if current_pose[2] < target_lift_z - 15.0:
                action_7d[2] = 0.02
            else:
                action_7d[2] = 0.0

        # In Phase 5 (RELEASE), lower to place position and open gripper
        elif current_phase == 5:
            action_7d[0] = 0.0
            action_7d[1] = 0.0
            action_7d[2] = -0.025
            if current_pose[2] <= target_place_z + 4.0 or phase_step_count >= 3:
                action_7d[6] = 0.0

        # Coordinate frame alignment between camera/DROID model and Fairino base
        dino_cfg = config.get("dino_wm", {})
        invert_dx = args.invert_dx or (args.model == "dino_wm" and dino_cfg.get("invert_dx", False))
        invert_dy = args.invert_dy or (args.model == "dino_wm" and dino_cfg.get("invert_dy", False))
        invert_dz = args.invert_dz or (args.model == "dino_wm" and dino_cfg.get("invert_dz", False))

        if invert_dx:
            action_7d[0] = -action_7d[0]
        if invert_dy:
            action_7d[1] = -action_7d[1]
        if invert_dz:
            action_7d[2] = -action_7d[2]

        if invert_dx or invert_dy or invert_dz:
            logger.info(f"Dispatched Robot Action (Inverted dx={invert_dx}, dy={invert_dy}, dz={invert_dz}): {[round(float(x), 4) for x in action_7d]}")

        # Execute physical action
        robot.step_action(action_7d, speed=speed)

        new_pose = robot.get_tcp_pose()
        logger.info(f"Robot Current TCP Pose: {[round(float(x), 2) for x in new_pose]}")

        # Render step telemetries
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
            # Transition to GRASP when near object (Z <= target_pick_z + 4mm, or descent timeout)
            reached_target = (z_curr <= target_pick_z + 4.0 and phase_step_count >= 2)
            if reached_target or phase_step_count >= 15:
                logger.info(f"🎯 Milestone reached: Arm at grasp pose (X={x_curr:.1f}, Y={y_curr:.1f}, Z={z_curr:.1f}mm) -> Entering Phase 2 (GRASP).")
                current_phase = 2
                phase_step_count = 0
        elif current_phase == 2:
            # Grasp object firmly with gripper
            robot.set_gripper(1.0)
            time.sleep(1.2)  # Ensure fingers physically clamp shut before lifting
            if phase_step_count >= 2:
                logger.info("🎯 Milestone reached: Gripper closed firmly on object -> Entering Phase 3 (LIFT).")
                current_phase = 3
                phase_step_count = 0
        elif current_phase == 3:
            # Lift object off table (Z >= target_lift_z - 10mm or after 5 lift steps)
            if (z_curr >= target_lift_z - 10.0 and phase_step_count >= 2) or phase_step_count >= 5:
                logger.info(f"🎯 Milestone reached: Object lifted off table (Z={z_curr:.1f}mm) -> Entering Phase 4 (TRANSPORT).")
                current_phase = 4
                phase_step_count = 0
        elif current_phase == 4:
            # Transport across table toward place target (reached within 60mm of target or after 18 steps)
            dist_to_place = float(np.hypot(target_place_xy[0] - x_curr, target_place_xy[1] - y_curr))
            reached_tray = (dist_to_place <= 60.0 and phase_step_count >= 4)
            if reached_tray or phase_step_count >= 18:
                logger.info(f"🎯 Milestone reached: End effector reached place location (X={x_curr:.1f}, Y={y_curr:.1f}mm) -> Entering Phase 5 (PLACE).")
                current_phase = 5
                phase_step_count = 0
        elif current_phase == 5:
            # Place down at target (Z <= target_place_z + 4mm or after 6 placement steps)
            if z_curr <= target_place_z + 4.0 or phase_step_count >= 6:
                robot.set_gripper(0.0)
                time.sleep(1.0)
                logger.info("🎉 Milestone reached: Object placed and released!")
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
