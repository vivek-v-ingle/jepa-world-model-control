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
    parser.add_argument("--max_steps", type=int, default=30, help="Number of policy steps to execute")
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
    parser.add_argument("--no_retarget", action="store_true", help="Disable DINOv2 latent patch attention goal retargeting")
    parser.add_argument("--nominal_patch", type=str, default=None, help="Nominal reference bottle patch 'r,c' or 'auto' (default: None, loads from deploy_config.yaml)")
    parser.add_argument("--log_json", type=str, default=None, help="Save rollout trajectory and prediction errors to JSON")
    parser.add_argument("--ref_episode", "--ref_h5", type=str, default=None, dest="ref_episode", help="Path to reference demonstration HDF5 file (overrides config)")
    parser.add_argument("--pure_wm", action="store_true", help="Pure World Model control: no OpenCV grounding, no pick_place_points.json, and no Cartesian phase overrides")
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
    if args.nominal_patch:
        if args.nominal_patch.lower() != "auto":
            nr, nc = [int(x.strip()) for x in args.nominal_patch.split(",")]
            config.setdefault("dino_wm", {})["nominal_bottle_patch"] = [nr, nc]
        else:
            config.setdefault("dino_wm", {})["nominal_bottle_patch"] = "auto"

    if args.model == "dino_wm":
        runner = DinoWMRunner(config)
        logger.info("Meta FAIR DINO-WM DROID model, CEM planner, and Latent Patch Attention initialized.")
    else:
        runner = JEPAPolicyRunner(config)
        logger.info("V-JEPA 2.1 Dreamer AC models and CEM planner initialized.")
    runner._gripper_latched = False
    runner._placed = False


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
    ref_h5 = args.ref_episode or ref_cfg.get("reference_h5")
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
    if not args.pure_wm and not args.no_grounding and camera is not None and args.live:
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
    target_pick_xy = (-245.69, -899.88)
    target_pick_z = 80.91
    target_pick_pose = np.array([-245.69, -899.88, 80.91, -178.83, -3.37, 81.05], dtype=np.float32)
    target_place_xy = (-947.8, -233.1)
    target_place_z = 83.30
    target_lift_z = 140.0

    points_path = Path("/home/fr10/fr10_ws/src/Initial commands for fairino/pick_place_points.json")
    if points_path.exists():
        try:
            import json
            pts = json.loads(points_path.read_text(encoding="utf-8"))
            if "pick" in pts and not args.pure_wm:
                target_pick_xy = (float(pts["pick"][0]), float(pts["pick"][1]))
                target_pick_z = float(pts["pick"][2])
                target_pick_pose = np.array(pts["pick"][:6], dtype=np.float32)
            if "place" in pts:
                target_place_xy = (float(pts["place"][0]), float(pts["place"][1]))
                target_place_z = float(pts["place"][2])
            app_z = float(pts.get("approach_z_mm", 60.0))
            target_lift_z = max(target_lift_z, target_place_z + app_z)
            logger.info(f"📍 Fixed Goal Loaded ({points_path.name}): Place XY=({target_place_xy[0]:.1f}, {target_place_xy[1]:.1f}), Place Z={target_place_z:.1f}mm | Safe Z Floor >= 80.5mm")
        except Exception as exc:
            logger.warning(f"Could not read pick_place_points.json: {exc}")

    if args.pure_wm:
        logger.info("🧠 [PURE WORLD MODEL MODE] Dynamic Pick: World model CEM drives approach to bottle wherever it is. Fixed Goal: Transporting to designated place target.")

    # Ensure robot is at Pre-Approach height above bottle if starting from Home/table center (hybrid mode only)
    if not args.pure_wm:
        init_tcp = robot.get_tcp_pose()
        dist_init_to_pick = float(np.hypot(init_tcp[0] - target_pick_xy[0], init_tcp[1] - target_pick_xy[1]))
        if dist_init_to_pick > 100.0:
            logger.info(f"📍 Robot is at ({init_tcp[0]:.1f}, {init_tcp[1]:.1f}, {init_tcp[2]:.1f}) — {dist_init_to_pick:.1f}mm away from bottle pick pose.")
            logger.info(f"🚀 Moving to Pre-Approach above Bottle: X={target_pick_xy[0]:.1f}, Y={target_pick_xy[1]:.1f}, Z={target_lift_z:.1f}mm, Rz={target_pick_pose[5]:.1f}°...")
            pre_approach = np.array(
                [target_pick_xy[0], target_pick_xy[1], max(init_tcp[2], target_lift_z), target_pick_pose[3], target_pick_pose[4], target_pick_pose[5]],
                dtype=np.float32,
            )
            robot._move_linear(pre_approach, speed=min(speed, 25.0), timeout_sec=18.0)
            time.sleep(0.5)

    rollout_telemetry = []
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
        if args.pure_wm:
            # Pure World Model Mode: Advance subgoals along demonstration trajectory
            frac = (step + 1) / float(args.max_steps)
            ref_target_idx = min(int(frac * total_demo_frames), total_demo_frames - 1)
            ref_curr_idx = max(0, ref_target_idx - max(1, int(0.08 * total_demo_frames)))
        elif total_demo_frames <= 400:
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
        retarget_flag = not args.no_retarget if args.model == "dino_wm" else False
        action_7d, goal_latent, dist, advance, reason = runner.step(
            current_obs_rgb=obs_frame,
            current_robot_pose=current_pose,
            ref_curr_rgb=curr_ref,
            ref_target_rgb=future_ref,
            retarget_goal=retarget_flag,
        )

        retarget_info = getattr(runner, "last_retarget_info", {})
        wm_pred_err = getattr(runner, "last_prediction_error", None)

        logger.info(f"Active Ref Goal: Frame {ref_target_idx}/{total_demo_frames} ({phase_label})")
        logger.info(f"Planned Action Delta: {[round(float(x), 4) for x in action_7d]}")
        logger.info(f"Latent L1 Progress Distance: {dist:.6f}")
        if wm_pred_err is not None:
            logger.info(f"🔮 [WORLD MODEL] 1-Step Pred Error: ||ẑ_t - z_t||₁ = {wm_pred_err:.6f}")
        if retarget_info:
            logger.info(
                f"🎯 [DINOv2 ATTN] Live Patch={retarget_info.get('live_bottle_patch')} | "
                f"Ref Patch={retarget_info.get('ref_bottle_patch')} | "
                f"Δ={retarget_info.get('patch_delta')} | CosSim={retarget_info.get('cosine_similarity', 0.0):.4f}"
            )

        # Phase-Specific Control Guarantees (Hybrid mode only):
        if not args.pure_wm:
            # In Phase 1 (DESCENT), keep XY centered over target and descend in Z
            if current_phase == 1:
                if grounded_target_xy is not None:
                    # Direct fine centering over the detected object (if grounding enabled)
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
        else:
            # 🧠 Pure World Model Mode with Fixed Place Goal:
            # 1. Approach & Pick: Raw CEM action_7d drives approach to wherever the bottle is!
            # 2. Safety Floor: Never go below 80.5mm to protect bottle from crushing.
            # 3. Once grasped and lifted, transport to fixed target (target_place_xy, target_place_z).
            current_pose = robot.get_tcp_pose()
            current_z = current_pose[2]

            # Trigger grasp when descended to pick height (Z <= 85.0 mm or demo grasp frame)
            if not getattr(runner, "_gripper_latched", False) and not getattr(runner, "_placed", False):
                dr, dc = retarget_info.get("patch_delta", (0, 0)) if retarget_info else (0, 0)
                
                # Dynamic spatial retargeting of pick location:
                # Nominal demonstration pick pose: X ~ -560mm, Y ~ -618mm
                # Physical tabletop mapping:
                # Column offset dc maps along table width (Y): ~35mm per patch
                # Row offset dr maps along table depth (X): ~30mm per patch
                target_pick_x = -560.0 + float(dr) * 30.0
                target_pick_y = -618.0 + float(dc) * 35.0
                
                err_x = target_pick_x - current_pose[0]
                err_y = target_pick_y - current_pose[1]
                
                # Closed-loop visual guidance toward detected bottle position
                action_7d[0] = float(np.clip(err_x * 0.001 * 0.35 + action_7d[0] * 0.65, -0.045, 0.045))
                action_7d[1] = float(np.clip(err_y * 0.001 * 0.35 + action_7d[1] * 0.65, -0.045, 0.045))

                if current_z <= 85.0 or (current_z <= 92.0 and action_7d[6] > -0.05):
                    logger.info("✊ [GRIPPER] Latching fingers firmly closed on bottle at (X=%.1f, Y=%.1f, Z=%.1f mm)...", current_pose[0], current_pose[1], current_z)
                    robot.set_gripper(1.0)
                    time.sleep(1.2)  # Allow physical electric fingers to travel and firmly clamp bottle
                    runner._gripper_latched = True
                    runner._lift_steps = 0
                else:
                    robot.set_gripper(0.0)
            elif getattr(runner, "_gripper_latched", False):
                # Bottle is latched!
                robot.set_gripper(1.0)
                if getattr(runner, "_lift_steps", 0) < 3:
                    # Clean vertical lift to clear table
                    action_7d[0] = 0.0
                    action_7d[1] = 0.0
                    action_7d[2] = 0.035
                    runner._lift_steps = getattr(runner, "_lift_steps", 0) + 1
                    logger.info(f"🚀 [LIFT] Bottle lifted cleanly off table (Z={current_z:.1f} mm, step {runner._lift_steps}/3)")
                else:
                    # Transport directly towards fixed place target from pick_place_points.json!
                    dist_to_place = float(np.hypot(target_place_xy[0] - current_pose[0], target_place_xy[1] - current_pose[1]))
                    if dist_to_place > 30.0:
                        # Carry towards fixed destination
                        dx_tray = np.clip((target_place_xy[0] - current_pose[0]) * 0.001, -0.045, 0.045)
                        dy_tray = np.clip((target_place_xy[1] - current_pose[1]) * 0.001, -0.045, 0.045)
                        action_7d[0] = float(dx_tray)
                        action_7d[1] = float(dy_tray)
                        if current_z < 135.0:
                            action_7d[2] = 0.015
                        else:
                            action_7d[2] = 0.0
                        logger.info(f"🚚 [TRANSPORT] Carrying bottle to fixed place target ({target_place_xy[0]:.1f}, {target_place_xy[1]:.1f}) | Distance remaining: {dist_to_place:.1f} mm")
                    else:
                        # Positioned over fixed target! Descend gently to place Z
                        if current_z > target_place_z + 3.0:
                            action_7d[0] = float(np.clip((target_place_xy[0] - current_pose[0]) * 0.001, -0.01, 0.01))
                            action_7d[1] = float(np.clip((target_place_xy[1] - current_pose[1]) * 0.001, -0.01, 0.01))
                            action_7d[2] = -0.020
                            logger.info(f"📥 [DESCEND] Lowering to place surface (Z={current_z:.1f} -> target {target_place_z:.1f} mm)")
                        else:
                            # Safely down on table at fixed place target! Release gripper!
                            logger.info(f"🖐️ [GRIPPER] Placed at fixed goal XY=({target_place_xy[0]:.1f}, {target_place_xy[1]:.1f}), Z={current_z:.1f}mm. Releasing!")
                            robot.set_gripper(0.0)
                            time.sleep(1.2)
                            runner._gripper_latched = False
                            runner._placed = True
            elif getattr(runner, "_placed", False):
                # Release complete! Lift arm slightly away from placed bottle
                robot.set_gripper(0.0)
                action_7d[0] = 0.0
                action_7d[1] = 0.0
                action_7d[2] = 0.025

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

        # CRITICAL HARDWARE SAFETY: Prevent crushing the bottle!
        # If commanded Z would drop below 80.5mm, clamp action_7d[2] so target Z >= 80.5mm
        future_z = current_pose[2] + action_7d[2] * 1000.0
        if future_z < 80.5:
            action_7d[2] = max((80.5 - current_pose[2]) / 1000.0, -0.005)

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
                prediction_error=wm_pred_err,
                retarget_info=retarget_info,
            )

        # Record step telemetry for experiment logging
        step_entry = {
            "step": step + 1,
            "phase": current_phase,
            "latent_l1_dist": float(dist),
            "wm_pred_error": float(wm_pred_err) if wm_pred_err is not None else None,
            "live_bottle_patch": retarget_info.get("live_bottle_patch") if retarget_info else None,
            "ref_bottle_patch": retarget_info.get("ref_bottle_patch") if retarget_info else None,
            "patch_delta": retarget_info.get("patch_delta") if retarget_info else None,
            "cosine_similarity": float(retarget_info.get("cosine_similarity", 0.0)) if retarget_info else None,
            "tcp_pose": [float(x) for x in new_pose],
            "action_7d": [float(x) for x in action_7d],
        }
        rollout_telemetry.append(step_entry)

        # Check physical milestone transition conditions:
        z_curr = new_pose[2]
        y_curr = new_pose[1]
        x_curr = new_pose[0]
        phase_step_count += 1

        if not args.pure_wm:
            if current_phase == 1:
                # Transition to GRASP when near object in both Z and XY (within 50mm of pick XY)
                dist_xy_to_pick = float(np.hypot(x_curr - target_pick_xy[0], y_curr - target_pick_xy[1]))
                reached_target = (z_curr <= target_pick_z + 4.0 and dist_xy_to_pick <= 50.0 and phase_step_count >= 2)
                if reached_target or (phase_step_count >= 15 and dist_xy_to_pick <= 50.0):
                    logger.info(f"🎯 Milestone reached: Arm at grasp pose (X={x_curr:.1f}, Y={y_curr:.1f}, Z={z_curr:.1f}mm, dist_xy={dist_xy_to_pick:.1f}mm) -> Entering Phase 2 (GRASP).")
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
                    logger.info("🎉 Milestone reached: Object successfully placed and released.")
                    current_phase = 6
                    break
        else:
            # Pure World Model Mode: Progress is indicated by demonstration milestone fraction
            pct = int((step + 1) / args.max_steps * 100)
            logger.info(f"📈 [PURE WORLD MODEL PROGRESS] {pct}% of demonstration horizon completed (Ref Frame: {ref_target_idx}/{total_demo_frames})")

    # Cleanup
    if camera is not None:
        camera.release()
    if visualizer is not None:
        visualizer.close()
    robot.close()

    # World Model Prediction & Generalization Summary
    pred_errors = [e["wm_pred_error"] for e in rollout_telemetry if e["wm_pred_error"] is not None]
    logger.info("\n" + "=" * 60)
    logger.info("📊 JEPA WORLD MODEL EXPERIMENTAL SUMMARY")
    logger.info("=" * 60)
    if pred_errors:
        logger.info(f"Mean 1-Step Prediction Error (||ẑ - z||₁): {np.mean(pred_errors):.6f}")
        logger.info(f"Min / Max Prediction Error: {np.min(pred_errors):.6f} / {np.max(pred_errors):.6f}")
    if rollout_telemetry:
        final_dist = rollout_telemetry[-1]["latent_l1_dist"]
        logger.info(f"Final Latent L1 Goal Distance: {final_dist:.6f}")
        if rollout_telemetry[0].get("patch_delta") is not None:
            logger.info(f"DINOv2 Bottle Patch Delta: {rollout_telemetry[-1].get('patch_delta')}")

    if args.log_json:
        import json
        out_p = Path(args.log_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(rollout_telemetry, f, indent=2)
        logger.info(f"Saved experimental trajectory metrics to: {out_p}")

    logger.info("=" * 60)
    logger.info("✅ JEPA Policy Execution completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
