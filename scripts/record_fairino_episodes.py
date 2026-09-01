#!/usr/bin/env python3
"""
Fairino FR10 Interaction & Demonstration Data Collector.
Records synchronized RGB camera frames + robot Cartesian poses & actions into HDF5 format.
Works seamlessly on both headless servers (mock mode) and physical lab workstations (ZED/Live FR10).
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import yaml

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_control.robot.fairino_driver import FairinoDriver
from jepa_control.perception.camera import get_camera_stream

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DataCollector")

def main():
    parser = argparse.ArgumentParser(description="Record Fairino FR10 episodes to HDF5")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "data" / "fairino_episodes"), help="Directory to save HDF5 files")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--steps_per_episode", type=int, default=50, help="Number of timesteps per episode")
    parser.add_argument("--hz", type=float, default=5.0, help="Recording frequency (Hz)")
    parser.add_argument("--live", action="store_true", help="Connect to live Fairino robot (default is mock)")
    parser.add_argument("--camera", type=str, default="mock", choices=["mock", "zed", "usb", "auto"], help="Camera source")
    parser.add_argument("--ip", type=str, default="192.168.57.2", help="Fairino controller IP")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dt = 1.0 / args.hz

    logger.info("=" * 60)
    logger.info(f"Fairino Data Collector (Mode: {'LIVE' if args.live else 'MOCK'} | Camera: {args.camera.upper()})")
    logger.info(f"Saving to: {args.output_dir}")
    logger.info("=" * 60)

    # 1. Initialize Robot & Camera
    robot = FairinoDriver(controller_ip=args.ip, mock=not args.live)
    robot.connect()

    camera = None
    if args.camera != "mock":
        camera = get_camera_stream(camera_type=args.camera)

    for ep_idx in range(args.num_episodes):
        logger.info(f"\n>>> Recording Episode {ep_idx + 1}/{args.num_episodes} <<<")
        if args.live:
            logger.info("Perform robot manipulation now (or let automated motion execute)...")
            time.sleep(1.0)

        frames = []
        poses = []
        actions = []

        prev_pose = robot.get_tcp_pose()

        for step in range(args.steps_per_episode):
            t_start = time.time()

            # Grab visual observation
            if camera is not None:
                ret, frame = camera.read()
                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Mock colored frame
                frame = np.full((480, 640, 3), (step * 5) % 255, dtype=np.uint8)

            curr_pose = robot.get_tcp_pose()

            # Compute relative Cartesian action [dx, dy, dz, drx, dry, drz, gripper]
            action = [
                (curr_pose[0] - prev_pose[0]) / 1000.0,
                (curr_pose[1] - prev_pose[1]) / 1000.0,
                (curr_pose[2] - prev_pose[2]) / 1000.0,
                curr_pose[3] - prev_pose[3],
                curr_pose[4] - prev_pose[4],
                curr_pose[5] - prev_pose[5],
                curr_pose[6],
            ]

            frames.append(frame)
            poses.append(curr_pose)
            actions.append(action)

            prev_pose = curr_pose

            # Regulate recording frequency
            elapsed = time.time() - t_start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

        # Save to HDF5
        ep_path = os.path.join(args.output_dir, f"episode_{ep_idx}.h5")
        with h5py.File(ep_path, "w") as f:
            obs_group = f.create_group("observations")
            img_group = obs_group.create_group("images")
            img_group.create_dataset("camera_front", data=np.stack(frames), compression="gzip")
            f.create_dataset("robot_states", data=np.stack(poses))
            f.create_dataset("actions", data=np.stack(actions))

        logger.info(f"✅ Episode saved: {ep_path} ({len(frames)} frames)")

    if camera is not None:
        camera.release()
    robot.close()
    logger.info("\nData collection finished successfully!")

if __name__ == "__main__":
    main()
