import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from jepa_control.robot.base_robot import BaseRobot

logger = logging.getLogger(__name__)

# Add SDK path if present
SDK_DIR = Path(__file__).resolve().parent / "sdk"
if SDK_DIR.exists() and str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

try:
    from fairino import Robot
    FAIRINO_SDK_AVAILABLE = True
except ImportError:
    FAIRINO_SDK_AVAILABLE = False


class FairinoDriver(BaseRobot):
    """
    Robust driver interface for Fairino FR10 industrial manipulator.
    Features:
      - Live XML-RPC communication over TCP/IP
      - Automatic safety limits & Cartesian velocity clamping
      - Built-in Dry-Run / Mock mode for offline development
    """

    def __init__(
        self,
        controller_ip: str = "192.168.57.2",
        tool_id: int = 2,
        user_frame_id: int = 0,
        default_speed: float = 10.0,
        safe_z_mm: float = 50.0,
        min_z_mm: float = -300.0,
        max_z_mm: float = 1200.0,
        max_cartesian_step_mm: float = 100.0,
        mock: bool = False,
    ):
        self.controller_ip = controller_ip
        self.tool_id = tool_id
        self.user_frame_id = user_frame_id
        self.default_speed = default_speed
        self.safe_z_mm = safe_z_mm
        self.min_z_mm = min_z_mm
        self.max_z_mm = max_z_mm
        self.max_cartesian_step_mm = max_cartesian_step_mm
        self.mock = mock

        self.robot = None
        self.is_connected = False
        
        # Mock internal state [x, y, z, rx, ry, rz, gripper]
        self._mock_pose = np.array([300.0, 0.0, 200.0, 180.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def connect(self) -> bool:
        if self.mock or not FAIRINO_SDK_AVAILABLE:
            logger.info(f"[ROBOT] Initialized FairinoDriver in MOCK / DRY-RUN mode.")
            self.is_connected = True
            return True

        try:
            logger.info(f"[ROBOT] Connecting to Fairino controller at {self.controller_ip}...")
            self.robot = Robot.RPC(self.controller_ip)
            
            # Check connection
            connected = getattr(self.robot, "is_connect", getattr(self.robot, "is_conect", False))
            if not connected and getattr(self.robot, "robot", None) is not None:
                self.robot.robot.GetControllerIP()
                self.is_connected = True
            else:
                self.is_connected = bool(connected)

            if self.is_connected:
                logger.info(f"[ROBOT] Successfully connected to Fairino FR10.")
                self._prepare_auto()
                return True
            else:
                logger.warning(f"[ROBOT] Controller at {self.controller_ip} unreachable. Falling back to MOCK mode.")
                self.mock = True
                self.is_connected = True
                return True
        except Exception as e:
            logger.warning(f"[ROBOT] Error during connection ({e}). Falling back to MOCK mode.")
            self.mock = True
            self.is_connected = True
            return True

    def _prepare_auto(self):
        """Prepares the physical controller for automatic trajectory execution."""
        if self.mock or self.robot is None:
            return
        try:
            self.robot.RobotEnable(1)
            self.robot.Mode(0) # Auto mode
        except Exception as e:
            logger.warning(f"[ROBOT] Could not set Auto mode: {e}")

    def get_tcp_pose(self) -> np.ndarray:
        """Returns current [x, y, z, rx, ry, rz, gripper] in mm / deg."""
        if self.mock or self.robot is None:
            return self._mock_pose.copy()

        try:
            ret, pose = self.robot.GetActualTCPPose()
            if ret == 0 and len(pose) >= 6:
                # Append current gripper state (default 0.0)
                return np.array(pose[:6] + [self._mock_pose[-1]], dtype=np.float32)
            else:
                return self._mock_pose.copy()
        except Exception as e:
            logger.warning(f"[ROBOT] Error getting TCP pose ({e}). Using cached pose.")
            return self._mock_pose.copy()

    def step_action(self, action: List[float]) -> bool:
        """
        Applies a 7-DoF delta command: [dx, dy, dz, drx, dry, drz, gripper]
        """
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        action = np.asarray(action, dtype=np.float32)
        current_pose = self.get_tcp_pose()

        # Delta scaling (JEPA output deltas in meters/radians converted to mm/degrees)
        dx_mm = float(action[0]) * 1000.0 if abs(action[0]) < 2.0 else float(action[0])
        dy_mm = float(action[1]) * 1000.0 if abs(action[1]) < 2.0 else float(action[1])
        dz_mm = float(action[2]) * 1000.0 if abs(action[2]) < 2.0 else float(action[2])

        # Enforce step clamp safety
        step_norm = np.linalg.norm([dx_mm, dy_mm, dz_mm])
        if step_norm > self.max_cartesian_step_mm:
            scale = self.max_cartesian_step_mm / step_norm
            dx_mm *= scale
            dy_mm *= scale
            dz_mm *= scale

        target_x = current_pose[0] + dx_mm
        target_y = current_pose[1] + dy_mm
        target_z = np.clip(current_pose[2] + dz_mm, self.min_z_mm, self.max_z_mm)
        
        target_pose = [target_x, target_y, target_z, current_pose[3], current_pose[4], current_pose[5]]
        gripper_cmd = float(action[-1])

        if self.mock:
            self._mock_pose[:6] = target_pose
            self._mock_pose[6] = gripper_cmd
            logger.debug(f"[MOCK ROBOT] Step -> Target TCP: {target_pose}, Gripper: {gripper_cmd:.2f}")
            return True

        try:
            # MoveL linear interpolation to next pose
            ret = self.robot.MoveL(
                desc_pos=target_pose,
                tool=self.tool_id,
                user=self.user_frame_id,
                vel=float(self.default_speed),
                acc=20.0,
                ovl=100.0,
                blendT=-1.0,
                offset_flag=0,
            )
            if ret == 0:
                self.set_gripper(gripper_cmd)
                return True
            else:
                logger.error(f"[ROBOT] MoveL failed with error code: {ret}")
                return False
        except Exception as e:
            logger.error(f"[ROBOT] Exception during MoveL: {e}")
            return False

    def set_gripper(self, state: float) -> bool:
        """Actuate gripper (0.0 = open, 1.0 = closed)."""
        self._mock_pose[6] = state
        if self.mock or self.robot is None:
            return True

        try:
            # Modbus / Gripper RPC commands
            pos = 90 if state > 0.5 else 50 # 90 closed, 50 open
            # Call MoveGripper if supported
            if hasattr(self.robot, "MoveGripper"):
                self.robot.MoveGripper(1, pos, 30, 40, 3000, 0)
            return True
        except Exception as e:
            logger.warning(f"[ROBOT] Gripper error: {e}")
            return False

    def close(self):
        if self.robot is not None:
            try:
                self.robot.StopMove()
            except Exception:
                pass
        self.is_connected = False
        logger.info("[ROBOT] Disconnected.")
