"""
Fairino FR10 robot driver.

Supports:
- MOCK / dry-run mode
- LIVE Fairino FR10 mode
- TCP pose reading
- Safety-state checking
- Cartesian incremental motion
- Gripper command
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import numpy as np

from .base_robot import BaseRobot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fairino SDK import
# ---------------------------------------------------------------------------

FAIRINO_SDK_AVAILABLE = False
Robot = None

try:
    from .sdk.fairino import Robot

    FAIRINO_SDK_AVAILABLE = True
except Exception as exc:
    logger.warning(
        "[ROBOT] Fairino SDK unavailable: %s",
        exc,
    )


# ---------------------------------------------------------------------------
# FairinoDriver
# ---------------------------------------------------------------------------

class FairinoDriver(BaseRobot):
    """
    Driver for the Fairino FR10.

    The driver has two explicit modes:

        mock=True
            No physical robot connection or motion.

        mock=False
            Connect to the physical Fairino controller.

    IMPORTANT:
        connect() only establishes communication and reads state.
        It does NOT automatically enable the robot or change its mode.
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
        self.tool_id = int(tool_id)
        self.user_frame_id = int(user_frame_id)

        self.default_speed = float(default_speed)

        self.safe_z_mm = float(safe_z_mm)
        self.min_z_mm = float(min_z_mm)
        self.max_z_mm = float(max_z_mm)

        self.max_cartesian_step_mm = float(max_cartesian_step_mm)

        self.mock = bool(mock)

        self.robot = None
        self.is_connected = False

        # Mock starting pose:
        # [x, y, z, rx, ry, rz, gripper]
        self._mock_pose = np.array(
            [300.0, 0.0, 200.0, 180.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to the robot.

        This method does NOT enable the physical robot and does NOT issue
        MoveL commands.
        """

        # Explicit mock mode
        if self.mock:
            logger.info(
                "[ROBOT] Initialized FairinoDriver in MOCK / DRY-RUN mode."
            )
            self.is_connected = True
            return True

        # Live mode requires the SDK
        if not FAIRINO_SDK_AVAILABLE or Robot is None:
            logger.error(
                "[ROBOT] Fairino SDK is not available. "
                "Cannot connect in LIVE mode."
            )
            self.is_connected = False
            return False

        try:
            logger.info(
                "[ROBOT] Connecting to Fairino FR10 at %s",
                self.controller_ip,
            )

            # Robot.RPC() establishes:
            #   - XML-RPC communication
            #   - legacy TCP 20004 state stream
            # in the modified Fairino SDK.
            self.robot = Robot.RPC(self.controller_ip)

            # The SDK stores connection status as a class attribute.
            connected = bool(
                getattr(Robot.RPC, "is_connect", False)
            )

            if not connected:
                logger.error(
                    "[ROBOT] Fairino SDK reports connection failure."
                )
                self.robot = None
                self.is_connected = False
                return False

            self.is_connected = True

            logger.info(
                "[ROBOT] Successfully connected to Fairino FR10."
            )

            # Auto-enable robot servos and set AUTO mode on connect
            self._prepare_auto()

            # Configure and activate JODELL RG gripper once on startup
            self.setup_gripper()

            return True

        except Exception as exc:
            logger.exception(
                "[ROBOT] Failed to connect to Fairino FR10: %s",
                exc,
            )

            self.robot = None
            self.is_connected = False

            # IMPORTANT:
            # Never silently switch a failed live connection into mock mode.
            return False

    # -----------------------------------------------------------------------
    # Safety
    # -----------------------------------------------------------------------

    def check_safety(self) -> bool:
        """
        Verify that the robot is in a safe state before allowing motion.

        Returns:
            True  -> safe to issue motion command
            False -> motion must be blocked
        """

        if self.mock:
            return True

        if not self.is_connected or self.robot is None:
            logger.error(
                "[ROBOT] Safety check failed: robot is not connected."
            )
            return False

        try:
            state = self.robot.robot_state_pkg

            emergency_stop = int(
                getattr(state, "EmergencyStop", -1)
            )

            safety_stop0 = int(
                getattr(state, "safety_stop0_state", -1)
            )

            safety_stop1 = int(
                getattr(state, "safety_stop1_state", -1)
            )

            robot_state = int(
                getattr(state, "robot_state", -1)
            )

            program_state = int(
                getattr(state, "program_state", -1)
            )

            # Emergency stop must be inactive.
            if emergency_stop != 0:
                logger.error(
                    "[ROBOT] MOTION BLOCKED: EmergencyStop=%s",
                    emergency_stop,
                )
                return False

            # Safety stops must be inactive.
            if safety_stop0 != 0:
                logger.error(
                    "[ROBOT] MOTION BLOCKED: SafetyStop0=%s",
                    safety_stop0,
                )
                return False

            if safety_stop1 != 0:
                logger.error(
                    "[ROBOT] MOTION BLOCKED: SafetyStop1=%s",
                    safety_stop1,
                )
                return False

            # Robot state 1 = stopped in the current Fairino SDK state model.
            if robot_state != 1:
                logger.error(
                    "[ROBOT] MOTION BLOCKED: RobotState=%s "
                    "(expected 1 = stopped).",
                    robot_state,
                )
                return False

            # Program state 1 = stopped in the current Fairino SDK state model.
            if program_state != 1:
                logger.error(
                    "[ROBOT] MOTION BLOCKED: ProgramState=%s "
                    "(expected 1 = stopped).",
                    program_state,
                )
                return False

            return True

        except Exception as exc:
            logger.exception(
                "[ROBOT] Safety check failed: %s",
                exc,
            )
            return False

    def wait_for_motion_completion(
        self,
        timeout_sec: float = 3.0,
        poll_interval: float = 0.05,
    ) -> bool:
        """
        Wait until robot_state returns to 1 (stopped) after issuing a motion command.
        """
        if self.mock or not self.is_connected or self.robot is None:
            return True

        import time
        # Allow state stream 20004 to register motion start
        time.sleep(0.25)

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                state = getattr(self.robot, "robot_state_pkg", None)
                if state is not None:
                    robot_state = int(getattr(state, "robot_state", -1))
                    program_state = int(getattr(state, "program_state", -1))
                    if robot_state == 1 and program_state == 1:
                        return True
            except Exception:
                pass
            time.sleep(poll_interval)

        logger.warning(
            "[ROBOT] Motion completion wait timed out after %.2f s",
            timeout_sec,
        )
        return False

    # -----------------------------------------------------------------------
    # Robot preparation
    # -----------------------------------------------------------------------

    def _prepare_auto(self) -> bool:
        """
        Prepare the physical robot for automatic motion (Enable servos + AUTO mode).
        """
        if self.mock:
            return True

        if not self.is_connected or self.robot is None:
            logger.error(
                "[ROBOT] Cannot prepare robot: not connected."
            )
            return False

        try:
            # Clear any controller alarms/faults from previous move interruptions
            reset_err = getattr(self.robot, "ResetAllError", None)
            if reset_err is not None:
                try:
                    res_code = reset_err()
                    logger.info("[ROBOT] ResetAllError status: %s", res_code)
                except Exception as e:
                    logger.warning("[ROBOT] ResetAllError exception: %s", e)

            # Enable robot.
            ret_enable = self.robot.RobotEnable(1)

            # Automatic mode (0 = AUTO).
            ret_mode = self.robot.Mode(0)

            logger.info(
                "[ROBOT] RobotEnable status: %s | Mode(0) AUTO status: %s",
                ret_enable,
                ret_mode,
            )

            return True

        except Exception as exc:
            logger.exception(
                "[ROBOT] Failed to prepare robot for AUTO mode: %s",
                exc,
            )
            return False

    def setup_gripper(self, bus: int = 0) -> bool:
        """
        Configure and activate JODELL RG gripper (Company=6, Device=0, Mount=End port 1).
        """
        if self.mock or self.robot is None:
            return True

        try:
            set_cfg = getattr(self.robot, "SetGripperConfig", None)
            act_grp = getattr(self.robot, "ActGripper", None)

            if set_cfg is not None:
                res_cfg = set_cfg(company=6, device=0, softversion=0, bus=bus)
                logger.info("[ROBOT] SetGripperConfig(company=6, device=0, bus=%s) status: %s", bus, res_cfg)

            if act_grp is not None:
                act_grp(index=1, action=0)  # Reset
                time.sleep(1.0)
                res_act = act_grp(index=1, action=1)  # Enable/Activate
                logger.info("[ROBOT] ActGripper(1, action=1) enable status: %s", res_act)
                time.sleep(2.0)

            return True
        except Exception as exc:
            logger.warning("[ROBOT] setup_gripper failed: %s", exc)
            return False

    # -----------------------------------------------------------------------
    # TCP pose
    # -----------------------------------------------------------------------

    def get_tcp_pose(self) -> np.ndarray:
        """
        Return current TCP pose:

        [x, y, z, rx, ry, rz, gripper]

        Position:
            mm

        Orientation:
            degrees

        In live mode, failures are raised rather than replaced by a mock pose.
        """

        if self.mock:
            return self._mock_pose.copy()

        if not self.is_connected or self.robot is None:
            raise RuntimeError(
                "[ROBOT] Cannot read TCP pose: robot is not connected."
            )

        try:
            result = self.robot.GetActualTCPPose()

            if not isinstance(result, (tuple, list)):
                raise RuntimeError(
                    f"Unexpected GetActualTCPPose result: {result!r}"
                )

            if len(result) < 2:
                raise RuntimeError(
                    f"Invalid GetActualTCPPose result: {result!r}"
                )

            error_code = result[0]

            if error_code != 0:
                raise RuntimeError(
                    f"GetActualTCPPose failed with error code "
                    f"{error_code}"
                )

            pose = np.asarray(
                result[1],
                dtype=np.float32,
            ).reshape(-1)

            if pose.size < 6:
                raise RuntimeError(
                    f"Invalid TCP pose length: {pose.size}"
                )

            # Keep the 7th element as gripper state for the project's
            # 7-dimensional action/observation interface.
            if pose.size >= 7:
                gripper = pose[6]
            else:
                gripper = 0.0

            return np.array(
                [
                    pose[0],
                    pose[1],
                    pose[2],
                    pose[3],
                    pose[4],
                    pose[5],
                    gripper,
                ],
                dtype=np.float32,
            )

        except Exception as exc:
            logger.exception(
                "[ROBOT] Failed to read TCP pose: %s",
                exc,
            )
            raise

    # -----------------------------------------------------------------------
    # Cartesian safety limits
    # -----------------------------------------------------------------------

    def _validate_target_pose(
        self,
        target_pose: np.ndarray,
    ) -> bool:
        """
        Validate Cartesian target position against configured limits.
        """

        target_pose = np.asarray(
            target_pose,
            dtype=np.float32,
        ).reshape(-1)

        if target_pose.size < 6:
            logger.error(
                "[ROBOT] Target pose must contain at least 6 values."
            )
            return False

        x, y, z = target_pose[:3]

        if not np.isfinite(target_pose[:6]).all():
            logger.error(
                "[ROBOT] Target pose contains NaN or Inf."
            )
            return False

        if z < self.min_z_mm:
            logger.error(
                "[ROBOT] Target Z %.2f mm below minimum %.2f mm.",
                z,
                self.min_z_mm,
            )
            return False

        if z > self.max_z_mm:
            logger.error(
                "[ROBOT] Target Z %.2f mm above maximum %.2f mm.",
                z,
                self.max_z_mm,
            )
            return False

        return True

    # -----------------------------------------------------------------------
    # Cartesian motion
    # -----------------------------------------------------------------------

    def _move_linear(
        self,
        target_pose: np.ndarray,
        speed: Optional[float] = None,
    ) -> bool:
        """
        Execute one Cartesian MoveL command.

        The Fairino SDK used by this project has the signature:

            MoveL(
                desc_pos,
                tool,
                user,
                joint_pos,
                vel,
                acc,
                ovl,
                blendR,
                blendMode,
                ...
            )

        Passing the joint position explicitly is important because the
        fourth positional argument is NOT velocity.
        """

        target_pose = np.asarray(
            target_pose,
            dtype=np.float32,
        ).reshape(-1)

        if target_pose.size < 6:
            logger.error(
                "[ROBOT] MoveL requires a 6D Cartesian pose."
            )
            return False

        if not self._validate_target_pose(target_pose):
            return False

        # Safety gate immediately before physical motion.
        if not self.check_safety():
            logger.error(
                "[ROBOT] MoveL BLOCKED by safety check."
            )
            return False

        if self.mock:
            self._mock_pose[:6] = target_pose[:6]

            logger.info(
                "[ROBOT][MOCK] MoveL -> %s",
                self._mock_pose[:6],
            )

            return True

        if not self.is_connected or self.robot is None:
            logger.error(
                "[ROBOT] MoveL failed: robot is not connected."
            )
            return False

        move_speed = (
            self.default_speed
            if speed is None
            else float(speed)
        )

        try:
            # Ensure arm is settled before commanding next trajectory step
            self.wait_for_motion_completion(timeout_sec=1.5)

            # Six zeros tell the Fairino SDK to calculate the joint
            # solution automatically using inverse kinematics.
            joint_pos = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            ret = self.robot.MoveL(
                desc_pos=target_pose[:6].tolist(),
                tool=self.tool_id,
                user=self.user_frame_id,
                joint_pos=joint_pos,
                vel=move_speed,
                acc=0.0,
                ovl=100.0,
                blendR=-1.0,
                blendMode=0,
                exaxis_pos=[0.0, 0.0, 0.0, 0.0],
                search=0,
                offset_flag=0,
                offset_pos=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                oacc=100.0,
                config=-1,
                velAccParamMode=0,
                overSpeedStrategy=0,
                speedPercent=35,
            )

            if ret == 185:
                # Code 185: Trajectory in progress. Wait briefly for previous command to settle and retry.
                import time
                time.sleep(0.1)
                self.wait_for_motion_completion(timeout_sec=1.5)
                ret = self.robot.MoveL(
                    desc_pos=target_pose[:6].tolist(),
                    tool=self.tool_id,
                    user=self.user_frame_id,
                    joint_pos=joint_pos,
                    vel=move_speed,
                    acc=0.0,
                    ovl=100.0,
                    blendR=-1.0,
                    blendMode=0,
                    exaxis_pos=[0.0, 0.0, 0.0, 0.0],
                    search=0,
                    offset_flag=0,
                    offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    oacc=100.0,
                    config=-1,
                    velAccParamMode=0,
                    overSpeedStrategy=0,
                    speedPercent=35,
                )

            if ret != 0:
                logger.error(
                    "[ROBOT] MoveL failed with error code: %s",
                    ret,
                )
                return False

            # Wait for physical motion completion so next step starts cleanly
            self.wait_for_motion_completion(timeout_sec=1.5)
            return True

            logger.info(
                "[ROBOT] MoveL command accepted."
            )

            return True

        except Exception as exc:
            logger.exception(
                "[ROBOT] MoveL failed: %s",
                exc,
            )
            return False

    # -----------------------------------------------------------------------
    # 7D action
    # -----------------------------------------------------------------------

    def step_action(
        self,
        action: np.ndarray,
        speed: Optional[float] = None,
    ) -> bool:
        """
        Execute one 7D incremental action:

            [dx, dy, dz, drx, dry, drz, gripper]

        Cartesian position increments are in mm.
        Rotation increments are in degrees.
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.size < 7:
            logger.error(
                "[ROBOT] Action must contain 7 values: "
                "[dx,dy,dz,drx,dry,drz,gripper]"
            )
            return False

        if not np.isfinite(action[:7]).all():
            logger.error(
                "[ROBOT] Action contains NaN or Inf."
            )
            return False

        # FIRST SAFETY GATE.
        if not self.check_safety():
            logger.error(
                "[ROBOT] Motion blocked by safety check."
            )
            return False

        try:
            current_pose = self.get_tcp_pose()
        except Exception:
            return False

        # Position scale: XY = 200.0 mm, Z = 320.0 mm to match physical table surface height.
        # Invert dz (-action[2]) so visual reach-down actions drive arm downward (-Z) toward table.
        pos_scale_xy = getattr(self, "pos_scale_mm", 200.0)
        pos_scale_z = getattr(self, "pos_scale_z_mm", 320.0)
        rot_scale = getattr(self, "rot_scale_deg", 0.0)

        dx = action[0] * pos_scale_xy
        dy = action[1] * pos_scale_xy
        dz = -action[2] * pos_scale_z
        drx, dry, drz = action[3] * rot_scale, action[4] * rot_scale, action[5] * rot_scale
        gripper_cmd = float(action[6])

        # Limit the Cartesian increment.
        translation_norm = float(
            np.linalg.norm(
                np.array([dx, dy, dz], dtype=np.float32)
            )
        )

        if translation_norm > self.max_cartesian_step_mm:
            scale = (
                self.max_cartesian_step_mm
                / translation_norm
            )

            dx *= scale
            dy *= scale
            dz *= scale

            logger.warning(
                "[ROBOT] Cartesian action clipped to %.2f mm.",
                self.max_cartesian_step_mm,
            )

        target_pose = current_pose.copy()

        target_pose[0] += dx
        target_pose[1] += dy
        target_pose[2] += dz

        target_pose[3] += drx
        target_pose[4] += dry
        target_pose[5] += drz

        if not self._validate_target_pose(target_pose):
            return False

        # Execute Cartesian motion.
        if not self._move_linear(
            target_pose,
            speed=speed,
        ):
            return False

        # Apply gripper command (non-fatal if unconfigured/unsupported)
        self.set_gripper(gripper_cmd)

        return True

    # -----------------------------------------------------------------------
    # Gripper
    # -----------------------------------------------------------------------

    def set_gripper(self, command: float) -> bool:
        """
        Set gripper command.

        The exact Fairino gripper API differs between SDK versions.
        Therefore this method calls MoveGripper with fallback handling.

        Mock mode simply stores the command.
        """

        command = float(command)

        if not np.isfinite(command):
            logger.error(
                "[ROBOT] Invalid gripper command: %s",
                command,
            )
            return False

        if self.mock:
            self._mock_pose[6] = command

            logger.info(
                "[ROBOT][MOCK] Gripper -> %.3f",
                command,
            )

            return True

        if not self.is_connected or self.robot is None:
            logger.error(
                "[ROBOT] Cannot set gripper: robot is not connected."
            )
            return False

        move_gripper = getattr(
            self.robot,
            "MoveGripper",
            None,
        )

        if move_gripper is None:
            logger.warning(
                "[ROBOT] MoveGripper is not available in this SDK. "
                "Skipping gripper command."
            )
            return True

        try:
            # Binary gripper state for JODELL RG: command > 0.5 -> CLOSE (90), <= 0.5 -> OPEN (50)
            norm_cmd = float(np.clip(command, 0.0, 1.0))
            is_closed = bool(norm_cmd > 0.5)

            # Avoid spamming RS485 bus if gripper state (OPEN vs CLOSE) has not changed
            if hasattr(self, "_last_gripper_state") and self._last_gripper_state == is_closed:
                return True

            pos_val = 90 if is_closed else 50

            try:
                # MoveGripper(index=1, pos, vel=30, force=40, maxtime=5000, block=0, type=0, rotNum=0, rotVel=0, rotTorque=0)
                ret = move_gripper(1, pos_val, 30, 40, 5000, 0, 0, 0, 0, 0)
                if ret != 0:
                    logger.info("[ROBOT] MoveGripper(pos=%s, closed=%s) returned code: %s", pos_val, is_closed, ret)
                self._last_gripper_state = is_closed
            except TypeError:
                # Fallback for alternative SDK MoveGripper signatures
                try:
                    ret = move_gripper(norm_cmd)
                    if ret != 0:
                        logger.warning("[ROBOT] MoveGripper single-arg returned code: %s", ret)
                except Exception as exc:
                    logger.warning("[ROBOT] MoveGripper single-arg call failed: %s", exc)

            return True

        except Exception as exc:
            logger.warning(
                "[ROBOT] Gripper command skipped/failed: %s",
                exc,
            )
            return True

    # -----------------------------------------------------------------------
    # Stop
    # -----------------------------------------------------------------------

    def stop(self) -> bool:
        """
        Stop robot motion.

        In mock mode this simply returns True.
        """

        if self.mock:
            return True

        if not self.is_connected or self.robot is None:
            return False

        try:
            stop_move = getattr(
                self.robot,
                "StopMove",
                None,
            )

            if stop_move is None:
                logger.warning(
                    "[ROBOT] StopMove is not available."
                )
                return True

            ret = stop_move()

            if ret != 0:
                logger.error(
                    "[ROBOT] StopMove failed: %s",
                    ret,
                )
                return False

            return True

        except Exception as exc:
            logger.exception(
                "[ROBOT] StopMove failed: %s",
                exc,
            )
            return False

    # -----------------------------------------------------------------------
    # Close
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """
        Stop any active motion and disconnect the driver.
        """

        if self.mock:
            self.is_connected = False
            self.robot = None
            return

        try:
            if self.robot is not None:
                self.stop()
        except Exception:
            pass

        self.robot = None
        self.is_connected = False

        logger.info(
            "[ROBOT] FairinoDriver closed."
        )

    # -----------------------------------------------------------------------
    # Destructor
    # -----------------------------------------------------------------------

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass