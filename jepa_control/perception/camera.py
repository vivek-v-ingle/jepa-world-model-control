import cv2
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check for Stereolabs ZED SDK
try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False


class BaseCamera:
    """Abstract interface for video capture sources."""
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError
    def release(self):
        pass


class ZEDCamera(BaseCamera):
    """Native ZED / ZED Mini / ZED 2i Camera interface via PyZED."""

    def __init__(
        self,
        resolution: str = "HD720",
        fps: int = 30,
        camera_id: int = 0,
    ):
        if not ZED_SDK_AVAILABLE:
            raise RuntimeError("pyzed.sl is not installed. Please install the ZED SDK.")

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        
        # Configure resolution
        res_dict = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        self.init_params.camera_resolution = res_dict.get(resolution, sl.RESOLUTION.HD720)
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER

        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.image_zed = sl.Mat()
        logger.info("[CAMERA] ZED Camera successfully initialized.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Captures an RGB frame as np.ndarray [H, W, 3] in RGB format."""
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            # Convert BGRA to RGB
            rgba_np = self.image_zed.get_data()
            rgb_np = cv2.cvtColor(rgba_np, cv2.COLOR_BGRA2RGB)
            return True, rgb_np
        return False, None

    def release(self):
        self.zed.close()
        logger.info("[CAMERA] ZED Camera released.")


class USBCamera(BaseCamera):
    """Standard OpenCV USB / Web camera stream."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            logger.warning(f"[CAMERA] Could not open USB camera index {camera_index}.")
        else:
            logger.info(f"[CAMERA] USB Camera index {camera_index} initialized ({width}x{height} @ {fps}fps).")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, rgb
        return False, None

    def release(self):
        self.cap.release()
        logger.info("[CAMERA] USB Camera released.")


def get_camera_stream(camera_type: str = "auto", camera_index: int = 0) -> BaseCamera:
    """
    Factory function to instantiate the best available camera.
    camera_type: 'zed', 'usb', or 'auto'
    """
    if camera_type.lower() == "zed" or (camera_type == "auto" and ZED_SDK_AVAILABLE):
        try:
            return ZEDCamera(resolution="HD720", fps=30)
        except Exception as e:
            logger.warning(f"[CAMERA] Could not initialize ZED SDK ({e}). Falling back to USB Camera.")
    
    return USBCamera(camera_index=camera_index)
