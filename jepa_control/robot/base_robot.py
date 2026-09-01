from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class BaseRobot(ABC):
    """Abstract Base Class for Robot Manipulators."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection with the robot controller."""
        pass

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """Returns current end-effector pose [x, y, z, rx, ry, rz, gripper] in mm/deg."""
        pass

    @abstractmethod
    def step_action(self, action: List[float]) -> bool:
        """
        Executes a 7-DoF relative or absolute delta command.
        action: [dx, dy, dz, drx, dry, drz, gripper]
        """
        pass

    @abstractmethod
    def set_gripper(self, state: float) -> bool:
        """Actuate gripper (0.0 = fully open, 1.0 = fully closed)."""
        pass

    @abstractmethod
    def close(self):
        """Safely disconnect from the robot."""
        pass
