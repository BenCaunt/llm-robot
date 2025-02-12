"""
Basic component models for robot hardware
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

@dataclass
class Component:
    name: str
    description: str

@dataclass
class Drivebase:
    name: str
    description: str

class JointType(Enum):
    MOTOR = "motor"
    SERVO = "servo"

@dataclass
class Joint:
    """Represents a robot joint (motor or servo)"""
    id: int
    name: str
    joint_type: JointType
    sensor_resolution: Optional[int] = None
    min_limit: Optional[float] = None
    max_limit: Optional[float] = None

    def set_limits(self, min_val: float, max_val: float) -> None:
        """Set the joint limits"""
        self.min_limit = min_val
        self.max_limit = max_val

@dataclass
class DriveBase:
    """Represents a wheeled drive base"""
    front_left_wheel: Joint
    front_right_wheel: Joint
    rear_left_wheel: Joint
    rear_right_wheel: Joint

    @property
    def joints(self) -> list[Joint]:
        """Get all joints in the drive base"""
        return [
            self.front_left_wheel,
            self.front_right_wheel,
            self.rear_left_wheel,
            self.rear_right_wheel
        ]



    
