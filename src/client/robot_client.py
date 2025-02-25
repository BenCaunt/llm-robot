import zenoh
from zenoh import Config
import numpy as np
import cv2
import json
import time
from dataclasses import dataclass
from typing import Optional, Callable
from constants import CAMERA_FRAME_KEY, ROBOT_TWIST_CMD_KEY
@dataclass
class TwistCommand:
    strafe: float  # x velocity
    forward: float  # y velocity
    turn: float    # angular velocity

class RobotClient:
    def __init__(self, image_callback: Optional[Callable] = None):
        # Initialize Zenoh session
        self.session = zenoh.open(Config())
        
        # Publisher for twist commands
        self.publisher = self.session.declare_publisher(ROBOT_TWIST_CMD_KEY)
        
        # Set up subscribers if callbacks provided
        if image_callback:
            self.image_sub = self.session.declare_subscriber(
                CAMERA_FRAME_KEY,
                lambda sample: self._handle_image(sample, image_callback)
            )
            
    def send_twist(self, twist: TwistCommand):
        """Send normalized twist command to robot"""
        # Ensure values are normalized between -1 and 1
        twist_data = {
            "strafe": max(-1.0, min(1.0, twist.strafe)),
            "forward": max(-1.0, min(1.0, twist.forward)),
            "turn": max(-1.0, min(1.0, twist.turn))
        }
        self.publisher.put(json.dumps(twist_data))

    def _handle_image(self, sample, callback):
        """Internal handler for image data"""
        try:
            np_data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
            received_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if received_img is not None:
                callback(received_img)
        except Exception as e:
            print(f"Failed to process image: {e}")
