import numpy as np
import cv2
from PIL import Image
import rerun as rr
from rerun.datatypes import Angle, RotationAxisAngle
from scipy.spatial.transform import Rotation as R
import os
import glob
import time
from VisualOdometry import VisualOdometry
from stereo_calibration import read_calibration
try:
    # Try absolute import first
    from src.client.camera_access.ZenohCamera import ZenohCamera
except ImportError:
    # Fall back to relative import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_access.ZenohCamera import ZenohCamera

def main():
    # Initialize Rerun
    rr.init("Visual Odometry Demo", spawn=True)

    # Load calibration data
    calibration_data = read_calibration('stereo_calibration.npz')
    K_l = calibration_data['left_camera_matrix']
    dist_l = calibration_data['left_distortion']
    K_r = calibration_data['right_camera_matrix']
    dist_r = calibration_data['right_distortion']
    R = calibration_data['rotation_matrix']
    T = calibration_data['translation_vector']
    
    # Initialize Visual Odometry
    vo = VisualOdometry(K_l, dist_l, K_r, dist_r, R, T)

    # Initialize Zenoh camera
    camera = ZenohCamera()

    while True:
        # Get current frame from Zenoh camera
        frame = camera.get_current_frame()
        vo.process_stereo_pair(frame)
        if frame is None:
            print("No frame received from Zenoh camera")
            time.sleep(1)
            continue

if __name__ == "__main__":
    main() 