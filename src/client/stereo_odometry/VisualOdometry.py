import numpy as np
import cv2
from PIL import Image
import rerun as rr
import time
from rerun.datatypes import Angle, RotationAxisAngle
from scipy.spatial.transform import Rotation as R
try:
    # Try absolute import first
    from src.client.camera_access.ZenohCamera import ZenohCamera
    from src.client.stereo_odometry.rerun_debugging_utils import (
        matrix_to_axis_angle, log_transform, log_stereo_images, log_undistorted_images,
        log_keypoints_and_matches, log_stereo_points, log_3d_points, log_trajectory,
        log_optical_flow, log_tracked_points, set_frame_time
    )
except ImportError:
    # Fall back to relative import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_access.ZenohCamera import ZenohCamera
    from stereo_odometry.rerun_debugging_utils import (
        matrix_to_axis_angle, log_transform, log_stereo_images, log_undistorted_images,
        log_keypoints_and_matches, log_stereo_points, log_3d_points, log_trajectory,
        log_optical_flow, log_tracked_points, set_frame_time
    )

from stereo_calibration import read_calibration

class VisualOdometry:
    def __init__(self, K_l, dist_l, K_r, dist_r, R, T):
        super().__init__()
        self.K_l = K_l
        self.dist_l = dist_l
        self.K_r = K_r
        self.dist_r = dist_r
        self.R = R
        self.T = T

        # Projection matrices for triangulation
        self.P_l = K_l @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P_r = K_r @ np.hstack((R, T))

        # Global pose (world-to-camera transformation) initialized as identity.
        self.cur_pose = np.eye(4)
        self.prev_left = None         # previous left image (grayscale)
        self.prev_pts3d = None        # corresponding 3D points from previous frame
        self.prev_kps = None          # corresponding 2D keypoints (in left image)

        # Feature detector
        self.orb = cv2.ORB_create(2000)
        
        # Frame counter for Rerun timeline
        self.frame_count = 0
        
        # Store trajectory points for visualization
        self.trajectory = []

    def triangulate_features(self, left_img, right_img):
        """
        Detect features in left/right images, match them, and triangulate to obtain 3D points.
        Returns: (pts3d, kps_left) where kps_left are 2D keypoint positions in the left image.
        """
        # Detect ORB keypoints and compute descriptors.
        kps_left, des_left = self.orb.detectAndCompute(left_img, None)
        kps_right, des_right = self.orb.detectAndCompute(right_img, None)
        if des_left is None or des_right is None:
            return None, None

        # Match descriptors using BFMatcher with Hamming distance.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des_left, des_right)
        if len(matches) < 10:
            return None, None

        # Log keypoints and matches
        log_keypoints_and_matches(left_img, right_img, kps_left, kps_right, matches)
        
        pts_left = np.float32([kps_left[m.queryIdx].pt for m in matches])
        pts_right = np.float32([kps_right[m.trainIdx].pt for m in matches])
        
        # Log stereo points
        log_stereo_points(pts_left, pts_right)
        
        # Reshape for triangulation (2 x N)
        pts_left_T = pts_left.T
        pts_right_T = pts_right.T

        # Triangulate points using the projection matrices.
        pts4d = cv2.triangulatePoints(self.P_l, self.P_r, pts_left_T, pts_right_T)
        pts3d = pts4d[:3, :] / pts4d[3, :]
        pts3d = pts3d.T  # shape: (N, 3)
        
        # Visualize 3D points
        log_3d_points(pts3d)
        
        return pts3d, pts_left

    def process_stereo_pair(self, stereo_img) -> np.ndarray:
        """
        Process a stereo image pair and return the current pose.
        Input:
          stereo_img: PIL Image assumed to be a side-by-side stereo image.
        Returns:
          SE(3) pose matrix (4x4 numpy array) representing the global camera pose.
        """
        # Set up Rerun timeline for this frame
        self.frame_count = set_frame_time(self.frame_count)
        
        # Convert PIL image to NumPy array.
        img_np = np.array(stereo_img)
        # Assume image is in RGB format.
        height, width, _ = img_np.shape
        mid = width // 2
        left_img_color = img_np[:, :mid, :]
        right_img_color = img_np[:, mid:, :]
        
        # Log input stereo images
        log_stereo_images(left_img_color, right_img_color, stereo_img)

        # Convert to grayscale.
        left_gray = cv2.cvtColor(left_img_color, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img_color, cv2.COLOR_RGB2GRAY)

        # Undistort images.
        left_undist = cv2.undistort(left_gray, self.K_l, self.dist_l)
        right_undist = cv2.undistort(right_gray, self.K_r, self.dist_r)
        
        # Log undistorted images
        log_undistorted_images(left_undist, right_undist)

        # First frame: initialize features via stereo matching.
        if self.prev_left is None:
            pts3d, kps_left = self.triangulate_features(left_undist, right_undist)
            if pts3d is None or kps_left is None:
                # If no features found, return current pose.
                return self.cur_pose
            self.prev_left = left_undist
            self.prev_pts3d = pts3d
            self.prev_kps = np.float32(kps_left)
            
            # Log the initial pose
            log_transform("world/camera", self.cur_pose)
            
            # Start tracking trajectory
            self.trajectory = [self.cur_pose[:3, 3]]
            log_trajectory(self.trajectory)
            
            return self.cur_pose

        # Subsequent frames: track features using optical flow.
        prev_pts_2d = self.prev_kps.reshape(-1, 1, 2)
        curr_pts_2d, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_left, left_undist, prev_pts_2d, None)
        status = status.reshape(-1)
        
        # Visualize optical flow tracking
        log_optical_flow(self.prev_left, left_undist, prev_pts_2d, curr_pts_2d, status)
        
        # Select only good points.
        good_old_pts3d = self.prev_pts3d[status == 1]
        good_new_pts2d = curr_pts_2d[status == 1].reshape(-1, 2)
        
        # Log tracked points
        log_tracked_points(good_new_pts2d)

        # If not enough points, reinitialize from stereo.
        if len(good_old_pts3d) < 6 or len(good_new_pts2d) < 6:
            pts3d, kps_left = self.triangulate_features(left_undist, right_undist)
            if pts3d is None or kps_left is None:
                return self.cur_pose
            self.prev_left = left_undist
            self.prev_pts3d = pts3d
            self.prev_kps = np.float32(kps_left)
            
            # Log the current pose
            log_transform("world/camera", self.cur_pose)
            
            return self.cur_pose

        # Estimate camera motion using solvePnPRansac.
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            good_old_pts3d, good_new_pts2d, self.K_l, self.dist_l,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not retval or rvec is None or tvec is None:
            return self.cur_pose

        # Convert rotation vector to matrix.
        R_cur, _ = cv2.Rodrigues(rvec)
        # Form the transformation matrix from previous to current frame.
        T_cur = np.eye(4)
        T_cur[:3, :3] = R_cur
        T_cur[:3, 3] = tvec.flatten()

        # Note on conventions:
        # solvePnP here gives the pose of the 3D points in the camera coordinate system.
        # To update the camera's global pose, we invert the relative motion.
        self.cur_pose = self.cur_pose @ np.linalg.inv(T_cur)
        
        # Log the updated camera pose
        log_transform("world/camera", self.cur_pose)
        
        # Update and visualize the trajectory
        position = self.cur_pose[:3, 3]
        self.trajectory.append(position)
        log_trajectory(self.trajectory)

        # Reinitialize feature set for the next frame using stereo matching.
        pts3d, kps_left = self.triangulate_features(left_undist, right_undist)
        if pts3d is not None and kps_left is not None:
            self.prev_left = left_undist
            self.prev_pts3d = pts3d
            self.prev_kps = np.float32(kps_left)
        else:
            # If reinitialization fails, update only the previous image.
            self.prev_left = left_undist

        return self.cur_pose

# Usage example
if __name__ == "__main__":
    # Initialize Rerun for visualization
    rr.init("Visual Odometry Visualization", spawn=True)
    
    calibration_data = read_calibration('stereo_calibration.npz')
    K_l = calibration_data['left_camera_matrix']
    dist_l = calibration_data['left_distortion']
    K_r = calibration_data['right_camera_matrix']
    dist_r = calibration_data['right_distortion']
    R = calibration_data['rotation_matrix']
    T = calibration_data['translation_vector']

    vo = VisualOdometry(K_l, dist_l, K_r, dist_r, R, T)
    camera = ZenohCamera()

    while True:
        frame = camera.get_current_frame()  # PIL Image
        if frame is None:
            break

        pose = vo.process_stereo_pair(frame)
        print(f"Current Pose:\n{pose}")
        
        # Small sleep to allow visualization to update
        time.sleep(0.01)

    camera.close()
