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
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_access.ZenohCamera import ZenohCamera
    from stereo_odometry.rerun_debugging_utils import (
        matrix_to_axis_angle, log_transform, log_stereo_images, log_undistorted_images,
        log_keypoints_and_matches, log_stereo_points, log_3d_points, log_trajectory,
        log_optical_flow, log_tracked_points, set_frame_time
    )

import BaseOdometry
from stereo_calibration import read_calibration

# Custom function to visualize trajectory with a connecting line
def log_enhanced_trajectory(trajectory_points, path="world/trajectory"):
    """
    Log trajectory as both points and a connected line for better visualization.
    
    Args:
        trajectory_points: List of trajectory points (camera positions)
        path: Rerun path to log the trajectory
    """
    # Log trajectory points
    rr.log(f"{path}/points", rr.Points3D(
        trajectory_points, 
        colors=(255, 0, 0),  # Red
        radii=0.03  # Slightly smaller points
    ))
    
    # Log trajectory line if we have at least 2 points
    if len(trajectory_points) >= 2:
        # Create line segments connecting consecutive points
        rr.log(f"{path}/line", rr.LineStrips3D(
            [trajectory_points],  # Wrap in list to make a single connected line
            colors=(255, 165, 0),  # Orange for the line
            radii=0.01  # Thin line
        ))

class VisualOdometry:
    def __init__(self, K_l, dist_l, K_r, dist_r, R, T, tracking_method="A"):
        """
        tracking_method: choose "A", "B", or "C" for the different approaches.
         - "A": Store only the keypoints that were triangulated.
         - "B": Build an explicit index-to-3D mapping.
         - "C": Use a sanity check to avoid out-of-bound indices.
        """
        self.K_l = K_l        # Left camera intrinsic
        self.dist_l = dist_l  # Left camera distortion
        self.K_r = K_r        # Right camera intrinsic
        self.dist_r = dist_r  # Right camera distortion
        self.R = R            # Rotation from left to right
        self.T = T            # Translation from left to right

        # Projection matrices
        self.P_l = K_l @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P_r = K_r @ np.hstack((R, T))

        # Pose initialization
        self.cur_pose = np.eye(4)

        # For frame-to-frame matching
        self.prev_left = None         # Grayscale left image from previous iteration
        self.prev_kps = None          # All keypoints detected in previous left image
        self.prev_des = None          # Their corresponding descriptors

        # Additional data structures for approaches:
        # Approach A: Only store the keypoints used in triangulation.
        self.prev_kps_3d = None       # List of keypoints that were stereo-matched and triangulated
        # Approach B: Map from the index (in prev_kps) to the corresponding row in prev_pts3d.
        self.prev_index_to_3d = {}    # Dictionary: key = index in prev_kps, value = row index in prev_pts3d

        # In all cases, store the triangulated 3D points.
        self.prev_pts3d = None

        # Feature detector
        self.orb = cv2.ORB_create(2000)
        
        # Select the matching approach ("A", "B", or "C")
        self.tracking_method = tracking_method
        
        # For Rerun visualization
        self.frame_count = 0
        self.trajectory = []

    def _extract_left_right(self, stereo_pil: Image.Image):
        """
        Split a side-by-side stereo PIL image into left and right grayscale images.
        Adjust if your stereo image is laid out differently.
        """
        stereo_np = np.array(stereo_pil)  # shape (H, W, 3) in RGB
        height, width, _ = stereo_np.shape
        mid = width // 2
        
        left_np = stereo_np[:, :mid, :]
        right_np = stereo_np[:, mid:, :]
        
        # Log input stereo images
        log_stereo_images(left_np, right_np, stereo_np)

        left_gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_np, cv2.COLOR_RGB2GRAY)
        return left_gray, right_gray

    def _match_features(self, des1, des2):
        """
        Brute-force matching with crossCheck.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

    def _triangulate(self, kps_left, kps_right):
        """
        Triangulate matched keypoints using projection matrices.
        """
        pts_left = np.array([kp.pt for kp in kps_left], dtype=np.float32)
        pts_right = np.array([kp.pt for kp in kps_right], dtype=np.float32)

        # Log stereo points
        log_stereo_points(pts_left, pts_right)

        pts_left_2xN = np.vstack([pts_left[:, 0], pts_left[:, 1]])
        pts_right_2xN = np.vstack([pts_right[:, 0], pts_right[:, 1]])

        pts_4d = cv2.triangulatePoints(self.P_l, self.P_r, pts_left_2xN, pts_right_2xN)
        pts_4d /= pts_4d[3]
        pts_3d = pts_4d[:3].T  # (N, 3)
        
        # Visualize 3D points
        log_3d_points(pts_3d)
        
        return pts_3d

    def process_stereo_pair(self, stereo_img) -> np.ndarray:
        """
        Process a stereo image pair and update the pose.
        """
        # Set up Rerun timeline for this frame
        self.frame_count = set_frame_time(self.frame_count)
        
        # 1. Extract left/right images and undistort them.
        left_gray, right_gray = self._extract_left_right(stereo_img)
        left_undist = cv2.undistort(left_gray, self.K_l, self.dist_l)
        right_undist = cv2.undistort(right_gray, self.K_r, self.dist_r)
        
        # Log undistorted images
        log_undistorted_images(left_undist, right_undist)

        # 2. Detect features in left and right images.
        kps_left, des_left = self.orb.detectAndCompute(left_undist, None)
        kps_right, des_right = self.orb.detectAndCompute(right_undist, None)
        if des_left is None or des_right is None or len(kps_left) < 10 or len(kps_right) < 10:
            return self.cur_pose

        # 3. Match features between left and right images for stereo.
        lr_matches = self._match_features(des_left, des_right)
        lr_matches = lr_matches[: min(len(lr_matches), 300)]
        
        # Log keypoints and matches
        log_keypoints_and_matches(left_undist, right_undist, kps_left, kps_right, lr_matches)
        
        matched_kps_left = [kps_left[m.queryIdx] for m in lr_matches]
        matched_kps_right = [kps_right[m.trainIdx] for m in lr_matches]
        pts3d_current = self._triangulate(matched_kps_left, matched_kps_right)

        # --- Update our bookkeeping structures for tracking ---
        # Depending on the chosen approach, update different data structures.
        if self.tracking_method == "A":
            # Approach A: Only store keypoints that were triangulated.
            self.prev_kps_3d = matched_kps_left  # these directly correspond to pts3d_current rows
        elif self.tracking_method == "B":
            # Approach B: Build a mapping from the index in kps_left to the 3D point index.
            # Here, we use the indices from the stereo matching.
            self.prev_index_to_3d = {m.queryIdx: i for i, m in enumerate(lr_matches)}
        # For Approach C, we leave self.prev_kps unchanged.
        # In all cases, store the full keypoints and descriptors as backup.
        self.prev_kps = kps_left
        self.prev_des = des_left
        self.prev_pts3d = pts3d_current

        # 4. If we have a previous frame, estimate motion via PnP.
        if self.prev_left is not None and self.prev_pts3d is not None:
            # Match descriptors between previous left image and current left image.
            matches_frame = self._match_features(self.prev_des, des_left)
            matches_frame = matches_frame[: min(len(matches_frame), 300)]
            
            # Collect correspondences.
            valid_3d = []
            valid_2d = []
            
            if self.tracking_method == "A":
                # Use stored keypoints that were triangulated.
                for m in matches_frame:
                    # m.queryIdx corresponds to the index in self.prev_des, and for approach A
                    # we assume that self.prev_kps_3d[i] corresponds to self.prev_pts3d[i].
                    # Use only if the matched keypoint was triangulated.
                    if m.queryIdx < len(self.prev_kps_3d):
                        valid_3d.append(self.prev_pts3d[m.queryIdx])
                        valid_2d.append(kps_left[m.trainIdx].pt)
            elif self.tracking_method == "B":
                # Use the mapping dictionary.
                for m in matches_frame:
                    if m.queryIdx in self.prev_index_to_3d:
                        row_idx = self.prev_index_to_3d[m.queryIdx]
                        valid_3d.append(self.prev_pts3d[row_idx])
                        valid_2d.append(kps_left[m.trainIdx].pt)
            elif self.tracking_method == "C":
                # Use all previous keypoints but check index boundaries.
                # WARNING: This approach may skip some correspondences.
                for m in matches_frame:
                    # Use sanity check before indexing
                    if m.queryIdx < len(self.prev_pts3d):
                        valid_3d.append(self.prev_pts3d[m.queryIdx])
                        valid_2d.append(kps_left[m.trainIdx].pt)
            
            valid_3d = np.array(valid_3d, dtype=np.float32)
            valid_2d = np.array(valid_2d, dtype=np.float32)
            
            # Visualize tracked points
            if len(valid_2d) > 0:
                log_tracked_points(valid_2d)
            
            if len(valid_3d) >= 6:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    valid_3d, valid_2d, self.K_l, self.dist_l,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=5.0,
                    confidence=0.99
                )
                if success:
                    R_est, _ = cv2.Rodrigues(rvec)
                    T_est = tvec.reshape(3)
                    transformation = np.eye(4)
                    transformation[:3, :3] = R_est
                    transformation[:3, 3] = T_est
                    self.cur_pose = self.cur_pose @ np.linalg.inv(transformation)
                    
                    # Log the updated camera pose
                    log_transform("world/camera", self.cur_pose)
                    
                    # Update and visualize the trajectory
                    position = self.cur_pose[:3, 3]
                    self.trajectory.append(position)
                    # Use enhanced trajectory visualization
                    log_enhanced_trajectory(self.trajectory)
        else:
            # Log the initial pose
            log_transform("world/camera", self.cur_pose)
            
            # Start tracking trajectory
            self.trajectory = [self.cur_pose[:3, 3]]
            # Use enhanced trajectory visualization
            log_enhanced_trajectory(self.trajectory)
        
        # 5. Update the previous left image.
        self.prev_left = left_undist

        return self.cur_pose


# Usage example
if __name__ == "__main__":
    # Initialize Rerun for visualization
    rr.init("O1Pro Visual Odometry Visualization", spawn=True)
    
    # Set up a world coordinate system for reference
    # Add coordinate axes to better understand orientation
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    # Log coordinate system as line strips from origin
    rr.log("world/axes/x", rr.LineStrips3D([[origin, x_axis]], colors=(255, 0, 0), radii=0.02))
    rr.log("world/axes/y", rr.LineStrips3D([[origin, y_axis]], colors=(0, 255, 0), radii=0.02))
    rr.log("world/axes/z", rr.LineStrips3D([[origin, z_axis]], colors=(0, 0, 255), radii=0.02))
    
    calibration_data = read_calibration('stereo_calibration.npz')
    K_l = calibration_data['left_camera_matrix']
    dist_l = calibration_data['left_distortion']
    K_r = calibration_data['right_camera_matrix']
    dist_r = calibration_data['right_distortion']
    R = calibration_data['rotation_matrix']
    T = calibration_data['translation_vector']

    # Change the tracking_method to "A", "B", or "C" to test each approach.
    vo = VisualOdometry(K_l, dist_l, K_r, dist_r, R, T, tracking_method="A")

    camera = ZenohCamera()
    while True:
        frame = camera.get_current_frame()  # expected to be a PIL image with left/right side-by-side
        if frame is None:
            break
        pose = vo.process_stereo_pair(frame)
        print(f"Current Pose:\n{pose}")
        
        # Small sleep to allow visualization to update
        time.sleep(0.01)
    camera.close()
