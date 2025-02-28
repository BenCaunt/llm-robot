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

        # Compute stereo rectification
        self.img_size = None  # Will be set on first frame
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None
        
        # Projection matrices for triangulation (will be set after rectification)
        self.P_l = None
        self.P_r = None

        # Global pose (world-to-camera transformation) initialized as identity.
        self.cur_pose = np.eye(4)
        self.prev_left = None         # previous left image (grayscale)
        self.prev_pts3d = None        # corresponding 3D points from previous frame
        self.prev_kps = None          # corresponding 2D keypoints (in left image)

        # Feature detector with more features and better distribution
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        
        # Frame counter for Rerun timeline
        self.frame_count = 0
        
        # Store trajectory points for visualization
        self.trajectory = []

    def _initialize_rectification(self, img_size):
        """
        Initialize stereo rectification maps for the given image size.
        Args:
            img_size: Tuple of (width, height) of the image
        """
        self.img_size = img_size
        
        # Compute stereo rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.K_l, self.dist_l, self.K_r, self.dist_r, 
            img_size, self.R, self.T, 
            flags=cv2.CALIB_ZERO_DISPARITY, 
            alpha=0  # 0 for full rectification, 1 for no black borders
        )
        
        # Update the projection matrices for triangulation with rectified params
        self.P_l = self.P1
        self.P_r = self.P2
        
        # Compute undistortion and rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_l, self.dist_l, self.R1, self.P1, img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K_r, self.dist_r, self.R2, self.P2, img_size, cv2.CV_32FC1
        )
        
        print("Stereo rectification initialized with image size:", img_size)

    def triangulate_features(self, left_img, right_img):
        """
        Detect features in left/right images, match them, and triangulate to obtain 3D points.
        Now with RANSAC-based filtering for more robust matching.
        Returns: (pts3d, kps_left) where kps_left are 2D keypoint positions in the left image.
        """
        # Detect ORB keypoints and compute descriptors.
        kps_left, des_left = self.orb.detectAndCompute(left_img, None)
        kps_right, des_right = self.orb.detectAndCompute(right_img, None)
        if des_left is None or des_right is None or len(kps_left) < 10 or len(kps_right) < 10:
            print("Not enough keypoints detected")
            return None, None

        # Match descriptors using BFMatcher with Hamming distance.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des_left, des_right)
        
        # Sort matches by distance for better quality matches first
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            print("Not enough matches found")
            return None, None

        # Extract keypoint coordinates for both images
        pts_left = np.float32([kps_left[m.queryIdx].pt for m in matches])
        pts_right = np.float32([kps_right[m.trainIdx].pt for m in matches])
        
        # Filter matches using RANSAC with the fundamental matrix constraint
        # This enforces the epipolar constraint between the two images
        F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is None or mask is None or np.sum(mask) < 8:
            print("Could not estimate fundamental matrix or too few inliers")
            return None, None
            
        # Keep only inlier matches
        mask = mask.ravel().astype(bool)
        pts_left = pts_left[mask]
        pts_right = pts_right[mask]
        
        # Create filtered matches list for visualization
        filtered_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        
        # Log keypoints and matches (now showing only inliers)
        log_keypoints_and_matches(left_img, right_img, kps_left, kps_right, filtered_matches)
        
        # Log stereo points
        log_stereo_points(pts_left, pts_right)
        
        # Additional filter: Ensure matches follow the epipolar constraint
        # For rectified images, corresponding points should have very similar y-coordinates
        y_diffs = np.abs(pts_left[:, 1] - pts_right[:, 1])
        epipolar_mask = y_diffs < 2.0  # 2 pixel tolerance for rectification errors
        
        pts_left = pts_left[epipolar_mask]
        pts_right = pts_right[epipolar_mask]
        
        if len(pts_left) < 8:
            print("Too few points after epipolar filtering")
            return None, None
        
        # Reshape for triangulation (2 x N)
        pts_left_T = pts_left.T
        pts_right_T = pts_right.T

        # Triangulate points using the rectified projection matrices
        pts4d = cv2.triangulatePoints(self.P_l, self.P_r, pts_left_T, pts_right_T)
        pts3d = pts4d[:3, :] / pts4d[3, :]
        pts3d = pts3d.T  # shape: (N, 3)
        
        # Filter out points with invalid depths (negative or too far)
        valid_depths = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 50)
        pts3d = pts3d[valid_depths]
        pts_left = pts_left[valid_depths]
        
        if len(pts3d) < 8:
            print("Too few valid 3D points after depth filtering")
            return None, None
        
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

        # Initialize rectification maps if not done already
        if self.img_size is None:
            self._initialize_rectification((left_gray.shape[1], left_gray.shape[0]))

        # Apply rectification
        left_rect = cv2.remap(left_gray, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self.map2x, self.map2y, cv2.INTER_LINEAR)
        
        # Log undistorted and rectified images
        log_undistorted_images(left_rect, right_rect)

        # First frame: initialize features via stereo matching.
        if self.prev_left is None:
            pts3d, kps_left = self.triangulate_features(left_rect, right_rect)
            if pts3d is None or kps_left is None or len(pts3d) < 6:
                # If no features found, return current pose.
                print("Failed to initialize on first frame, not enough features")
                return self.cur_pose
            self.prev_left = left_rect
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
        curr_pts_2d, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_left, left_rect, prev_pts_2d, None,
            winSize=(21, 21),  # Larger window for better tracking
            maxLevel=3,        # More pyramid levels for handling larger motions
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        status = status.reshape(-1)
        
        # Visualize optical flow tracking
        log_optical_flow(self.prev_left, left_rect, prev_pts_2d, curr_pts_2d, status)
        
        # Select only good points.
        good_old_pts3d = self.prev_pts3d[status == 1]
        good_new_pts2d = curr_pts_2d[status == 1].reshape(-1, 2)
        
        # Log tracked points
        log_tracked_points(good_new_pts2d)

        # If not enough points, reinitialize from stereo.
        if len(good_old_pts3d) < 10 or len(good_new_pts2d) < 10:
            print(f"Too few tracked points ({len(good_old_pts3d)}), reinitializing...")
            pts3d, kps_left = self.triangulate_features(left_rect, right_rect)
            if pts3d is None or kps_left is None or len(pts3d) < 6:
                # If reinit fails, just return the current pose
                return self.cur_pose
                
            self.prev_left = left_rect
            self.prev_pts3d = pts3d
            self.prev_kps = np.float32(kps_left)
            
            # Log the current pose
            log_transform("world/camera", self.cur_pose)
            
            return self.cur_pose

        # Refine matched points with RANSAC to remove outliers
        if len(good_old_pts3d) > 15:  # Only attempt RANSAC if enough points
            ransac_retval, _, _, inliers = cv2.solvePnPRansac(
                good_old_pts3d, good_new_pts2d, self.K_l, None,  # No distortion as images are already rectified
                iterationsCount=100,
                reprojectionError=2.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if ransac_retval and inliers is not None and len(inliers) > 8:
                inlier_indices = inliers.ravel()
                good_old_pts3d = good_old_pts3d[inlier_indices]
                good_new_pts2d = good_new_pts2d[inlier_indices]
                print(f"RANSAC filtered to {len(inlier_indices)} inliers")
            else:
                print("RANSAC filtering failed, using all points")

        # Estimate camera motion using solvePnP first with EPNP for initial guess
        retval, rvec, tvec = cv2.solvePnP(
            good_old_pts3d, good_new_pts2d, self.K_l, None,  # No distortion as images are already rectified
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not retval or rvec is None or tvec is None:
            print("Initial PnP estimation failed")
            return self.cur_pose

        # Refine the pose estimate using LM optimization for more accuracy
        refined_retval, refined_rvec, refined_tvec = cv2.solvePnP(
            good_old_pts3d, good_new_pts2d, self.K_l, None, 
            rvec=rvec, tvec=tvec,  # Use initial estimate
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if refined_retval and refined_rvec is not None and refined_tvec is not None:
            rvec = refined_rvec
            tvec = refined_tvec
            print("Pose refined successfully")
        else:
            print("Pose refinement failed, using initial estimate")

        # Convert rotation vector to matrix
        R_cur, _ = cv2.Rodrigues(rvec)
        
        # Form the transformation matrix from previous to current frame
        T_cur = np.eye(4)
        T_cur[:3, :3] = R_cur
        T_cur[:3, 3] = tvec.flatten()

        # Print the magnitude of rotation and translation for debugging
        rot_angle_deg = np.linalg.norm(rvec) * 180 / np.pi
        trans_mag = np.linalg.norm(tvec)
        print(f"Rotation: {rot_angle_deg:.2f}°, Translation: {trans_mag:.4f} units")
        
        # Skip updates with extremely large motions that are likely errors
        if rot_angle_deg > 10.0 or trans_mag > 1.0:  # Thresholds depend on your scene scale
            print("WARNING: Motion too large, likely an error. Skipping update.")
            return self.cur_pose

        # solvePnP gives the pose of the 3D points in the camera coordinate system.
        # To update the camera's global pose, we invert the relative motion.
        self.cur_pose = self.cur_pose @ np.linalg.inv(T_cur)
        
        # Log the updated camera pose
        log_transform("world/camera", self.cur_pose)
        
        # Update and visualize the trajectory
        position = self.cur_pose[:3, 3]
        self.trajectory.append(position)
        log_trajectory(self.trajectory)

        # Reinitialize feature set for the next frame using stereo matching but only 
        # if we have fewer than half the max features for efficiency
        if len(good_old_pts3d) < 100:  # Only reinitialize if running low on features
            pts3d, kps_left = self.triangulate_features(left_rect, right_rect)
            if pts3d is not None and kps_left is not None and len(pts3d) > 10:
                self.prev_left = left_rect
                self.prev_pts3d = pts3d
                self.prev_kps = np.float32(kps_left)
                print(f"Reinitialized with {len(pts3d)} new features")
            else:
                # If reinitialization fails, keep old 3D points but update previous image
                self.prev_left = left_rect
                print("Feature reinitialization failed, keeping old features")
        else:
            # Just update the previous image and keep tracking good features
            self.prev_left = left_rect

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
