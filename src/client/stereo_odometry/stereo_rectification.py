import rerun as rr 
import numpy as np
from PIL import Image
import cv2

from stereo_calibration import read_calibration
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

class StereoRectifier:
    """
    A class to handle stereo image rectification with a simple API.
    Manages calibration data and homography matrices internally.
    """
    
    def __init__(self, calibration_file='stereo_calibration.npz', visualize=True):
        """
        Initialize the StereoRectifier with calibration data.
        
        Args:
            calibration_file: Path to the calibration file
            visualize: Whether to visualize the rectification with Rerun
        """
        self.visualize = visualize
        self.calibration_data = read_calibration(calibration_file)
        
        # Extract calibration parameters
        self.dist_left = self.calibration_data["left_distortion"]
        self.dist_right = self.calibration_data["right_distortion"]
        self.matrix_left = self.calibration_data["left_camera_matrix"]
        self.matrix_right = self.calibration_data["right_camera_matrix"]
        
        # Initialize homography matrices to None (will be computed on first use)
        self.H1 = None
        self.H2 = None
        
        # Print available calibration keys for debugging
        print(f"Available calibration keys: {list(self.calibration_data.keys())}")
    
    def rectify_image(self, stereo_image):
        """
        Rectify a stereo image. Computes homography matrices on first call.
        
        Args:
            stereo_image: PIL Image or numpy array containing both left and right views
            
        Returns:
            tuple: (left_rectified, right_rectified) as numpy arrays
        """
        # Split the stereo image
        left_image, right_image = self._split_stereo_image(stereo_image)
        
        # Undistort the images
        left_undistorted, right_undistorted = self._undistort_images(left_image, right_image)
        
        # If we don't have homographies yet, compute them
        if self.H1 is None or self.H2 is None:
            self._compute_homographies(left_undistorted, right_undistorted)
            
        # If we still don't have valid homographies, return undistorted images
        if self.H1 is None or self.H2 is None:
            if self.visualize:
                log_stereo_images(left_undistorted, right_undistorted, stereo_image)
            return left_undistorted, right_undistorted
        
        # Apply homographies to rectify images
        image_size = left_undistorted.shape[1::-1]  # (width, height)
        left_rectified = cv2.warpPerspective(left_undistorted, self.H1, image_size)  # Testing showed this was more stable
        right_rectified = cv2.warpPerspective(right_undistorted, self.H2, image_size)
        
        # Visualize if requested
        if self.visualize:
            self._visualize_rectification(left_rectified, right_rectified, stereo_image)
        
        return left_rectified, right_rectified
    
    def _split_stereo_image(self, image):
        """Split a stereo image into left and right images."""
        # Convert PIL Image to numpy array if it's not already
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Get the width of the full image
        height, width = image.shape[:2]
        
        # Split the image in half horizontally
        mid_point = width // 2
        
        # Extract left and right images
        left_image = image[:, :mid_point].copy()
        right_image = image[:, mid_point:].copy()
        
        return left_image, right_image
    
    def _undistort_images(self, left_image, right_image):
        """Undistort left and right images using camera calibration parameters."""
        left_undistorted = cv2.undistort(left_image, self.matrix_left, self.dist_left)
        right_undistorted = cv2.undistort(right_image, self.matrix_right, self.dist_right)
        
        return left_undistorted, right_undistorted
    
    def _compute_homographies(self, left_image, right_image):
        """Compute rectification homographies using feature matching."""
        try:
            # Find corresponding points in both images
            sift = cv2.SIFT_create()
            
            # Find keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(left_image, None)
            kp2, des2 = sift.detectAndCompute(right_image, None)
            
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            # Use FLANN matcher
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Find matches
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test to get good matches
            good_matches = []
            pts1 = []
            pts2 = []
            
            for m, n in matches:
                if m.distance < 0.7 * n.distance:  # Ratio test
                    good_matches.append(m)
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)
            
            # Convert points to numpy arrays
            pts1 = np.float32(pts1)
            pts2 = np.float32(pts2)
            
            # Ensure we have enough points
            if len(pts1) < 8:
                print("Not enough matching points found")
                return
            
            # Compute fundamental matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            
            # Select only inlier points
            inlier_mask = mask.ravel() == 1
            pts1 = pts1[inlier_mask]
            pts2 = pts2[inlier_mask]
            
            # Filter good_matches to only include inliers
            inlier_matches = [m for i, m in enumerate(good_matches) if i < len(inlier_mask) and inlier_mask[i]]
            
            # Visualize matches if requested
            if self.visualize:
                match_img = cv2.drawMatches(
                    left_image, kp1, 
                    right_image, kp2, 
                    inlier_matches, None, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                rr.log("stereo/matches", rr.Image(match_img))
            
            # Compute rectification homographies
            image_size = left_image.shape[1::-1]  # (width, height)
            ret, H1, H2 = cv2.stereoRectifyUncalibrated(
                pts1, pts2, F, image_size
            )
            
            if not ret:
                print("Rectification failed")
                return
            
            # Store the homography matrices
            self.H1 = H1
            self.H2 = H2
            print("Successfully computed rectification homographies")
            
        except Exception as e:
            print(f"Error computing rectification homographies: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _visualize_rectification(self, left_image, right_image, original_image, line_spacing=30):
        """Draw horizontal lines on rectified images to visualize the rectification quality."""
        left_lines = left_image.copy()
        right_lines = right_image.copy()
        
        for i in range(0, left_image.shape[0], line_spacing):
            cv2.line(left_lines, (0, i), (left_image.shape[1], i), (0, 255, 0), 1)
            cv2.line(right_lines, (0, i), (right_image.shape[1], i), (0, 255, 0), 1)
        
        log_stereo_images(left_lines, right_lines, original_image)


def main():
    """Simple example of using the StereoRectifier class."""
    # Initialize Rerun for visualization
    rr.init("stereo camera processing", spawn=True)
    
    # Initialize camera
    camera = ZenohCamera()
    
    # Create a stereo rectifier
    rectifier = StereoRectifier()
    
    # Process frames
    while True:
        frame = camera.get_current_frame()
        if frame is not None:
            # Get rectified images with a single function call
            left_rectified, right_rectified = rectifier.rectify_image(frame)
            
            # Now you can use left_rectified and right_rectified for further processing


if __name__ == "__main__":
    main()