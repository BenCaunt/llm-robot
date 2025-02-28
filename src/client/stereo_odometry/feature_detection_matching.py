import cv2
import numpy as np
import rerun as rr
from typing import Tuple, List, Optional

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
from stereo_rectification import StereoRectifier


class StereoFeatureMatcher:
    """Class for detecting and matching features in stereo images."""
    
    def __init__(self, 
                 detector_type: str = 'ORB', 
                 use_bucketing: bool = True,
                 bucket_size: Tuple[int, int] = (10, 10),
                 max_features_per_bucket: int = 3,
                 max_disparity: int = 128,
                 block_size: int = 15,
                 visualize: bool = True):
        """
        Initialize the feature detector and matcher.
        
        Args:
            detector_type: Type of feature detector ('ORB', 'FAST', 'SIFT', etc.)
            use_bucketing: Whether to use bucketing for uniform feature distribution
            bucket_size: Size of buckets for feature distribution (rows, cols)
            max_features_per_bucket: Maximum number of features to keep per bucket
            max_disparity: Maximum disparity value for stereo matching
            block_size: Block size for feature matching
            visualize: Whether to visualize results with Rerun
        """
        self.detector_type = detector_type
        self.use_bucketing = use_bucketing
        self.bucket_size = bucket_size
        self.max_features_per_bucket = max_features_per_bucket
        self.max_disparity = max_disparity
        self.block_size = block_size
        self.visualize = visualize
        
        # Initialize feature detector
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1500)
        elif detector_type == 'FAST':
            self.detector = cv2.FastFeatureDetector_create()
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def detect_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect features in an image, optionally using bucketing for uniform distribution.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Debug: Check image type and shape
        print(f"Image type: {type(image)}, shape: {image.shape}, dtype: {image.dtype}")
        
        if not self.use_bucketing:
            # Simple feature detection without bucketing
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
            print(f"Without bucketing: detected {len(keypoints)} keypoints")
            return keypoints, descriptors
        
        # With bucketing: divide image into buckets and detect features in each
        height, width = image.shape[:2]
        bucket_height = height // self.bucket_size[0]
        bucket_width = width // self.bucket_size[1]
        
        print(f"Bucketing: image size {width}x{height}, bucket size {bucket_width}x{bucket_height}")
        
        all_keypoints = []
        bucket_counts = []  # For debugging
        
        # Detect features in each bucket
        for i in range(self.bucket_size[0]):
            for j in range(self.bucket_size[1]):
                # Calculate bucket boundaries
                y_start = i * bucket_height
                y_end = (i + 1) * bucket_height if i < self.bucket_size[0] - 1 else height
                x_start = j * bucket_width
                x_end = (j + 1) * bucket_width if j < self.bucket_size[1] - 1 else width
                
                # Extract bucket region
                bucket_image = image[y_start:y_end, x_start:x_end]
                
                # Skip empty buckets
                if bucket_image.size == 0:
                    print(f"Bucket [{i},{j}] is empty")
                    continue
                
                # Detect features in this bucket
                bucket_keypoints = self.detector.detect(bucket_image, None)
                
                # Debug: print bucket info
                if len(bucket_keypoints) > 0:
                    print(f"Bucket [{i},{j}] ({x_start},{y_start}) to ({x_end},{y_end}): {len(bucket_keypoints)} keypoints")
                
                # Adjust keypoint coordinates to global image coordinates
                for kp in bucket_keypoints:
                    kp.pt = (kp.pt[0] + x_start, kp.pt[1] + y_start)
                
                # Sort by response and keep the strongest ones
                bucket_keypoints = sorted(bucket_keypoints, key=lambda x: x.response, reverse=True)
                bucket_keypoints = bucket_keypoints[:self.max_features_per_bucket]
                
                all_keypoints.extend(bucket_keypoints)
                bucket_counts.append(len(bucket_keypoints))
        
        print(f"Total keypoints after bucketing: {len(all_keypoints)}")
        print(f"Bucket distribution: {bucket_counts}")
        
        # Check if we have any keypoints
        if len(all_keypoints) == 0:
            print("WARNING: No keypoints detected in any bucket!")
            # Try without bucketing as fallback
            print("Trying without bucketing...")
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
            print(f"Without bucketing: detected {len(keypoints)} keypoints")
            return keypoints, descriptors
        
        # Compute descriptors for all keypoints
        all_keypoints, descriptors = self.detector.compute(image, all_keypoints)
        
        # Check if compute returned valid results
        if all_keypoints is None or len(all_keypoints) == 0:
            print("WARNING: compute() returned no keypoints!")
            return [], None
        
        print(f"Final keypoints after compute: {len(all_keypoints)}")
        return all_keypoints, descriptors
    
    def match_stereo_features(self, 
                             left_image: np.ndarray, 
                             right_image: np.ndarray) -> Tuple[List[cv2.KeyPoint], 
                                                              List[cv2.KeyPoint], 
                                                              List[Tuple[int, int]], 
                                                              np.ndarray]:
        """
        Detect and match features between left and right stereo images.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            
        Returns:
            Tuple of (left_keypoints, right_keypoints, matches, disparity_map)
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        print(f"Left image shape: {left_gray.shape}, Right image shape: {right_gray.shape}")
        
        # Try direct feature detection first (skip bucketing)
        print("Detecting features directly in left image...")
        left_keypoints, left_descriptors = self.detector.detectAndCompute(left_gray, None)
        
        # If direct detection fails, try with bucketing
        if len(left_keypoints) == 0:
            print("Direct detection failed, trying with bucketing...")
            left_keypoints, left_descriptors = self.detect_features(left_gray)
        
        # Visualize detected keypoints for debugging
        if self.visualize and len(left_keypoints) > 0:
            left_with_kp = cv2.drawKeypoints(left_gray, left_keypoints, None, color=(0, 255, 0))
            rr.log("stereo/left_keypoints", rr.Image(left_with_kp))
        
        print(f"Detected {len(left_keypoints)} keypoints in left image")
        
        # Early exit if no keypoints
        if len(left_keypoints) == 0:
            print("No keypoints detected in left image, cannot match")
            return [], [], [], np.zeros(left_gray.shape, dtype=np.float32)
        
        # For each keypoint in left image, search along epipolar line in right image
        matched_left_kps = []
        matched_right_kps = []
        matches = []
        
        # Create a disparity map (same size as input images)
        disparity_map = np.zeros(left_gray.shape, dtype=np.float32)
        
        # Lower the matching threshold
        match_threshold = 0.6  # Reduced from 0.8
        
        # Reduce block size if it's too large
        effective_block_size = min(self.block_size, 11)  # Smaller block size
        
        print(f"Using block size: {effective_block_size}, threshold: {match_threshold}")
        
        # Count how many keypoints are skipped due to various reasons
        skipped_edge = 0
        skipped_search_range = 0
        skipped_dimensions = 0
        skipped_threshold = 0
        skipped_disparity = 0
        
        # For each keypoint in left image
        for i, kp in enumerate(left_keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Skip keypoints too close to the edge
            if x < effective_block_size//2 or x >= left_gray.shape[1] - effective_block_size//2 or \
               y < effective_block_size//2 or y >= left_gray.shape[0] - effective_block_size//2:
                skipped_edge += 1
                continue
            
            # Extract template from left image
            template = left_gray[y - effective_block_size//2:y + effective_block_size//2 + 1,
                               x - effective_block_size//2:x + effective_block_size//2 + 1]
            
            # Search range in right image (along same row, but only leftward)
            x_min = max(effective_block_size//2, x - self.max_disparity)
            
            # Ensure we have a valid search range
            if x_min >= right_gray.shape[1] - effective_block_size//2:
                skipped_search_range += 1
                continue
            
            # Extract the search row
            search_row = right_gray[max(0, y - effective_block_size//2):min(right_gray.shape[0], y + effective_block_size//2 + 1), 
                                  x_min - effective_block_size//2:min(right_gray.shape[1], x + effective_block_size//2 + 1)]
            
            # Check if search_row and template have valid dimensions
            if search_row.shape[0] < template.shape[0] or search_row.shape[1] < template.shape[1]:
                skipped_dimensions += 1
                continue
            
            # Use template matching to find the best match
            try:
                result = cv2.matchTemplate(search_row, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # Only keep good matches
                if max_val > match_threshold:  # Lower threshold for more matches
                    match_x = x_min - effective_block_size//2 + max_loc[0] + effective_block_size//2
                    
                    # Calculate disparity
                    disparity = x - match_x
                    
                    if disparity > 0 and disparity < self.max_disparity:
                        # Create a keypoint for the match in right image
                        right_kp = cv2.KeyPoint(match_x, y, kp.size)
                        
                        matched_left_kps.append(kp)
                        matched_right_kps.append(right_kp)
                        matches.append((i, len(matched_right_kps) - 1))
                        
                        # Update disparity map
                        disparity_map[y, x] = disparity
                    else:
                        skipped_disparity += 1
                else:
                    skipped_threshold += 1
                
            except Exception as e:
                print(f"Error matching template: {e}")
                continue
        
        # Print statistics
        print(f"Keypoint matching statistics:")
        print(f"  Total keypoints: {len(left_keypoints)}")
        print(f"  Skipped (edge): {skipped_edge}")
        print(f"  Skipped (search range): {skipped_search_range}")
        print(f"  Skipped (dimensions): {skipped_dimensions}")
        print(f"  Skipped (threshold): {skipped_threshold}")
        print(f"  Skipped (disparity): {skipped_disparity}")
        print(f"  Matched: {len(matches)}")
        
        # If we have matches, print some details about them
        if len(matches) > 0:
            # The error is in this line - right_keypoints doesn't exist
            # We need to use matched_right_kps and matched_left_kps directly
            disparities = []
            for i in range(len(matched_left_kps)):
                disparities.append(matched_left_kps[i].pt[0] - matched_right_kps[i].pt[0])
            
            print(f"  Disparity range: {min(disparities):.1f} to {max(disparities):.1f}, avg: {sum(disparities)/len(disparities):.1f}")
        
        # Visualize if requested
        if self.visualize:
            # Convert keypoints to numpy arrays for visualization
            if len(matched_left_kps) > 0:
                left_pts = np.array([kp.pt for kp in matched_left_kps])
                right_pts = np.array([kp.pt for kp in matched_right_kps])
                
                # Create a colorized disparity map for visualization
                disparity_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(disparity_map, alpha=255/self.max_disparity), 
                    cv2.COLORMAP_JET
                )
                
                # Log to Rerun
                rr.log("stereo/disparity", rr.Image(disparity_vis))
                
                # Create a match visualization
                match_img = np.hstack((left_image, right_image))
                
                # Draw matches
                for i in range(len(matched_left_kps)):
                    left_pt = (int(matched_left_kps[i].pt[0]), int(matched_left_kps[i].pt[1]))
                    right_pt = (int(matched_right_kps[i].pt[0] + left_image.shape[1]), 
                               int(matched_right_kps[i].pt[1]))
                    cv2.line(match_img, left_pt, right_pt, (0, 255, 0), 1)
                    cv2.circle(match_img, left_pt, 3, (0, 0, 255), -1)
                    cv2.circle(match_img, right_pt, 3, (0, 0, 255), -1)
                
                rr.log("stereo/matches", rr.Image(match_img))
                
                # Log stereo points
                log_stereo_points(left_pts, right_pts)
        
        return matched_left_kps, matched_right_kps, matches, disparity_map
    
    def compute_dense_disparity(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """
        Compute dense disparity map using Semi-Global Block Matching.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            
        Returns:
            Disparity map
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Create StereoSGBM object
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.max_disparity,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size**2,
            P2=32 * 3 * self.block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray)
        
        # Normalize disparity for visualization
        normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Visualize if requested
        if self.visualize:
            disparity_vis = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
            rr.log("stereo/dense_disparity", rr.Image(disparity_vis))
        
        return disparity


if __name__ == "__main__":
    rr.init("feature detection and matching", spawn=True)
    camera = ZenohCamera()
    rectifier = StereoRectifier(visualize=False)
    
    # Create feature matcher with default parameters
    matcher = StereoFeatureMatcher(
        detector_type='ORB',  # Try ORB first
        use_bucketing=True,
        bucket_size=(8, 10),
        max_features_per_bucket=5,  # Increased from 3
        max_disparity=128,
        block_size=9,  # Reduced from 15
        visualize=True
    )
    
    # Try different detector types if needed
    detector_types = ['ORB', 'FAST', 'SIFT']
    current_detector_idx = 0

    while True: 
        frame = camera.get_current_frame()
        if frame is not None:
            # Get rectified images
            left_rec, right_rec, left_undistorted, right_undistorted = rectifier.rectify_image(frame)
            
            # Log undistorted/rectified images
            log_stereo_images(left_rec, right_rec, frame)
            
            # Match features between stereo images
            left_kps, right_kps, matches, disparity = matcher.match_stereo_features(left_rec, right_rec)
            
            print(f"Found {len(matches)} stereo matches")
            
            # If template matching approach fails, try descriptor-based matching as fallback
            if len(matches) == 0:
                print("Trying descriptor-based matching instead...")
                
                # Convert to grayscale if needed
                if len(left_rec.shape) == 3:
                    left_gray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
                else:
                    left_gray = left_rec
                    right_gray = right_rec
                
                # Try a different detector if we've had multiple failures
                if current_detector_idx < len(detector_types) - 1:
                    current_detector_idx += 1
                    new_detector = detector_types[current_detector_idx]
                    print(f"Switching to {new_detector} detector")
                    matcher = StereoFeatureMatcher(
                        detector_type=new_detector,
                        use_bucketing=True,
                        bucket_size=(8, 10),
                        max_features_per_bucket=5,
                        max_disparity=128,
                        block_size=9,
                        visualize=True
                    )
                    continue
                
                # Detect features in both images
                orb = cv2.ORB_create(nfeatures=1000)
                kp1, des1 = orb.detectAndCompute(left_gray, None)
                kp2, des2 = orb.detectAndCompute(right_gray, None)
                
                print(f"Direct detection found {len(kp1)} keypoints in left, {len(kp2)} in right")
                
                # Use BFMatcher with Hamming distance
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    # Match descriptors
                    direct_matches = bf.match(des1, des2)
                    
                    # Sort matches by distance
                    direct_matches = sorted(direct_matches, key=lambda x: x.distance)
                    
                    # Filter matches by y-coordinate (should be similar in rectified images)
                    filtered_matches = []
                    for m in direct_matches:
                        y1 = int(kp1[m.queryIdx].pt[1])
                        y2 = int(kp2[m.trainIdx].pt[1])
                        x1 = int(kp1[m.queryIdx].pt[0])
                        x2 = int(kp2[m.trainIdx].pt[0])
                        
                        # Check if points are on roughly the same scanline and disparity is positive
                        if abs(y1 - y2) < 5 and x1 > x2 and x1 - x2 < matcher.max_disparity:
                            filtered_matches.append(m)
                    
                    # Draw matches
                    img_matches = cv2.drawMatches(left_gray, kp1, right_gray, kp2, filtered_matches[:50], None, 
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    rr.log("stereo/direct_matches", rr.Image(img_matches))
                    
                    print(f"Found {len(filtered_matches)} direct matches")
            
            # Compute dense disparity (optional)
            dense_disparity = matcher.compute_dense_disparity(left_rec, right_rec)

