import numpy as np
import cv2
import rerun as rr
from rerun.datatypes import Angle, RotationAxisAngle
from scipy.spatial.transform import Rotation as R

def matrix_to_axis_angle(matrix):
    """
    Convert a 4x4 transformation matrix to axis-angle representation for Rerun.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        Tuple of (axis, angle, translation) where:
            - axis is the normalized rotation axis
            - angle is the rotation angle in radians
            - translation is the translation vector
    """
    # Extract rotation matrix (3x3)
    rot_matrix = matrix[:3, :3]
    # Convert to scipy Rotation
    r = R.from_matrix(rot_matrix)
    # Convert to axis-angle
    axis_angle = r.as_rotvec()
    # Normalize axis and get angle
    if np.linalg.norm(axis_angle) > 0:
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle
    else:
        angle = 0
        axis = np.array([0, 0, 1])  # Default axis if no rotation
    
    # Extract translation
    translation = matrix[:3, 3]
    
    return axis, angle, translation

def log_transform(path, matrix):
    """
    Log a transformation matrix as a Transform3D in Rerun.
    
    Args:
        path: Rerun path to log the transform
        matrix: 4x4 transformation matrix
    """
    axis, angle, translation = matrix_to_axis_angle(matrix)
    rotation = RotationAxisAngle(axis=axis, angle=Angle(rad=angle))
    rr.log(path, rr.Transform3D(
        translation=translation,
        rotation=rotation
    ))

def log_stereo_images(left_img, right_img, stereo_img=None):
    """
    Log stereo image pairs to Rerun.
    
    Args:
        left_img: Left image (numpy array)
        right_img: Right image (numpy array)
        stereo_img: Optional combined stereo image
    """
    rr.log("input/left", rr.Image(left_img))
    rr.log("input/right", rr.Image(right_img))
    
    if stereo_img is not None:
        rr.log("input/stereo", rr.Image(stereo_img))

def log_undistorted_images(left_undist, right_undist):
    """
    Log undistorted stereo image pairs to Rerun.
    
    Args:
        left_undist: Undistorted left image
        right_undist: Undistorted right image
    """
    rr.log("undistorted/left", rr.Image(left_undist))
    rr.log("undistorted/right", rr.Image(right_undist))

def log_keypoints_and_matches(left_img, right_img, kps_left, kps_right, matches):
    """
    Log keypoints and matches between stereo images to Rerun.
    
    Args:
        left_img: Left image
        right_img: Right image
        kps_left: Keypoints in left image (cv2.KeyPoint objects)
        kps_right: Keypoints in right image (cv2.KeyPoint objects)
        matches: Matches between keypoints
    """
    # Visualize detected keypoints in both images
    left_vis = cv2.drawKeypoints(left_img, kps_left, None, color=(0, 255, 0))
    right_vis = cv2.drawKeypoints(right_img, kps_right, None, color=(0, 255, 0))
    
    rr.log("stereo/left/keypoints", rr.Image(left_vis))
    rr.log("stereo/right/keypoints", rr.Image(right_vis))
    
    # Visualize matches
    matches = sorted(matches, key=lambda x: x.distance)
    matches_vis = cv2.drawMatches(
        left_img, kps_left, right_img, kps_right, 
        matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    rr.log("stereo/matches", rr.Image(matches_vis))

def log_stereo_points(pts_left, pts_right):
    """
    Log matched points in stereo images as 2D point clouds.
    
    Args:
        pts_left: Points in left image (Nx2 numpy array)
        pts_right: Points in right image (Nx2 numpy array)
    """
    stereo_points = np.array([(x, y) for (x, y) in pts_left])
    rr.log("stereo/left_points", rr.Points2D(stereo_points, colors=(0, 255, 0), radii=3))
    
    stereo_points_right = np.array([(x, y) for (x, y) in pts_right])
    rr.log("stereo/right_points", rr.Points2D(stereo_points_right, colors=(255, 0, 0), radii=3))

def log_3d_points(pts3d, path="world/points3d", colors=(0, 255, 0), radii=0.01):
    """
    Log 3D points to Rerun.
    
    Args:
        pts3d: 3D points (Nx3 numpy array)
        path: Rerun path to log the points
        colors: Color for the points
        radii: Radius for the points
    """
    rr.log(path, rr.Points3D(pts3d, colors=colors, radii=radii))

def log_trajectory(trajectory_points, path="world/trajectory", colors=(255, 0, 0), radii=0.05):
    """
    Log trajectory points to Rerun.
    
    Args:
        trajectory_points: List of trajectory points (camera positions)
        path: Rerun path to log the trajectory
        colors: Color for the trajectory points
        radii: Radius for the trajectory points
    """
    rr.log(path, rr.Points3D(trajectory_points, colors=colors, radii=radii))

def log_optical_flow(prev_frame, curr_frame, prev_pts, curr_pts, status):
    """
    Visualize optical flow tracking results.
    
    Args:
        prev_frame: Previous grayscale frame
        curr_frame: Current grayscale frame
        prev_pts: Previous tracked points (Nx1x2 numpy array)
        curr_pts: Current tracked points (Nx1x2 numpy array)
        status: Status of tracked points (Nx1 numpy array, 1=success)
    """
    flow_vis = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
    for i, (new, old) in enumerate(zip(curr_pts, prev_pts)):
        if status[i] == 1:
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            # Draw the tracks with arrows
            flow_vis = cv2.arrowedLine(flow_vis, (int(x_old), int(y_old)), (int(x_new), int(y_new)), 
                                     (0, 255, 0), 1, tipLength=0.3)
    
    rr.log("tracking/optical_flow", rr.Image(flow_vis))

def log_tracked_points(points, path="tracking/tracked_points", colors=(0, 255, 0), radii=3):
    """
    Log tracked points to Rerun.
    
    Args:
        points: 2D points (Nx2 numpy array)
        path: Rerun path to log the points
        colors: Color for the points
        radii: Radius for the points
    """
    if len(points) > 0:
        rr.log(path, rr.Points2D(points, colors=colors, radii=radii))

def set_frame_time(frame_count):
    """
    Set the Rerun timeline for the current frame.
    
    Args:
        frame_count: Current frame count
    
    Returns:
        Updated frame count
    """
    rr.set_time_sequence("frame", frame_count)
    return frame_count + 1
