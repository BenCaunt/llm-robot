import cv2
import numpy as np
import torch
import requests
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# --------------------------------------------------
# 1) Initialize DepthPro Model
# --------------------------------------------------

device_name = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depthpro_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
depthpro_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

# Helper function to get a normalized depth map
def get_depth_map(pil_image):
    """Run Apple DepthPro on an image and return the depth array (H x W)."""
    inputs = depthpro_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depthpro_model(**inputs)

    post_processed_output = depthpro_processor.post_process_depth_estimation(
        outputs, target_sizes=[(pil_image.height, pil_image.width)],
    )[0]

    # The output from DepthPro may be a dictionary containing 'predicted_depth' etc.
    depth = post_processed_output["predicted_depth"]  # shape (H, W)
    
    # Normalize to 0-255 for easier display or debugging
    # but for actual 3D back-projection you'd keep metric units from the model
    depth = (depth - depth.min()) / depth.max()
    depth_255 = (depth * 255.).detach().cpu().numpy().astype("uint8")
    
    # For real usage in 3D, keep depth in meters or valid metric scale.
    # We'll return both (normalized for debug, float for real usage).
    return depth.detach().cpu().numpy(), depth_255

# --------------------------------------------------
# 2) Camera intrinsics (example)
# --------------------------------------------------
# Suppose you have fx, fy, cx, cy from your camera.
# If you get them from DepthPro, they come in post_processed_output
# under "focal_length", "principal_point", or "field_of_view".
# For demonstration, define them manually:
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

# Build an intrinsics matrix K
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

# --------------------------------------------------
# 3) Utility for back-projecting 2D -> 3D
# --------------------------------------------------
def back_project_2d_to_3d(pts_2d, depth_map, K):
    """Convert an array of 2D pixel coordinates into 3D points using depth map and K."""
    # pts_2d is (N,2) in pixel space
    # depth_map is HxW with the depth at each pixel
    # K is 3x3
    inv_K = np.linalg.inv(K)
    
    pts_3d = []
    for (u, v) in pts_2d:
        z = depth_map[int(v), int(u)]  # depth at that pixel
        if z <= 0:
            # Skip or handle invalid depth
            pts_3d.append([0, 0, 0])
            continue
        # [u, v, 1]^T
        uv1 = np.array([u, v, 1.0], dtype=np.float32)
        # 3D in camera coords:  (inv_K * uv1) * z
        xyz = z * (inv_K @ uv1)
        pts_3d.append(xyz)
    return np.array(pts_3d, dtype=np.float32)

# --------------------------------------------------
# 4) Initialize data structures for VO loop
# --------------------------------------------------
# Let's say we have a list of images in PIL or a stream from a camera
# frames = [list_of_PIL_images ...] # or a generator

# Keep track of the previous frame, its keypoints, depth, etc.
prev_frame_gray = None
prev_depth = None
prev_kpts = None

# Keep a 4x4 matrix for the global camera pose: starts at identity
global_pose = np.eye(4, dtype=np.float32)

# --------------------------------------------------
# 5) Main visual odometry loop
# --------------------------------------------------
def process_frame(pil_image):
    global prev_frame_gray, prev_depth, prev_kpts, global_pose
    
    # Convert current frame to grayscale OpenCV image
    curr_frame_cv = np.array(pil_image.convert("RGB"))  # convert to RGB
    curr_frame_gray = cv2.cvtColor(curr_frame_cv, cv2.COLOR_RGB2GRAY)
    
    # Estimate depth
    depth_float, _ = get_depth_map(pil_image)
    
    # If this is the first frame, detect keypoints and store them
    if prev_frame_gray is None:
        prev_frame_gray = curr_frame_gray
        prev_depth = depth_float
        # Detect features
        prev_kpts_cv = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=1000,
                                               qualityLevel=0.01, minDistance=10)
        if prev_kpts_cv is not None:
            prev_kpts = prev_kpts_cv.reshape(-1, 2)
        else:
            prev_kpts = None
        return global_pose  # still identity
    
    # --------------------------------------------------
    # 5a) Track keypoints forward with optical flow
    # --------------------------------------------------
    if prev_kpts is not None and len(prev_kpts) > 0:
        curr_kpts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, curr_frame_gray, prev_kpts.astype(np.float32), None
        )
        
        # Keep only good matches
        status = status.reshape(-1)
        good_prev = prev_kpts[status == 1]
        good_curr = curr_kpts[status == 1]
        
        # --------------------------------------------------
        # 5b) Convert the "good_prev" 2D points to 3D using prev_depth
        # --------------------------------------------------
        pts_3d = back_project_2d_to_3d(good_prev, prev_depth, K)

        # Filter out zero or invalid depths
        valid_mask = np.linalg.norm(pts_3d, axis=1) > 0.0
        pts_3d = pts_3d[valid_mask]
        good_curr = good_curr[valid_mask]
        
        if len(pts_3d) >= 6:  # need enough points for PnP
            # --------------------------------------------------
            # 5c) Estimate pose using PnP with RANSAC
            # --------------------------------------------------
            # We have 3D points from the previous frame, and corresponding 2D from current
            # solvePnPRansac expects shape (N,1,3) and (N,1,2)
            pts_3d_reshaped = np.expand_dims(pts_3d, axis=1)
            good_curr_reshaped = np.expand_dims(good_curr, axis=1)
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d_reshaped, good_curr_reshaped, K, distCoeffs=None,
                reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and rvec is not None and tvec is not None:
                # Convert rvec/tvec into 4x4 SE(3) transform
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = tvec.squeeze()
                
                # Update global pose: new_pose = old_pose * T
                # The transform T is how the camera moved from the old frame to the current.
                global_pose = global_pose @ np.linalg.inv(T)
                
                # Filter outliers if desired:
                # inliers is the set of indices that passed RANSAC. You can further refine the set.
        
        # --------------------------------------------------
        # 5d) Update keypoints for next iteration
        # --------------------------------------------------
        # Redetect keypoints if too few remain
        if len(good_curr) < 200:
            new_kpts_cv = cv2.goodFeaturesToTrack(curr_frame_gray, maxCorners=1000,
                                                  qualityLevel=0.01, minDistance=10)
            if new_kpts_cv is not None:
                prev_kpts = new_kpts_cv.reshape(-1, 2)
            else:
                prev_kpts = good_curr
        else:
            prev_kpts = good_curr
    else:
        # No previous keypoints, detect new
        kpts_cv = cv2.goodFeaturesToTrack(curr_frame_gray, maxCorners=1000,
                                          qualityLevel=0.01, minDistance=10)
        if kpts_cv is not None:
            prev_kpts = kpts_cv.reshape(-1, 2)
    
    # Update references
    prev_frame_gray = curr_frame_gray
    prev_depth = depth_float
    
    # Return the camera's current global SE(3) pose
    return global_pose

# --------------------------------------------------
# 6) Example usage on a small set of frames
# --------------------------------------------------
if __name__ == "__main__":
    # For demonstration, just load two frames from the same image or a sequence
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image1 = Image.open(requests.get(url, stream=True).raw)
    image2 = image1.copy()  # In practice, you'd load the next frame in a real sequence

    # Pass the first frame
    pose_after_frame1 = process_frame(image1)
    print("Pose after frame1:\n", pose_after_frame1)

    # Pass the second frame
    pose_after_frame2 = process_frame(image2)
    print("Pose after frame2:\n", pose_after_frame2)
