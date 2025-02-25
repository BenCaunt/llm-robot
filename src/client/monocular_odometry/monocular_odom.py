import cv2
import numpy as np
import torch
import zenoh
import time
import threading
import argparse
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# --------------------------------------------------
# 1) Initialize DepthPro Model
# --------------------------------------------------

def get_device(force_cpu=False):
    """Get the appropriate device based on availability and user preference."""
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available() and not force_cpu:
        # MPS (Metal Performance Shaders) for Apple Silicon
        print("MPS is available and will be used for compatible operations.")
        return torch.device("mps")
    
    return torch.device("cpu")

# Parse command line arguments early to determine device
parser = argparse.ArgumentParser(description="Monocular Visual Odometry with Zenoh")
parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
parser.add_argument("--force-cpu", action="store_true", 
                    help="Force CPU usage even if GPU/MPS is available")
parser.add_argument("--skip-frames", type=int, default=0,
                    help="Process only 1 out of N frames for depth estimation (0=all frames)")
args, _ = parser.parse_known_args()  # Partial parsing to get device info

# Set up device
device = get_device(force_cpu=args.force_cpu)
print(f"Using device: {device}")

# Initialize model on the selected device
depthpro_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
depthpro_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

# Constants for Zenoh
CAMERA_FRAME_KEY = "robot/camera/frame"

# MPS compatibility layer for operations that might fall back to CPU
class MPSCompatibilityLayer:
    def __init__(self, device):
        self.device = device
        self.is_mps = device.type == 'mps'
        # Keep track of the last depth map for frame skipping
        self.last_depth = None
        self.last_depth_255 = None
        self.frame_counter = 0
        
    def process_depth(self, pil_image, skip_frames=0):
        """Process depth with MPS compatibility and optional frame skipping."""
        # If frame skipping is enabled and not on a key frame, return the last depth
        if skip_frames > 0:
            self.frame_counter += 1
            if self.frame_counter % (skip_frames + 1) != 0 and self.last_depth is not None:
                return self.last_depth, self.last_depth_255
        
        try:
            # Process the image with the model
            inputs = depthpro_processor(images=pil_image, return_tensors="pt").to(self.device)
            
            # If using MPS, handle potential incompatible operations
            if self.is_mps:
                # Move to CPU for operations that might use im2col
                with torch.no_grad():
                    # First run the model on MPS
                    try:
                        outputs = depthpro_model(**inputs)
                    except RuntimeError as e:
                        if "not currently supported on the MPS backend" in str(e):
                            print("Detected MPS incompatibility, moving to CPU for this operation...")
                            # Move model and inputs to CPU for this operation
                            cpu_inputs = {k: v.to('cpu') for k, v in inputs.items()}
                            outputs = depthpro_model.to('cpu')(**cpu_inputs)
                            # Move model back to MPS for next operations
                            depthpro_model.to(self.device)
                        else:
                            raise e
            else:
                # For CPU or CUDA, just run normally
                with torch.no_grad():
                    outputs = depthpro_model(**inputs)
            
            # Post-process the outputs
            post_processed_output = depthpro_processor.post_process_depth_estimation(
                outputs, target_sizes=[(pil_image.height, pil_image.width)],
            )[0]
            
            # Extract and normalize depth
            depth = post_processed_output["predicted_depth"]  # shape (H, W)
            depth = (depth - depth.min()) / depth.max()
            depth_255 = (depth * 255.).detach().cpu().numpy().astype("uint8")
            
            # Store for frame skipping
            self.last_depth = depth.detach().cpu().numpy()
            self.last_depth_255 = depth_255
            
            return self.last_depth, self.last_depth_255
            
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            # Return dummy depth maps in case of error
            h, w = pil_image.height, pil_image.width
            dummy_depth = np.ones((h, w), dtype=np.float32)
            dummy_depth_255 = np.ones((h, w), dtype=np.uint8) * 128
            return dummy_depth, dummy_depth_255

# Initialize the MPS compatibility layer
mps_compat = MPSCompatibilityLayer(device)

# Helper function to get a normalized depth map
def get_depth_map(pil_image):
    """Run Apple DepthPro on an image and return the depth array (H x W)."""
    return mps_compat.process_depth(pil_image, skip_frames=args.skip_frames)

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
# Keep track of the previous frame, its keypoints, depth, etc.
prev_frame_gray = None
prev_depth = None
prev_kpts = None
curr_kpts = None  # Store current keypoints for visualization

# Keep a 4x4 matrix for the global camera pose: starts at identity
global_pose = np.eye(4, dtype=np.float32)

# --------------------------------------------------
# 5) Main visual odometry loop
# --------------------------------------------------
def process_frame(pil_image):
    global prev_frame_gray, prev_depth, prev_kpts, global_pose, curr_kpts
    
    # Convert current frame to grayscale OpenCV image
    curr_frame_cv = np.array(pil_image.convert("RGB"))  # convert to RGB
    curr_frame_gray = cv2.cvtColor(curr_frame_cv, cv2.COLOR_RGB2GRAY)
    
    # Estimate depth
    depth_float, depth_255 = get_depth_map(pil_image)
    
    # If this is the first frame, detect keypoints and store them
    if prev_frame_gray is None:
        prev_frame_gray = curr_frame_gray
        prev_depth = depth_float
        # Detect features
        prev_kpts_cv = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=1000,
                                               qualityLevel=0.01, minDistance=10)
        if prev_kpts_cv is not None:
            prev_kpts = prev_kpts_cv.reshape(-1, 2)
            curr_kpts = prev_kpts.copy()  # Initialize curr_kpts
        else:
            prev_kpts = None
            curr_kpts = None
        return global_pose, depth_255  # still identity
    
    # --------------------------------------------------
    # 5a) Track keypoints forward with optical flow
    # --------------------------------------------------
    if prev_kpts is not None and len(prev_kpts) > 0:
        curr_kpts_flow, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, curr_frame_gray, prev_kpts.astype(np.float32), None
        )
        
        # Keep only good matches
        status = status.reshape(-1)
        good_prev = prev_kpts[status == 1]
        good_curr = curr_kpts_flow[status == 1]
        
        # Store current keypoints for visualization
        curr_kpts = good_curr
        
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
                curr_kpts = prev_kpts.copy()  # Update curr_kpts
            else:
                prev_kpts = good_curr
                curr_kpts = good_curr.copy()
        else:
            prev_kpts = good_curr
            curr_kpts = good_curr.copy()
    else:
        # No previous keypoints, detect new
        kpts_cv = cv2.goodFeaturesToTrack(curr_frame_gray, maxCorners=1000,
                                          qualityLevel=0.01, minDistance=10)
        if kpts_cv is not None:
            prev_kpts = kpts_cv.reshape(-1, 2)
            curr_kpts = prev_kpts.copy()  # Update curr_kpts
    
    # Update references
    prev_frame_gray = curr_frame_gray
    prev_depth = depth_float
    
    # Return the camera's current global SE(3) pose and depth visualization
    return global_pose, depth_255

# --------------------------------------------------
# 6) Visualization utilities
# --------------------------------------------------
def draw_keypoints(frame, keypoints):
    """Draw keypoints on the frame for visualization."""
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    vis_frame = frame.copy()
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
    
    return vis_frame

def draw_pose_info(frame, pose):
    """Draw pose information on the frame."""
    # Extract translation (position) from the pose matrix
    tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
    
    # Extract rotation in Euler angles (roll, pitch, yaw)
    # This is a simplified conversion and might not be accurate for all cases
    R = pose[:3, :3]
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # Draw text on the frame
    cv2.putText(frame, f"Position: ({tx:.2f}, {ty:.2f}, {tz:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Rotation: ({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# --------------------------------------------------
# 7) Zenoh-based implementation for live camera feed
# --------------------------------------------------
class MonocularOdometryZenoh:
    def __init__(self, visualize=True):
        # Initialize Zenoh session
        self.session = zenoh.open(zenoh.Config())
        
        # Initialize a lock and storage for the latest camera frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Subscribe to the camera feed published from the robot
        self.frame_sub = self.session.declare_subscriber(CAMERA_FRAME_KEY, self.on_frame)
        
        # For FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.monotonic()
        
        # Flag to signal program termination
        self.should_exit = False
        
        # Visualization flag
        self.visualize = visualize
        
        # Performance monitoring
        self.depth_times = []
        self.tracking_times = []
        
    def on_frame(self, sample):
        """Callback for Zenoh subscriber: decodes the frame and stores it."""
        try:
            # Convert the Zenoh payload to a numpy array
            np_arr = np.frombuffer(sample.payload.to_bytes(), np.uint8)
            
            # Decode the JPEG image
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Store the frame with thread safety
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            print(f"Error processing camera frame: {e}")
    
    def run(self):
        """Main loop: processes incoming frames and computes visual odometry."""
        try:
            print("Running monocular odometry with Zenoh camera feed.")
            print("Press 'q' to quit.")
            
            while not self.should_exit:
                # Get the latest frame with thread safety
                frame = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                
                if frame is not None:
                    # Convert OpenCV BGR to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Process the frame with our visual odometry
                    start_time = time.time()
                    pose, depth_vis = process_frame(pil_image)
                    process_time = time.time() - start_time
                    
                    # Visualization
                    if self.visualize:
                        # Draw keypoints on the frame
                        if curr_kpts is not None:
                            frame = draw_keypoints(frame, curr_kpts)
                        
                        # Draw pose information
                        frame = draw_pose_info(frame, pose)
                        
                        # Add FPS counter
                        current_time = time.monotonic()
                        fps = self.frame_count / (current_time - self.fps_start_time) if current_time > self.fps_start_time else 0
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add processing time
                        cv2.putText(frame, f"Process time: {process_time*1000:.1f} ms", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show device info
                        device_text = f"Device: {device.type}"
                        if args.skip_frames > 0:
                            device_text += f" (skip={args.skip_frames})"
                        cv2.putText(frame, device_text, (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show the frame
                        cv2.imshow("Monocular Odometry", frame)
                        
                        # Show depth visualization
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imshow("Depth", depth_color)
                        
                        # Check for key press to exit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.should_exit = True
                    
                    # Calculate and display FPS
                    self.frame_count += 1
                    if self.frame_count % 10 == 0:  # Print every 10 frames
                        current_time = time.monotonic()
                        fps = self.frame_count / (current_time - self.fps_start_time)
                        print(f"FPS: {fps:.1f}")
                        print(f"Process time: {process_time*1000:.1f} ms")
                        print(f"Current pose:\n{pose}")
                        self.frame_count = 0
                        self.fps_start_time = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.session.close()
            if self.visualize:
                cv2.destroyAllWindows()

# --------------------------------------------------
# 8) Main entry point
# --------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments (full parsing)
    parser = argparse.ArgumentParser(description="Monocular Visual Odometry with Zenoh")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force CPU usage even if GPU/MPS is available")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Process only 1 out of N frames for depth estimation (0=all frames)")
    args = parser.parse_args()
    
    # Create and run the Zenoh-based monocular odometry
    mono_odom = MonocularOdometryZenoh(visualize=not args.no_vis)
    mono_odom.run()
