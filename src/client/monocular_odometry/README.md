# Monocular Visual Odometry with Zenoh

This module implements monocular visual odometry using Apple's DepthPro model for depth estimation and Zenoh for receiving camera frames from a robot.

## Overview

The system performs the following steps:
1. Subscribes to camera frames from a robot via Zenoh
2. Estimates depth for each frame using Apple's DepthPro model
3. Tracks keypoints between consecutive frames using optical flow
4. Computes camera pose using PnP with RANSAC
5. Visualizes the results with keypoints, depth map, and pose information

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Zenoh
- PIL (Pillow)
- Transformers (Hugging Face)

## Installation

```bash
pip install torch opencv-python zenoh pillow transformers
```

## Usage

### Running the Monocular Odometry

Make sure the robot is publishing camera frames to the Zenoh topic `robot/camera/frame`. Then run:

```bash
python monocular_odom.py
```

### Command Line Options

- `--no-vis`: Disable visualization (useful for headless operation)
- `--force-cpu`: Force CPU usage even if GPU/MPS is available (not recommended for Apple Silicon as we now have MPS compatibility)
- `--skip-frames N`: Process only 1 out of N+1 frames for depth estimation (0=all frames, 1=every other frame, etc.)

```bash
# Run without visualization
python monocular_odom.py --no-vis

# Skip every other frame for depth estimation (improves performance)
python monocular_odom.py --skip-frames 1
```

## Visualization

The system provides two visualization windows:
1. **Monocular Odometry**: Shows the camera feed with tracked keypoints and pose information
2. **Depth**: Shows the depth map estimated by DepthPro

Press 'q' to quit the application.

## Integration with Robot

This module subscribes to the Zenoh topic `robot/camera/frame` to receive camera frames from the robot. The frames should be published as JPEG-encoded images.

To publish camera frames from a robot, you can use the `webcam_publisher.py` script in the `src/robot` directory.

## Notes

- The camera intrinsics (fx, fy, cx, cy) are currently hardcoded and may need to be adjusted for your specific camera.
- The depth estimation model (DepthPro) requires significant computational resources. Performance may vary depending on your hardware.
- For best results, ensure the camera moves smoothly and the scene contains sufficient texture for feature tracking.

## MPS Compatibility

This implementation includes special handling for Apple Silicon (M1/M2/M3) devices using the Metal Performance Shaders (MPS) backend. When operations that are not supported by MPS are detected (like the `im2col` operation), the system automatically:

1. Detects the incompatibility
2. Temporarily moves the specific operation to CPU
3. Moves back to MPS for subsequent operations

This approach provides the best of both worlds - using MPS acceleration where possible while gracefully falling back to CPU only for incompatible operations.

## Performance Optimization

- **Frame Skipping**: Use the `--skip-frames` option to reduce the frequency of depth estimation, which is the most computationally intensive part of the pipeline. For example, `--skip-frames 2` will only run depth estimation on every third frame, significantly improving performance.

- **Resolution**: If you're experiencing low FPS, try reducing the resolution of the camera frames in the webcam publisher.

- **Visualization**: Disable visualization with `--no-vis` for maximum performance in headless operation.

## Troubleshooting

### Performance Issues

If you're still experiencing performance issues with MPS on Apple Silicon:

1. Try increasing the frame skip value: `--skip-frames 2` or higher
2. Check the process time displayed on the visualization to identify bottlenecks
3. Monitor the console for any warnings about MPS compatibility issues

### Depth Estimation Errors

If you see errors related to depth estimation:

1. The system will automatically provide dummy depth maps to prevent crashes
2. Check the console for specific error messages
3. Try restarting the application 