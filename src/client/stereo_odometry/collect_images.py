try:
    # Try absolute import first
    from src.client.camera_access.ZenohCamera import ZenohCamera
except ImportError:
    # Fall back to relative import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_access.ZenohCamera import ZenohCamera

import rerun as rr
import os
import datetime
import time

# Determine the project root directory (3 levels up from this file)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Create images directory in the project root
images_dir = os.path.join(project_root, "images")
try:
    os.makedirs(images_dir, exist_ok=True)
    print(f"Using images directory: {images_dir}")
except PermissionError:
    # Fall back to creating in the current working directory
    images_dir = os.path.join(os.getcwd(), "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"Permission denied at project root. Using images directory: {images_dir}")

camera = ZenohCamera()

rr.init("collect_images", spawn=True)

frame_count = 0
print(f"Starting image collection. Images will be saved to {images_dir}")
print("Press Ctrl+C to stop collection.")

try:
    while True:
        frame = camera.get_current_frame()

        if frame is None:
            print("No frame received, exiting.")
            break
        
        # Save the image with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(images_dir, f"frame_{timestamp}.jpg")
        
        try:
            frame.save(image_path)
            frame_count += 1
            if frame_count % 10 == 0:  # Only print every 10 frames to reduce console spam
                print(f"Saved {frame_count} images so far...")
        except Exception as e:
            print(f"Error saving image: {e}")

        # Display in rerun
        rr.log("camera", rr.Image(frame))
        
        # Small delay to avoid flooding the disk with images
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nImage collection stopped by user.")

print(f"Collection complete. Saved {frame_count} images to {images_dir}")
camera.close()