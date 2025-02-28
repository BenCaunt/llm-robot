import cv2
import numpy as np
import glob
import os

def read_calibration(calibration_file='stereo_calibration.npz'):
    """
    Read stereo camera calibration parameters from a saved file.
    
    :param calibration_file: Path to the calibration file (.npz format)
    :return: Dictionary containing all calibration parameters
    """
    try:
        # Load the calibration file
        data = np.load(calibration_file)
        
        # Create a dictionary with all calibration parameters
        calibration_data = {
            'left_camera_matrix': data['mtx_left'],
            'left_distortion': data['dist_left'],
            'right_camera_matrix': data['mtx_right'],
            'right_distortion': data['dist_right'],
            'rotation_matrix': data['R'],
            'translation_vector': data['T'],
            'essential_matrix': data['E'],
            'fundamental_matrix': data['F'],
            'left_camera_error': data['ret_left'],
            'right_camera_error': data['ret_right'],
            'stereo_error': data['ret_stereo']
        }
        
        print(f"Successfully loaded calibration data from {calibration_file}")
        return calibration_data
    
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return None

def stereo_calibrate(images_path='images',
                     board_width=8, 
                     board_height=6,
                     square_size=22.0,
                     save_file='stereo_calibration.npz'):
    """
    Perform stereo calibration for two cameras given side-by-side images.

    :param images_path: Directory containing the stereo images.
    :param board_width: Number of internal corners in width (e.g. 8 for a 9x7 checkerboard).
    :param board_height: Number of internal corners in height (e.g. 6 for a 9x7 checkerboard).
    :param square_size: The physical size of each square in real-world units (e.g., millimeters).
    :param save_file: Filename to save the calibration results.
    """

    # Criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the checkerboard size
    # E.g., (0,0,0), (1,0,0), (2,0,0) ... in real-world coordinates
    objp = np.zeros((board_height * board_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    # from all the images for LEFT and RIGHT, respectively.
    objpoints = []  # 3D points in real-world space
    imgpoints_left = []  # 2D points in left camera image plane
    imgpoints_right = [] # 2D points in right camera image plane

    # Get a list of all .jpg files in the specified directory
    pattern = os.path.join(images_path, '*.jpg')
    images = glob.glob(pattern)
    images.sort()
    
    # Check if any images were found
    if not images:
        print(f"No images found in {images_path}. Please check the path.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for images in: {os.path.abspath(images_path)}")
        return
    
    print(f"Found {len(images)} images for calibration.")
    
    # Initialize variables for image dimensions
    img_shape = None

    # For each image file
    for idx, fname in enumerate(images):
        # Read the stereo image
        full_image = cv2.imread(fname)
        if full_image is None:
            print(f"Could not read image {fname}. Skipping...")
            continue
        
        # Split left and right images (assuming width is even)
        height, width = full_image.shape[:2]
        half_width = width // 2
        img_left = full_image[:, :half_width]
        img_right = full_image[:, half_width:]
        
        # Convert to gray
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Store image shape for later use
        if img_shape is None:
            img_shape = gray_left.shape[::-1]

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, (board_width, board_height), None)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, (board_width, board_height), None)

        # If both left and right corners are found
        if ret_left and ret_right:
            # Refine corner locations
            corners_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            # Append to our list of points
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            
            print(f"Processed image {idx+1}/{len(images)}: {os.path.basename(fname)} - Checkerboard found")
        else:
            print(f"Processed image {idx+1}/{len(images)}: {os.path.basename(fname)} - Checkerboard NOT found")
    
    # Check if we have enough points for calibration
    if not objpoints:
        print("No checkerboard patterns were detected in any images. Cannot calibrate.")
        return
    
    print(f"Successfully detected checkerboard in {len(objpoints)} images.")
    
    # Calibrate each camera individually
    print("Calibrating left camera...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_shape, None, None)
    print("Calibrating right camera...")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_shape, None, None)

    # Stereo calibration
    print("Performing stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER +
                       cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape,
        criteria=criteria_stereo, flags=flags)

    print(f"calibration results: {ret_l}, {mtx_l}, {dist_l}, {rvecs_l}, {tvecs_l}")
    # Save the calibration results
    np.savez(save_file,
             ret_left=ret_l, mtx_left=mtx_l, dist_left=dist_l,
             ret_right=ret_r, mtx_right=mtx_r, dist_right=dist_r,
             ret_stereo=ret_stereo, R=R, T=T, E=E, F=F)
    print(f"Stereo calibration complete. Parameters saved to {save_file}.")
    
    # Return the calibration data
    return {
        'left_camera_matrix': mtx_l,
        'left_distortion': dist_l,
        'right_camera_matrix': mtx_r,
        'right_distortion': dist_r,
        'rotation_matrix': R,
        'translation_vector': T,
        'essential_matrix': E,
        'fundamental_matrix': F,
        'left_camera_error': ret_l,
        'right_camera_error': ret_r,
        'stereo_error': ret_stereo
    }

if __name__ == '__main__':
    # Example usage. You may adjust or remove as needed:
    stereo_calibrate(
        images_path='images',
        board_width=8,
        board_height=6,
        square_size=25.4 / 1000.0, # mm to meters
        save_file='stereo_calibration.npz'
    )
    
    # Example of how to read the calibration data
    # calibration_data = read_calibration('stereo_calibration.npz')
    # if calibration_data:
    #     print("Camera matrices:")
    #     print("Left:", calibration_data['left_camera_matrix'])
    #     print("Right:", calibration_data['right_camera_matrix'])
