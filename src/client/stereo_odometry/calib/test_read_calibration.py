#!/usr/bin/env python3
"""
Test script to demonstrate how to use the read_calibration function
from the stereo_calibration module.
"""

from stereo_calibration import read_calibration
import numpy as np

def main():
    # Load the calibration data
    calibration_data = read_calibration('stereo_calibration.npz')
    
    if calibration_data:
        # Print some basic information about the calibration
        print("\nCalibration Summary:")
        print("-" * 50)
        
        # Camera matrices
        print("\nLeft Camera Matrix:")
        print(calibration_data['left_camera_matrix'])
        
        print("\nRight Camera Matrix:")
        print(calibration_data['right_camera_matrix'])
        
        # Distortion coefficients
        print("\nLeft Camera Distortion Coefficients:")
        print(calibration_data['left_distortion'])
        
        print("\nRight Camera Distortion Coefficients:")
        print(calibration_data['right_distortion'])
        
        # Stereo parameters
        print("\nRotation Matrix (Right camera with respect to Left camera):")
        print(calibration_data['rotation_matrix'])
        
        print("\nTranslation Vector (Right camera with respect to Left camera):")
        print(calibration_data['translation_vector'])
        
        # Calibration errors
        print("\nCalibration Errors:")
        print(f"Left Camera RMS Error: {calibration_data['left_camera_error']:.6f}")
        print(f"Right Camera RMS Error: {calibration_data['right_camera_error']:.6f}")
        print(f"Stereo Calibration RMS Error: {calibration_data['stereo_error']:.6f}")
        
        # Calculate and print the baseline (distance between cameras)
        baseline = np.linalg.norm(calibration_data['translation_vector'])
        print(f"\nBaseline (distance between cameras): {baseline:.2f} units")
        
        print("\nNote: If square_size was specified in mm during calibration,")
        print("      then the baseline and translation vector are also in mm.")
    else:
        print("Failed to load calibration data. Please run stereo_calibration.py first.")

if __name__ == "__main__":
    main() 