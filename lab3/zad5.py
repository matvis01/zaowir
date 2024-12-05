import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Load calibration parameters
with open('stereo_calibration_data.json', 'r') as f:
    stereo_calibration_data = json.load(f)
    mtxL = np.array(stereo_calibration_data['camera_matrix_left'])
    distL = np.array(stereo_calibration_data['distortion_coefficients_left'])
    mtxR = np.array(stereo_calibration_data['camera_matrix_right'])
    distR = np.array(stereo_calibration_data['distortion_coefficients_right'])
    R = np.array(stereo_calibration_data['rotation_matrix'])
    T = np.array(stereo_calibration_data['translation_vector'])

# Load stereo images
imgL = cv2.imread('lab2/s1/left_10.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('lab2/s1/right_10.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if imgL is None or imgR is None:
    print("Error: Could not load images.")
    exit()

# Perform stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, imgL.shape[::-1], R, T, alpha=0)

# Compute the rectification maps
map1L, map2L = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, imgL.shape[::-1], cv2.CV_32FC1)
map1R, map2R = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, imgR.shape[::-1], cv2.CV_32FC1)

# Apply the rectification maps to the images
rectifiedL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

# Compute the disparity map using the best stereo matching method
numDisparities = 16 * 5  # Must be divisible by 16
blockSize = 19  # Must be odd
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
disparity = stereo.compute(rectifiedL, rectifiedR)
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Display the disparity map
plt.imshow(disparity, cmap='gray')
plt.title('Disparity Map with Rectification')
plt.colorbar()
plt.show()