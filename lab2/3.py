import cv2 as cv
import numpy as np
import json
import glob

# Load previously calculated stereo calibration data
calibration_file = 'stereo_calibration_data.json'
with open(calibration_file, 'r') as f:
    stereo_calibration_data = json.load(f)
    mtxL = np.array(stereo_calibration_data['camera_matrix_left'])
    distL = np.array(stereo_calibration_data['distortion_coefficients_left'])
    mtxR = np.array(stereo_calibration_data['camera_matrix_right'])
    distR = np.array(stereo_calibration_data['distortion_coefficients_right'])
    R = np.array(stereo_calibration_data['rotation_matrix'])
    T = np.array(stereo_calibration_data['translation_vector'])

# Load images
print("Loading images...")
images_left = sorted(glob.glob('lab2/s1/left_*.png'))
images_right = sorted(glob.glob('lab2/s1/right_*.png'))

# Read the first pair of images for rectification
imgL = cv.imread(images_left[0])
imgR = cv.imread(images_right[0])
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

# Perform stereo rectification
print("Performing stereo rectification...")
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=0
)

# Compute the rectification maps
map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv.CV_32FC1)
map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv.CV_32FC1)

# Apply the rectification maps to the images
rectifiedL = cv.remap(imgL, map1L, map2L, cv.INTER_LINEAR)
rectifiedR = cv.remap(imgR, map1R, map2R, cv.INTER_LINEAR)

# Display the rectified images side by side
combined = np.hstack((rectifiedL, rectifiedR))
cv.imshow('Rectified Images', combined)
cv.waitKey(0)
cv.destroyAllWindows()