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
imgL = cv.imread(images_left[10])
imgR = cv.imread(images_right[10])
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

# Resize images to half their original size for display
scale = 0.5
imgL_small = cv.resize(imgL, None, fx=scale, fy=scale)
imgR_small = cv.resize(imgR, None, fx=scale, fy=scale)
rectifiedL_small = cv.resize(rectifiedL, None, fx=scale, fy=scale)
rectifiedR_small = cv.resize(rectifiedR, None, fx=scale, fy=scale)

# Create combined images
original_combined = np.hstack((imgL_small, imgR_small))
rectified_combined = np.hstack((rectifiedL_small, rectifiedR_small))

# Add text labels
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(original_combined, 'Original Left', (10, 30), font, 1, (0, 255, 0), 2)
cv.putText(original_combined, 'Original Right', (imgL_small.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
cv.putText(rectified_combined, 'Rectified Left', (10, 30), font, 1, (0, 255, 0), 2)
cv.putText(rectified_combined, 'Rectified Right', (imgL_small.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)

# Stack original and rectified images vertically
final_display = np.vstack((original_combined, rectified_combined))

# Draw horizontal lines to help visualize rectification
line_interval = 50
for y in range(0, final_display.shape[0], line_interval):
    cv.line(final_display, (0, y), (final_display.shape[1], y), (0, 0, 255), 1)

# Display the results
cv.imshow('Original vs Rectified Images', final_display)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the results
cv.imwrite('comparison_result.png', final_display)
print("Results saved as 'comparison_result.png'")