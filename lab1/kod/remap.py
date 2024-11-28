import cv2 as cv
import numpy as np
import json

# Load the calibration data from the JSON file
with open('calibration_data.json', 'r') as json_file:
    calibration_data = json.load(json_file)

# Extract the calibration parameters
mtx = np.array(calibration_data["camera_matrix"])
dist = np.array(calibration_data["distortion_coefficients"])

# Load the image
img = cv.imread('./data/cam4/94.png')
h, w = img.shape[:2]

# Get the optimal new camera matrix
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.1, (w, h))

# Initialize undistort rectify map
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv.CV_32FC1)

# Remap the image using high-quality interpolation
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Verify the ROI and adjust if necessary
x, y, w, h = roi
if roi != (0, 0, 0, 0):
    dst = dst[y:y+h, x:x+w]

# Save the undistorted image
cv.imwrite('calibration_remap.png', dst)