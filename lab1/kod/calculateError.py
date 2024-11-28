import cv2 as cv
import numpy as np
import json

import cv2 as cv
import numpy as np
import json

# Load the calibration data from the JSON file
with open('calibration_data.json', 'r') as json_file:
    calibration_data = json.load(json_file)

# Extract the calibration parameters
mtx = np.array(calibration_data["camera_matrix"])
dist = np.array(calibration_data["distortion_coefficients"])
rvecs = [np.array(rvec) for rvec in calibration_data["rotation_vectors"]]
tvecs = [np.array(tvec) for tvec in calibration_data["translation_vectors"]]

# Load the object points and image points from the original calibration process
with open('points_data.json', 'r') as points_file:
    points_data = json.load(points_file)

objpoints = [np.array(objp) for objp in points_data["object_points"]]
imgpoints = [np.array(imgp) for imgp in points_data["image_points"]]

# Calculate the mean reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Mean reprojection error: {}".format(mean_error / len(objpoints)))