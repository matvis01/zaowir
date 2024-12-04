import cv2 as cv
import numpy as np
import glob
import json
import os

# Define the chessboard size
chessboard_size = (10, 7)
SINGLE_SQUARE = 50  # Size of a single square in mm

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * SINGLE_SQUARE

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane for left camera
imgpoints_right = []  # 2d points in image plane for right camera

# Load images
print("Loading images...")
images_left = sorted(glob.glob('kolos/kolokwium-zestaw/left/*.png'))
images_right = sorted(glob.glob('kolos/kolokwium-zestaw/right/*.png'))

print(f"Found {len(images_left)} left images and {len(images_right)} right images.")

# Zadanie 2.2: Save the list of files used for calibration
calibration_files = {
    "left_images": images_left,
    "right_images": images_right
}
with open('kolos/calibration_files_stereo.json', 'w') as f:
    json.dump(calibration_files, f)

# Function to find chessboard corners
def findChessboardCorners(image, chessboard_size):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

# Function to calibrate camera
def calibrateCamera(objpoints, imgpoints, image_shape):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    return ret, mtx, dist, rvecs, tvecs

# Find chessboard corners
for img_left, img_right in zip(images_left, images_right):
    print(f"Processing pair: {img_left} and {img_right}")
    imgL = cv.imread(img_left)
    imgR = cv.imread(img_right)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retL, cornersL = findChessboardCorners(imgL, chessboard_size)
    retR, cornersR = findChessboardCorners(imgR, chessboard_size)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

# Calibrate the left and right cameras
print("Calibrating left camera...")
retL, mtxL, distL, rvecsL, tvecsL = calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1])
print("Calibrating right camera...")
retR, mtxR, distR, rvecsR, tvecsR = calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1])

# Stereo calibration
print("Performing stereo calibration...")
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
flags = cv.CALIB_FIX_INTRINSIC
ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria=criteria, flags=flags
)

# Save stereo calibration data
print("Saving stereo calibration data...")
stereo_calibration_data = {
    "Return": ret,
    "CameraMatrix1": CM1.tolist(),
    "DistCoeffs1": dist1.tolist(),
    "CameraMatrix2": CM2.tolist(),
    "DistCoeffs2": dist2.tolist(),
    "RotationMatrix": R.tolist(),
    "TranslationVector": T.tolist(),
    "EssentialMatrix": E.tolist(),
    "FundamentalMatrix": F.tolist()
}
with open('kolos/stereo_calibration_data.json', 'w') as f:
    json.dump(stereo_calibration_data, f)

print("Stereo calibration data saved.")

# Zadanie 2.3: Rectify a selected pair of images
selected_frame_left = images_left[0]
selected_frame_right = images_right[0]

print(f"Rectifying selected frames: {selected_frame_left} and {selected_frame_right}")

imgL = cv.imread(selected_frame_left)
imgR = cv.imread(selected_frame_right)
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    CM1, dist1, CM2, dist2, grayL.shape[::-1], R, T, alpha=0
)

# Compute the rectification maps
map1L, map2L = cv.initUndistortRectifyMap(CM1, dist1, R1, P1, grayL.shape[::-1], cv.CV_32FC1)
map1R, map2R = cv.initUndistortRectifyMap(CM2, dist2, R2, P2, grayL.shape[::-1], cv.CV_32FC1)

# Apply the rectification maps to the images
rectifiedL = cv.remap(imgL, map1L, map2L, cv.INTER_LINEAR)
rectifiedR = cv.remap(imgR, map1R, map2R, cv.INTER_LINEAR)

# Save the rectified images
cv.imwrite('kolos/rectified_left.png', rectifiedL)
cv.imwrite('kolos/rectified_right.png', rectifiedR)

print(f"Rectified images saved as kolos/rectified_left.png and kolos/rectified_right.png")

# Zadanie 2.4: Calculate the baseline (distance between the cameras)
baseline_mm = np.linalg.norm(T)
baseline_cm = baseline_mm / 10  # Convert mm to cm
print(f"The baseline (distance between the cameras) is: {baseline_cm} cm.")