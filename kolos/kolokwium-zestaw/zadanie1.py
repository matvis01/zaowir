import cv2 as cv
import numpy as np
import glob
import json

# Define the chessboard size
chessboard_size = (10, 7)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane for left camera
imgpoints_right = []  # 2d points in image plane for right camera

# Load images
print("Loading images...")
images_left = sorted(glob.glob('kolos/kolokwium-zestaw/left/*.png'))
images_right = sorted(glob.glob('kolos/kolokwium-zestaw/right/*.png'))

print(f"Found {len(images_left)} left images and {len(images_right)} right images.")

# Zadanie 1.3: Save the list of files used for calibration
calibration_files = {
    "left_images": images_left,
    "right_images": images_right
}
with open('kolos/calibration_files.json', 'w') as f:
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

# Zadanie 1.1: Save intrinsic parameters
print("Saving intrinsic parameters...")
calibration_data_left = {
    "camera_matrix": mtxL.tolist(),
    "distortion_coefficients": distL.tolist()
}
with open('kolos/calibration_data_left.json', 'w') as f:
    json.dump(calibration_data_left, f)

calibration_data_right = {
    "camera_matrix": mtxR.tolist(),
    "distortion_coefficients": distR.tolist()
}
with open('kolos/calibration_data_right.json', 'w') as f:
    json.dump(calibration_data_right, f)

print("Intrinsic parameters saved.")

# Zadanie 1.2: Save distortion coefficients
print("Saving distortion coefficients...")
distortion_coefficients = {
    "left_camera": {
        "k1": distL[0][0],
        "k2": distL[0][1],
        "p1": distL[0][2],
        "p2": distL[0][3],
        "k3": distL[0][4]
    },
    "right_camera": {
        "k1": distR[0][0],
        "k2": distR[0][1],
        "p1": distR[0][2],
        "p2": distR[0][3],
        "k3": distR[0][4]
    }
}
with open('kolos/distortion_coefficients.json', 'w') as f:
    json.dump(distortion_coefficients, f)

print("Distortion coefficients saved.")

# Zadanie 1.4: Undistort a selected frame from both cameras
selected_frame_left = images_left[0]
selected_frame_right = images_right[0]

print(f"Undistorting selected frames: {selected_frame_left} and {selected_frame_right}")

imgL = cv.imread(selected_frame_left)
imgR = cv.imread(selected_frame_right)

hL, wL = imgL.shape[:2]
newcameramtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 0.1, (wL, hL))
undistortedL = cv.undistort(imgL, mtxL, distL, None, newcameramtxL)

hR, wR = imgR.shape[:2]
newcameramtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 0.1, (wR, hR))
undistortedR = cv.undistort(imgR, mtxR, distR, None, newcameramtxR)

# Verify the ROI and adjust if necessary
xL, yL, wL, hL = roiL
if roiL != (0, 0, 0, 0):
    undistortedL = undistortedL[yL:yL+hL, xL:xL+wL]

xR, yR, wR, hR = roiR
if roiR != (0, 0, 0, 0):
    undistortedR = undistortedR[yR:yR+hR, xR:xR+wR]

# Save the undistorted images
cv.imwrite('kolos/undistorted_left.png', undistortedL)
cv.imwrite('kolos/undistorted_right.png', undistortedR)

print(f"Undistorted images saved as kolos/undistorted_left.png and kolos/undistorted_right.png")

# Brief description of intrinsic parameters
description = """
Parametry wewnętrzne kamery obejmują macierz kamery i współczynniki dystorsji.
- Macierz kamery (mtx) zawiera ogniskowe (fx, fy) i środek optyczny (cx, cy) kamery.
  Jest reprezentowana jako:
      [ fx  0  cx ]
      [  0  fy  cy ]
      [  0   0   1 ]
- Współczynniki dystorsji (dist) uwzględniają zniekształcenia soczewki, które mogą powodować, że proste linie wydają się zakrzywione na obrazie.
  Te współczynniki są używane do korekcji zniekształceń na obrazach.
"""

print(description)