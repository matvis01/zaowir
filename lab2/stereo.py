import cv2 as cv
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt

# Define the chessboard size
chessboard_size = (9, 6)

# Define the termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane for left camera
imgpoints_right = []  # 2d points in image plane for right camera

# Load images
print("Loading images...")
images_left = sorted(glob.glob('lab2/s1/left_*.png'))
images_right = sorted(glob.glob('lab2/s1/right_*.png'))

# Check if points data exists
points_file = 'points_data.json'
if os.path.exists(points_file):
    print("Loading points data from file...")
    with open(points_file, 'r') as f:
        points_data = json.load(f)
        objpoints = [np.array(pts) for pts in points_data['object_points']]
        imgpoints_left = [np.array(pts) for pts in points_data['imgpoints_left']]
        imgpoints_right = [np.array(pts) for pts in points_data['imgpoints_right']]
else:
    print("Points data file not found. Performing calibration...")
    for img_left, img_right in zip(images_left, images_right):
        print(f"Processing pair: {img_left} and {img_right}")
        imgL = cv.imread(img_left)
        imgR = cv.imread(img_right)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        print("Finding chessboard corners...")
        retL, cornersL = cv.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR:
            print("Chessboard corners found.")
            objpoints.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(cornersR)
        else:
            print("Chessboard corners not found.")

    # Save points data to file
    print("Saving points data to file...")
    points_data = {
        'object_points': [pts.tolist() for pts in objpoints],
        'imgpoints_left': [pts.tolist() for pts in imgpoints_left],
        'imgpoints_right': [pts.tolist() for pts in imgpoints_right]
    }
    with open(points_file, 'w') as f:
        json.dump(points_data, f)

# Load previously calculated camera parameters
calibration_file_left = 'calibration_data_left.json'
calibration_file_right = 'calibration_data_right.json'
if os.path.exists(calibration_file_left) and os.path.exists(calibration_file_right):
    print("Loading calibration data from files...")
    with open(calibration_file_left, 'r') as f:
        calibration_data_left = json.load(f)
        mtxL = np.array(calibration_data_left['camera_matrix'])
        distL = np.array(calibration_data_left['distortion_coefficients'])
    with open(calibration_file_right, 'r') as f:
        calibration_data_right = json.load(f)
        mtxR = np.array(calibration_data_right['camera_matrix'])
        distR = np.array(calibration_data_right['distortion_coefficients'])
else:
    print("Calibration data files not found. Performing calibration...")
    # Calibrate the left and right cameras
    print("Calibrating left camera...")
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
    print("Calibrating right camera...")
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

    # Save calibration data to files
    print("Saving calibration data to files...")
    calibration_data_left = {
        "camera_matrix": mtxL.tolist(),
        "distortion_coefficients": distL.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecsL],
        "translation_vectors": [tvec.tolist() for tvec in tvecsL]
    }
    with open(calibration_file_left, 'w') as f:
        json.dump(calibration_data_left, f)

    calibration_data_right = {
        "camera_matrix": mtxR.tolist(),
        "distortion_coefficients": distR.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecsR],
        "translation_vectors": [tvec.tolist() for tvec in tvecsR]
    }
    with open(calibration_file_right, 'w') as f:
        json.dump(calibration_data_right, f)

# Ensure there are enough points for calibration
if len(objpoints) == 0 or len(imgpoints_left) == 0 or len(imgpoints_right) == 0:
    print("Error: Not enough points for calibration.")
    exit()

# Stereo calibration
print("Performing stereo calibration...")
flags = cv.CALIB_FIX_INTRINSIC
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria=criteria, flags=flags
)

# Save stereo calibration data
print("Saving stereo calibration data...")
stereo_calibration_data = {
    "camera_matrix_left": mtxL.tolist(),
    "distortion_coefficients_left": distL.tolist(),
    "camera_matrix_right": mtxR.tolist(),
    "distortion_coefficients_right": distR.tolist(),
    "rotation_matrix": R.tolist(),
    "translation_vector": T.tolist(),
    "essential_matrix": E.tolist(),
    "fundamental_matrix": F.tolist()
}

with open('stereo_calibration_data.json', 'w') as json_file:
    json.dump(stereo_calibration_data, json_file, indent=4)

print("Stereo calibration completed.")

# Task 2: Calculate Baseline
baseline = np.linalg.norm(T)
print(f"Baseline (distance between cameras): {baseline} units")

# Task 3: Rectify Images
print("Performing stereo rectification...")
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=0
)

# Save rectification data
rectification_data = {
    "rectification_transform_left": R1.tolist(),
    "rectification_transform_right": R2.tolist(),
    "projection_matrix_left": P1.tolist(),
    "projection_matrix_right": P2.tolist(),
    "disparity_to_depth_mapping_matrix": Q.tolist()
}

with open('rectification_data.json', 'w') as json_file:
    json.dump(rectification_data, json_file, indent=4)

print("Stereo rectification completed.")

# Task 4: Compare Interpolation Methods
interpolation_methods = {
    "INTER_NEAREST": cv.INTER_NEAREST,
    "INTER_LINEAR": cv.INTER_LINEAR,
    "INTER_CUBIC": cv.INTER_CUBIC,
    "INTER_AREA": cv.INTER_AREA,
    "INTER_LANCZOS4": cv.INTER_LANCZOS4
}

print("Comparing interpolation methods...")
for method_name, method in interpolation_methods.items():
    print(f"Using interpolation method: {method_name}")
    map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv.CV_32FC1)
    map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv.CV_32FC1)
    rectifiedL = cv.remap(cv.imread(images_left[0]), map1L, map2L, method)
    rectifiedR = cv.remap(cv.imread(images_right[0]), map1R, map2R, method)
    combined = np.hstack((rectifiedL, rectifiedR))
    cv.imshow(f"Rectified Images - {method_name}", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Task 5: Visualize Epipolar Lines
print("Visualizing epipolar lines...")
def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

# Find epilines corresponding to points in right image (second image) and draw them on the left image
lines1 = cv.computeCorrespondEpilines(imgpoints_right[0].reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = draw_epipolar_lines(cv.imread(images_left[0]), cv.imread(images_right[0]), lines1, imgpoints_left[0], imgpoints_right[0])

# Find epilines corresponding to points in left image (first image) and draw them on the right image
lines2 = cv.computeCorrespondEpilines(imgpoints_left[0].reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = draw_epipolar_lines(cv.imread(images_right[0]), cv.imread(images_left[0]), lines2, imgpoints_right[0], imgpoints_left[0])

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

# Task 6: Export Rectified Images
print("Exporting rectified images...")
map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv.CV_32FC1)
map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv.CV_32FC1)
rectifiedL = cv.remap(cv.imread(images_left[0]), map1L, map2L, cv.INTER_LINEAR)
rectifiedR = cv.remap(cv.imread(images_right[0]), map1R, map2R, cv.INTER_LINEAR)
cv.imwrite('rectified_left.png', rectifiedL)
cv.imwrite('rectified_right.png', rectifiedR)
print("Rectified images exported.")