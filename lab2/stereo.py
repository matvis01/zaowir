import cv2
import numpy as np
import os
import json

# Define the chessboard size
chessboard_size = (8, 6)

# Define the criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# File paths for calibration data
stereo_calibration_file = 'stereo_calibration_data.json'
points_data_file = 'points_data.json'
calibration_data_left_file = 'calibration_data_left.json'
calibration_data_right_file = 'calibration_data_right.json'

# Check if stereo calibration data already exists
if os.path.exists(stereo_calibration_file):
    print("Loading existing stereo calibration data...")
    with open(stereo_calibration_file, 'r') as json_file:
        stereo_calibration_data = json.load(json_file)
    
    # Extract data from loaded files
    mtxL = np.array(stereo_calibration_data["camera_matrix_left"])
    distL = np.array(stereo_calibration_data["distortion_coefficients_left"])
    mtxR = np.array(stereo_calibration_data["camera_matrix_right"])
    distR = np.array(stereo_calibration_data["distortion_coefficients_right"])
    R = np.array(stereo_calibration_data["rotation_matrix"])
    T = np.array(stereo_calibration_data["translation_vector"])
    E = np.array(stereo_calibration_data["essential_matrix"])
    F = np.array(stereo_calibration_data["fundamental_matrix"])
    
    print("Stereo calibration data loaded successfully.")
else:
    print("Stereo calibration data not found. Starting calibration process...")

    # Load calibration data for the left camera
    with open(calibration_data_left_file, 'r') as left_file:
        calibration_data_left = json.load(left_file)
    mtxL = np.array(calibration_data_left["camera_matrix"])
    distL = np.array(calibration_data_left["distortion_coefficients"])

    # Load calibration data for the right camera
    with open(calibration_data_right_file, 'r') as right_file:
        calibration_data_right = json.load(right_file)
    mtxR = np.array(calibration_data_right["camera_matrix"])
    distR = np.array(calibration_data_right["distortion_coefficients"])

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane for left camera
    imgpoints_right = []  # 2d points in image plane for right camera

    # Path to the images
    image_folder = 'lab2/s1'

    # List all images in the folder
    images_left = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.startswith('left') and fname.endswith('.png')]
    images_right = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.startswith('right') and fname.endswith('.png')]

    # Ensure both lists are sorted and have the same length
    images_left.sort()
    images_right.sort()

    # Initialize gray to None
    grayL = None
    grayR = None

    for left_image_path, right_image_path in zip(images_left, images_right):
        imgL = cv2.imread(left_image_path)
        imgR = cv2.imread(right_image_path)
        
        # Check if the images are loaded correctly
        if imgL is None or imgR is None:
            print(f"Failed to load images: {left_image_path}, {right_image_path}")
            continue
        
        # Convert to grayscale
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve corner detection
        grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
        grayR = cv2.GaussianBlur(grayR, (5, 5), 0)

        # Find the chessboard corners with different flags
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

        # If corners found in both images, refine and store them
        if retL and retR:
            objpoints.append(objp)
            corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2L)
            imgpoints_right.append(corners2R)
        else:
            print(f"Chessboard corners not found in images: {left_image_path}, {right_image_path}")

    # Stereo calibration if corners were found in any image
    if grayL is not None and objpoints and imgpoints_left and imgpoints_right:
        # Stereo calibration
        print("Performing stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
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

        with open(stereo_calibration_file, 'w') as json_file:
            json.dump(stereo_calibration_data, json_file, indent=4)

        # Save the object points and image points to a JSON file
        points_data = {
            "object_points": [objp.tolist() for objp in objpoints],
            "image_points_left": [imgp.tolist() for imgp in imgpoints_left],
            "image_points_right": [imgp.tolist() for imgp in imgpoints_right]
        }
        
        with open(points_data_file, 'w') as points_file:
            json.dump(points_data, points_file, indent=4)

        print("Stereo calibration completed.")
    else:
        print("No chessboard corners were found in any image pair.")

# Calculate the baseline (distance between the cameras)
baseline = np.linalg.norm(T)
print(f"The baseline (distance between the cameras) is: {baseline} units.")


