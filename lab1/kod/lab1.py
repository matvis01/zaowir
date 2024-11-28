import cv2
import numpy as np
import os
import json

# Define the chessboard size
chessboard_size = (8, 6)

# Define the criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Path to the images
image_folder = 'lab2/s1'


# List all images in the folder
images = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.png')]

# Initialize gray to None
gray = None

for image_path in images:
    img = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve corner detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find the chessboard corners with different flags
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If corners found, refine and store them
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners for visual verification
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(1000)  # Increase wait time to see the result
    else:
        print(f"Chessboard corners not found in image: {image_path}")

# Close all OpenCV windows
cv2.destroyAllWindows()

# Camera calibration if corners were found in any image
if gray is not None and objpoints and imgpoints:
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Convert calibration results to lists for JSON serialization
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecs],
        "translation_vectors": [tvec.tolist() for tvec in tvecs]
    }

    # Save the calibration results to a JSON file
    with open('calibration_data.json', 'w') as json_file:
        json.dump(calibration_data, json_file, indent=4)
    
    # Save the object points and image points to a JSON file
    points_data = {
        "object_points": [objp.tolist() for objp in objpoints],
        "image_points": [imgp.tolist() for imgp in imgpoints]
    }
    
    with open('points_data.json', 'w') as points_file:
        json.dump(points_data, points_file, indent=4)
    
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
else:
    print("No chessboard corners were found in any image.")