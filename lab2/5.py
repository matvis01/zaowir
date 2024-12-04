import cv2 as cv
import numpy as np
import json
import glob
import matplotlib.pyplot as plt

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
    F = np.array(stereo_calibration_data['fundamental_matrix'])

# Load points data
points_file = 'points_data.json'
with open(points_file, 'r') as f:
    points_data = json.load(f)
    objpoints = [np.array(pts) for pts in points_data['object_points']]
    imgpoints_left = [np.array(pts) for pts in points_data['imgpoints_left']]
    imgpoints_right = [np.array(pts) for pts in points_data['imgpoints_right']]

# Load images
print("Loading images...")
images_left = sorted(glob.glob('lab2/s1/left_*.png'))
images_right = sorted(glob.glob('lab2/s1/right_*.png'))

# Read the first pair of images for visualization
imgL = cv.imread(images_left[10])
imgR = cv.imread(images_right[10])
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

# Function to draw epipolar lines
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
        img1 = cv.circle(img1, (int(pt1[0][0]), int(pt1[0][1])), 5, color, -1)
        img2 = cv.circle(img2, (int(pt2[0][0]), int(pt2[0][1])), 5, color, -1)
    return img1, img2

# Find epilines corresponding to points in right image (second image) and draw them on the left image
lines1 = cv.computeCorrespondEpilines(imgpoints_right[0].reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = draw_epipolar_lines(grayL, grayR, lines1, imgpoints_left[0], imgpoints_right[0])

# Find epilines corresponding to points in left image (first image) and draw them on the right image
lines2 = cv.computeCorrespondEpilines(imgpoints_left[0].reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = draw_epipolar_lines(grayR, grayL, lines2, imgpoints_right[0], imgpoints_left[0])

# Display the images with epipolar lines
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

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

# Draw the useful area without distortions
x, y, w, h = roi1
rectifiedL = cv.rectangle(rectifiedL, (x, y), (x + w, y + h), (0, 255, 0), 2)
x, y, w, h = roi2
rectifiedR = cv.rectangle(rectifiedR, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the rectified images with useful area marked
combined = np.hstack((rectifiedL, rectifiedR))
cv.imshow('Rectified Images with Useful Area', combined)
cv.waitKey(0)
cv.destroyAllWindows()