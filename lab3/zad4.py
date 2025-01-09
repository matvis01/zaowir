import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the stereo images
imgL = cv2.imread('lab3/cones/im2.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('lab3/cones/im6.png', cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread('lab3/cones/disp2.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if imgL is None or imgR is None or ground_truth is None:
    print("Error: Could not load images.")
    exit()

# Normalize ground truth disparity map
ground_truth = ground_truth / 4.0

# Function to compute disparity using StereoBM
def compute_disparity_bm(imgL, imgR):
    numDisparities = 16 * 4  # Must be divisible by 16
    blockSize = 21  # Must be odd
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.uint8(disparity)

# Function to compute disparity using StereoSGBM
def compute_disparity_sgbm(imgL, imgR):
    minDisparity = 0
    numDisparities = 16 * 4  # Must be divisible by 16
    blockSize = 3  # Must be odd
    P1 = 8 * 3 * blockSize ** 2
    P2 = 32 * 3 * blockSize ** 2
    disp12MaxDiff = 1
    preFilterCap = 63
    uniquenessRatio = 10
    speckleWindowSize = 100
    speckleRange = 32
    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                   numDisparities=numDisparities,
                                   blockSize=blockSize,
                                   P1=P1,
                                   P2=P2,
                                   disp12MaxDiff=disp12MaxDiff,
                                   preFilterCap=preFilterCap,
                                   uniquenessRatio=uniquenessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.uint8(disparity)

# Function to compute disparity using custom stereo matching
def custom_stereo_matching(imgL, imgR, numDisparities, blockSize):
    rows, cols = imgL.shape
    disparity_map = np.zeros((rows, cols), np.uint8)
    half_block = blockSize // 2
    for y in range(half_block, rows - half_block):
        for x in range(half_block, cols - half_block):
            best_offset = 0
            min_ssd = float('inf')
            for offset in range(numDisparities):
                ssd = 0
                for v in range(-half_block, half_block + 1):
                    for u in range(-half_block, half_block + 1):
                        left_pixel = int(imgL[y + v, x + u])
                        right_pixel = int(imgR[y + v, x + u - offset]) if (x + u - offset) >= 0 else 0
                        ssd += (left_pixel - right_pixel) ** 2
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = offset
            disparity_map[y, x] = best_offset * (255 // numDisparities)
    return disparity_map

# Compute disparity maps
disparity_bm = compute_disparity_bm(imgL, imgR)
disparity_sgbm = compute_disparity_sgbm(imgL, imgR)
disparity_custom = custom_stereo_matching(imgL, imgR, numDisparities=16*4, blockSize=17)

# Function to compute error map
def compute_error_map(disparity, ground_truth):
    error_map = np.abs(disparity.astype(np.float32) - ground_truth)
    return error_map

# Compute error maps
error_bm = compute_error_map(disparity_bm, ground_truth)
error_sgbm = compute_error_map(disparity_sgbm, ground_truth)
error_custom = compute_error_map(disparity_custom, ground_truth)

# Display the error maps
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(error_bm, cmap='hot')
plt.title('Error Map - StereoBM')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(error_sgbm, cmap='hot')
plt.title('Error Map - StereoSGBM')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(error_custom, cmap='hot')
plt.title('Error Map - Custom Stereo Matching')
plt.colorbar()

plt.show()