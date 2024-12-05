import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the stereo images
imgL = cv2.imread('lab3/cones/im2.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('lab3/cones/im6.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if imgL is None or imgR is None:
    print("Error: Could not load images.")
    exit()

# Create StereoSGBM object
minDisparity = 0
numDisparities = 16 * 5  # Must be divisible by 16
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

# Compute the disparity map
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Normalize the disparity map for visualization
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Display the disparity map
plt.imshow(disparity, cmap='gray')
plt.title('Disparity Map - StereoSGBM')
plt.show()