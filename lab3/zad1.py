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

# Create StereoBM object
numDisparities = 16 * 4 # Must be divisible by 16
blockSize = 21  # Must be odd
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

# Compute the disparity map
disparity = stereo.compute(imgL, imgR)

# Normalize the disparity map for visualization
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Display the disparity map
plt.imshow(disparity, cmap='gray')
plt.title('Disparity Map')
plt.show()