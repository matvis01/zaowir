import cv2
import numpy as np
import matplotlib.pyplot as plt

def census_transform(img, block_size):
    height, width = img.shape
    half_block_size = block_size // 2
    census = np.zeros((height, width), dtype=np.uint64)

    for y in range(half_block_size, height - half_block_size):
        for x in range(half_block_size, width - half_block_size):
            center_pixel = img[y, x]
            bit_string = 0
            bit_index = 0

            for v in range(-half_block_size, half_block_size + 1):
                for u in range(-half_block_size, half_block_size + 1):
                    if v == 0 and u == 0:
                        continue
                    bit_string <<= 1
                    if img[y + v, x + u] < center_pixel:
                        bit_string |= 1
                    bit_index += 1

            census[y, x] = bit_string

    return census

def compute_disparity(img_left, img_right, block_size, max_disparity):
    height, width = img_left.shape
    disparity_map = np.zeros((height, width), np.uint8)

    half_block_size = block_size // 2

    # Apply Census Transform to both images
    census_left = census_transform(img_left, block_size)
    census_right = census_transform(img_right, block_size)

    for y in range(half_block_size, height - half_block_size):
        for x in range(half_block_size, width - half_block_size):
            best_offset = 0
            min_hamming_distance = float('inf')

            for offset in range(max_disparity):
                if x - offset - half_block_size < 0:
                    continue

                hamming_distance = 0
                for v in range(-half_block_size, half_block_size + 1):
                    for u in range(-half_block_size, half_block_size + 1):                    # Apply Census Transform to both images
                        census_left = census_transform(img_left, block_size)
                        census_right = census_transform(img_right, block_size)
                    
                        for y in range(half_block_size, height - half_block_size):
                            for x in range(half_block_size, width - half_block_size):
                                optimal_offset = 0
                                lowest_hamming_distance = float('inf')
                    
                                for offset in range(max_disparity):
                                    if x - offset - half_block_size < 0:
                                        continue
                    
                                    current_hamming_distance = 0
                                    for v in range(-half_block_size, half_block_size + 1):
                                        for u in range(-half_block_size, half_block_size + 1):
                                            left_pixel_value = census_left[y + v, x + u]
                                            right_pixel_value = census_right[y + v, x + u - offset]
                                            current_hamming_distance += bin(left_pixel_value ^ right_pixel_value).count('1')
                    
                                    if current_hamming_distance < lowest_hamming_distance:
                                        lowest_hamming_distance = current_hamming_distance
                                        optimal_offset = offset
                    
                                disparity_map[y, x] = optimal_offset * (255 // max_disparity)
                    
                        return disparity_map
                    
                    img_left = cv2.imread('lab3/cones/im2.png', cv2.IMREAD_GRAYSCALE)
                    img_right = cv2.imread('lab3/cones/im6.png', cv2.IMREAD_GRAYSCALE)
                    
                    if img_left is None or img_right is None:
                        print("Error: Could not load images.")
                        exit()
                        left_pixel = census_left[y + v, x + u]
                        right_pixel = census_right[y + v, x + u - offset]
                        hamming_distance += bin(left_pixel ^ right_pixel).count('1')

                if hamming_distance < min_hamming_distance:
                    min_hamming_distance = hamming_distance
                    best_offset = offset

            disparity_map[y, x] = best_offset * (255 // max_disparity)

    return disparity_map

img_left = cv2.imread('lab3/cones/im2.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('lab3/cones/im6.png', cv2.IMREAD_GRAYSCALE)

if img_left is None or img_right is None:
    print("Error: Could not load images.")
    exit()

block_size = 7
max_disparity = 64

disparity_map = compute_disparity(img_left, img_right, block_size, max_disparity)


plt.imshow(disparity_map, cmap='gray')
plt.title('Mapa dysparycji')
plt.show()