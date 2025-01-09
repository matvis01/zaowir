import numpy as np
import cv2

def read_calibration(calib_path):
    params = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('cam0='):
                matrix_str = line.replace('cam0=[', '').replace(']', '')
                rows = [row.strip() for row in matrix_str.split(';')]
                matrix = []
                for row in rows:
                    matrix.append([float(x) for x in row.split()])
                params['focal_length'] = matrix[0][0] 
            elif line.startswith('baseline='):
                params['baseline'] = float(line.split('=')[1])
            elif line.startswith('ndisp='):
                params['num_disparities'] = int(line.split('=')[1])
    return params

def compute_disparity(left_img, right_img, params):
    window_size = 3
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=params['num_disparities'],
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    return disparity

def disparity_to_depth(disparity, params):
    mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[mask] = (params['baseline'] * params['focal_length']) / disparity[mask]
    return depth

def normalize_depth(depth):
    depth_valid = depth[depth > 0]
    if len(depth_valid) == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    
    min_depth = np.percentile(depth_valid, 5)
    max_depth = np.percentile(depth_valid, 95)
    normalized = np.clip(depth, min_depth, max_depth)
    normalized = ((normalized - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    return normalized

def main():
    params = read_calibration('lab4/lab4_materialy/Z1/calib.txt')
    
    left_img = cv2.imread('lab4/lab4_materialy/Z1/im0.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('lab4/lab4_materialy/Z1/im1.png', cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        raise ValueError("Failed to load images")
    
    disparity = compute_disparity(left_img, right_img, params)
    
    depth = disparity_to_depth(disparity, params)
    
    depth_normalized = normalize_depth(depth)
    cv2.imwrite('lab4_output/zad2_depth.png', depth_normalized)

if __name__ == '__main__':
    main()