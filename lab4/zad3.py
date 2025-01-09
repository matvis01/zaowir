import numpy as np
import cv2
import os

def read_pfm(file_path):
    with open(file_path, 'rb') as file:
        header = file.readline().decode().rstrip()
        if header != 'Pf':
            raise ValueError('Not a PFM file')
        
        dims = file.readline().decode().rstrip()
        width, height = map(int, dims.split())
        
        scale = float(file.readline().decode().rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        data = np.fromfile(file, endian + 'f')
        data = data.reshape(height, width)
        
        data = np.flipud(data)
            
        return data

def read_calib(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('baseline'):
                baseline = float(line.split('=')[1])
            elif line.startswith('cam0'):
                # Extract focal length from first element of camera matrix
                focal_length = float(line.split('[')[1].split()[0])
    return baseline, focal_length

def depth_to_rgb(depth, max_depth=10000):
    """Convert depth map to 24-bit RGB image"""
    normalized = depth / max_depth
    
    values = normalized * (256 * 256 * 256 - 1)
    
    B = np.floor(values / (256 * 256)).astype(np.uint8)
    G = np.floor((values % (256 * 256)) / 256).astype(np.uint8)
    R = np.floor(values % 256).astype(np.uint8)
    
    return cv2.merge([R, G, B])

def rgb_to_depth(rgb_image, max_depth=1000):
    """Convert 24-bit RGB image back to depth map"""
    R, G, B = cv2.split(rgb_image)
    
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)
    
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    
    return normalized * max_depth

def main():
    os.makedirs('lab4_output', exist_ok=True)
    
    disparity = read_pfm('lab4/lab4_materialy/Z1/disp0.pfm')
    baseline, focal_length = read_calib('lab4/lab4_materialy/Z1/calib.txt')
    
    mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[mask] = (baseline * focal_length) / disparity[mask]
    
    depth_rgb = depth_to_rgb(depth)
    
    cv2.imwrite('lab4_output/zad3_depth_rgb.png', depth_rgb)
    
    rgb_loaded = cv2.imread('lab4_output/zad3_depth_rgb.png')
    depth_recovered = rgb_to_depth(rgb_loaded)
    
    depth_recovered_norm = cv2.normalize(depth_recovered, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('lab4_output/zad3_depth_recovered.png', depth_recovered_norm.astype(np.uint8))

if __name__ == "__main__":
    main()