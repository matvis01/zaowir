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

def main():
    # Create output directory
    os.makedirs('lab4_output', exist_ok=True)
    
    # Read input data
    disparity = read_pfm('lab4/lab4_materialy/Z1/disp0.pfm')
    baseline, focal_length = read_calib('lab4/lab4_materialy/Z1/calib.txt')
    
    # Convert disparity to depth
    mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[mask] = (baseline * focal_length) / disparity[mask]
    
    # Normalize to 8-bit
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    
    # Save result
    cv2.imwrite('lab4_output/zad1_depth.png', depth_norm)

if __name__ == "__main__":
    main()