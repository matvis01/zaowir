import numpy as np
import cv2
import os
from PIL import Image

def depth_to_disparity(depth_map, baseline, f):
    """Convert depth map to disparity map."""
    disparity_map = np.zeros_like(depth_map, dtype=np.float32)
    mask = depth_map > 0
    disparity_map[mask] = (baseline * f) / depth_map[mask]
    return disparity_map

def load_depth_map(filepath, max_depth=1000):
    """Load depth map saved as 24-bit image (uint24)."""
    depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"Cannot open file {filepath}")
    
    # Convert to single channel if needed
    if len(depth_img.shape) > 2:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    
    # Convert 24-bit image to depth in meters
    max_value = 2**24 - 1  # Maximum value for 24-bit depth
    depth_map = depth_img.astype(np.float32) / max_value * max_depth  # Scale to 1000m
    return depth_map

def main():
    # Create output directory
    os.makedirs("lab4_output", exist_ok=True)
    
    # Parameters
    h_fov = 60  # degrees
    baseline = 0.1  # meters
    max_depth = 1000  # meters
    
    # Load depth map
    depth_map = load_depth_map("lab4/lab4_materialy/Z4/depth.png", max_depth)
    
    # Get image dimensions (now guaranteed 2D)
    h, w = depth_map.shape
    
    # Calculate focal length from HFoV
    f = (w / 2) / np.tan(np.radians(h_fov / 2))
    
    # Calculate disparity map
    disparity_map = depth_to_disparity(depth_map, baseline, f)
    
    # Print debug info
    print(f"Disparity range before norm: {np.min(disparity_map):.2f} - {np.max(disparity_map):.2f}")
    
    # Clip extreme values for better visualization
    disparity_map = np.clip(disparity_map, 0, np.percentile(disparity_map, 95))
    
    # Normalize to full 8-bit range for better contrast
    normalized_disparity = ((disparity_map - np.min(disparity_map)) / 
                          (np.max(disparity_map) - np.min(disparity_map)) * 255)
    
    # Convert to uint8
    output_disparity = np.uint8(normalized_disparity)
    
    # Optional: Apply histogram equalization for better visibility
    output_disparity = cv2.equalizeHist(output_disparity)
    
    # Save result
    cv2.imwrite("lab4_output/zad4_disparity_map.png", output_disparity)
    
    print(f"Disparity range after norm: {np.min(output_disparity)} - {np.max(output_disparity)}")

if __name__ == "__main__":
    main()