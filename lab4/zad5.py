import numpy as np
import cv2
import os

PLY_HEADER = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(filename, points, colors):
    """Write 3D points and colors to PLY file."""
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([points, colors])
    
    with open(filename, 'wb') as f:
        f.write((PLY_HEADER % dict(vert_num=len(vertices))).encode('utf-8'))
        np.savetxt(f, vertices, fmt='%f %f %f %d %d %d ')

def load_depth_map(filepath, max_depth=1000):
    """Load depth map from 24-bit image."""
    depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"Cannot open file {filepath}")
    
    if len(depth_img.shape) > 2:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    
    max_value = 2**24 - 1
    depth_map = depth_img.astype(np.float32) / max_value * max_depth
    return depth_map

def depth_to_points(depth_map, h_fov):
    """Convert depth map to 3D points."""
    h, w = depth_map.shape
    f = (w / 2) / np.tan(np.radians(h_fov / 2))
    
    # Create coordinate grid
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D coordinates
    Z = depth_map
    X = (x_grid - w/2) * Z / f
    Y = (y_grid - h/2) * Z / f
    
    return np.dstack((X, Y, Z))

def main():
    # Parameters
    h_fov = 60  # degrees
    max_distance = 50  # meters
    
    # Load images
    depth_map = load_depth_map("lab4/lab4_materialy/Z4/depth.png")
    color_img = cv2.imread("lab4/lab4_materialy/Z4/left.png")
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    
    # Generate 3D points
    points = depth_to_points(depth_map, h_fov)
    
    # Filter points by distance
    mask = depth_map <= max_distance
    filtered_points = points[mask]
    filtered_colors = color_img[mask]
    
    # Save PLY file
    os.makedirs("lab4_output", exist_ok=True)
    write_ply("lab4_output/points.ply", filtered_points, filtered_colors)

if __name__ == "__main__":
    main()