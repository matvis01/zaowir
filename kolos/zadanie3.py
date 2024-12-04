import cv2 as cv
import numpy as np
import json

def calculate_fov(cameraMatrix: np.ndarray, imageSize: tuple[float, float]):
    fx = cameraMatrix[0, 0]  # Focal length in x-axis
    fy = cameraMatrix[1, 1]  # Focal length in y-axis
    width, height = imageSize

    fov_horizontal = 2 * np.arctan2(width, (2 * fx)) * (180 / np.pi)  # Convert radians to degrees
    fov_vertical = 2 * np.arctan2(height, (2 * fy)) * (180 / np.pi)  # Convert radians to degrees

    return fov_horizontal, fov_vertical

# Load stereo calibration data
with open('kolos/stereo_calibration_data.json', 'r') as f:
    stereo_calibration_data = json.load(f)

# Extract camera matrices
CM1 = np.array(stereo_calibration_data["CameraMatrix1"])
CM2 = np.array(stereo_calibration_data["CameraMatrix2"])

# Assuming the image size is the same for both cameras
image_size = (640, 480)  # Replace with actual image size if different

# Calculate HFov for both cameras
hfov_left, vfov_left = calculate_fov(CM1, image_size)
hfov_right, vfov_right = calculate_fov(CM2, image_size)

print(f"Horizontal Field of View (HFov) for Left Camera: {hfov_left} degrees")
print(f"Horizontal Field of View (HFov) for Right Camera: {hfov_right} degrees")

# Save HFov data to a JSON file
hfov_data = {
    "LeftCamera": {
        "HFov": hfov_left,
        "VFov": vfov_left
    },
    "RightCamera": {
        "HFov": hfov_right,
        "VFov": vfov_right
    }
}

with open('kolos/hfov_data.json', 'w') as f:
    json.dump(hfov_data, f, indent=4)

print("HFov data saved to kolos/hfov_data.json")