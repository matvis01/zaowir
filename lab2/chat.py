import cv2
import numpy as np
import os
import json

# Definicje rozmiaru szachownicy i kryteriów
chessboard_size = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Funkcja do ładowania punktów kalibracyjnych z poprzedniego zadania
def load_points(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    objpoints = [np.array(objp, dtype=np.float32) for objp in data["object_points"]]
    imgpoints = [np.array(imgp, dtype=np.float32) for imgp in data["image_points"]]
    return objpoints, imgpoints

# Ładowanie punktów dla obu kamer (np. left i right)
objpoints, imgpoints_left = load_points('points_left.json')
_, imgpoints_right = load_points('points_right.json')

# Załóżmy, że macierze i dystorsje dla obu kamer zostały wcześniej obliczone
with open('calibration_data_left.json', 'r') as f:
    left_data = json.load(f)

with open('calibration_data_right.json', 'r') as f:
    right_data = json.load(f)

mtx_left = np.array(left_data["camera_matrix"])
dist_left = np.array(left_data["distortion_coefficients"])
mtx_right = np.array(right_data["camera_matrix"])
dist_right = np.array(right_data["distortion_coefficients"])

# Stereo kalibracja
ret, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    (640, 480),  # Rozdzielczość zdjęć
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Zapis wyników do pliku JSON
stereo_calib_data = {
    "camera_matrix_left": camera_matrix1.tolist(),
    "distortion_coefficients_left": dist_coeffs1.tolist(),
    "camera_matrix_right": camera_matrix2.tolist(),
    "distortion_coefficients_right": dist_coeffs2.tolist(),
    "rotation_matrix": R.tolist(),
    "translation_vector": T.tolist(),
    "essential_matrix": E.tolist(),
    "fundamental_matrix": F.tolist()
}

with open('stereo_calibration_data.json', 'w') as f:
    json.dump(stereo_calib_data, f, indent=4)

print("Stereo calibration complete. Results saved to 'stereo_calibration_data.json'.")