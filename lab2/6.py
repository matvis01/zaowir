import cv2
import numpy as np
import json
import os

def load_calibration_data():
    """Wczytuje dane kalibracyjne z pliku JSON."""
    stereo_calibration_file = 'stereo_calibration_data.json'
    
    if not os.path.exists(stereo_calibration_file):
        raise FileNotFoundError("Brak pliku z danymi kalibracji stereo.")
        
    with open(stereo_calibration_file, 'r') as file:
        data = json.load(file)
        
    return (np.array(data["camera_matrix_left"]), 
            np.array(data["distortion_coefficients_left"]),
            np.array(data["camera_matrix_right"]), 
            np.array(data["distortion_coefficients_right"]),
            np.array(data["rotation_matrix"]),
            np.array(data["translation_vector"]))

def draw_epipolar_lines(img_left, img_right, lines, pts1, pts2):
    """Rysuje linie epipolarne i punkty na obrazach."""
    h, w = img_left.shape[:2]
    
    # Stwórz kopie obrazów do rysowania
    img1_lines = img_left.copy()
    img2_lines = img_right.copy()
    
    # Rysuj linie epipolarne i punkty
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
        
        # Rysuj linie
        cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        # Rysuj punkty
        cv2.circle(img1_lines, tuple(map(int, pt1)), 5, color, -1)
        cv2.circle(img2_lines, tuple(map(int, pt2)), 5, color, -1)
    
    return img1_lines, img2_lines

def draw_valid_area(img, roi):
    """Rysuje obwiednię użytecznego obszaru bez zniekształceń."""
    x, y, w, h = roi
    img_with_roi = img.copy()
    cv2.rectangle(img_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_with_roi

def process_stereo_pair(left_image_path, right_image_path, output_dir='output'):
    """Przetwarza parę obrazów stereo."""
    # Utwórz katalog wyjściowy jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)
    
    # Wczytaj obrazy
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        raise ValueError("Nie można wczytać obrazów")
    
    # Wczytaj dane kalibracyjne
    mtx_left, dist_left, mtx_right, dist_right, R, T = load_calibration_data()
    
    # Oblicz macierz fundamentalną
    F, _ = cv2.findFundamentalMat(np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 1]]))
    
    # Przygotuj parametry rektyfikacji stereo
    h, w = img_left.shape[:2]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_left, dist_left,
        mtx_right, dist_right,
        (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1
    )
    
    # Oblicz mapy rektyfikacji
    mapL1, mapL2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, (w, h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, (w, h), cv2.CV_32FC1)
    
    # Rektyfikuj obrazy
    rect_left = cv2.remap(img_left, mapL1, mapL2, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, mapR1, mapR2, cv2.INTER_LINEAR)
    
    # Narysuj obwiednie użytecznego obszaru
    rect_left_roi = draw_valid_area(rect_left, roi1)
    rect_right_roi = draw_valid_area(rect_right, roi2)
    
    # Rysuj linie epipolarne
    for y in range(0, h, 30):  # Rysuj co 30 pikseli
        cv2.line(rect_left_roi, (0, y), (w, y), (0, 0, 255), 1)
        cv2.line(rect_right_roi, (0, y), (w, y), (0, 0, 255), 1)
    
    # Zapisz wyniki
    base_name_left = os.path.splitext(os.path.basename(left_image_path))[0]
    base_name_right = os.path.splitext(os.path.basename(right_image_path))[0]
    
    cv2.imwrite(os.path.join(output_dir, f'{base_name_left}_rect_epipolar.png'), rect_left_roi)
    cv2.imwrite(os.path.join(output_dir, f'{base_name_right}_rect_epipolar.png'), rect_right_roi)
    cv2.imwrite(os.path.join(output_dir, f'{base_name_left}_rect.png'), rect_left)
    cv2.imwrite(os.path.join(output_dir, f'{base_name_right}_rect.png'), rect_right)
    
    return rect_left_roi, rect_right_roi

def main():
    print("Wizualizacja linii epipolarnych i eksport zrektyfikowanych obrazów")
    print("=" * 50)
    
    try:
        # Przetwórz parę obrazów
        left_image = 'lab2/s1/left_230.png'
        right_image = 'lab2/s1/right_230.png'
        
        rect_left, rect_right = process_stereo_pair(left_image, right_image)
        
        # Wyświetl wyniki
        cv2.imshow('Left Image with Epipolar Lines', rect_left)
        cv2.imshow('Right Image with Epipolar Lines', rect_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\nWyniki zostały zapisane w katalogu 'output':")
        print("1. Obrazy z liniami epipolarnymi i ROI (*_rect_epipolar.png)")
        print("2. Zrektyfikowane obrazy bez dodatkowych oznaczeń (*_rect.png)")
        
    except Exception as e:
        print(f"Wystąpił błąd: {str(e)}")

if __name__ == "__main__":
    main() 