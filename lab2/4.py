import cv2
import numpy as np
import time
import json
import os
from tabulate import tabulate

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

def compare_interpolation_methods(image_path):
    """Porównuje różne metody interpolacji."""
    # Wczytaj obraz
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    
    # Wczytaj dane kalibracyjne
    mtx, dist, _, _, _, _ = load_calibration_data()
    
    # Przygotuj mapowanie dla undistortion
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    
    # Metody interpolacji do porównania
    interpolation_methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }
    
    results = []
    
    # Testuj każdą metodę
    for method_name, method in interpolation_methods.items():
        # Mierz czas
        start_time = time.time()
        
        # Wykonaj remap 10 razy dla dokładniejszego pomiaru
        for _ in range(10):
            dst = cv2.remap(img, mapx, mapy, method)
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Zapisz wynik
        output_filename = f'remap_{method_name.lower()}.png'
        cv2.imwrite(output_filename, dst)
        
        results.append([method_name, f"{avg_time*1000:.2f} ms", output_filename])
    
    return results

def main():
    print("Porównanie metod interpolacji w funkcji remap()")
    print("=" * 50)
    
    try:
        # Użyj pierwszego obrazu z lewej kamery do testu
        image_path = 'lab2/s1/left_230.png'
        results = compare_interpolation_methods(image_path)
        
        # Wyświetl wyniki w formie tabeli
        headers = ["Metoda interpolacji", "Średni czas wykonania", "Plik wynikowy"]
        print("\nWyniki porównania:")
        print(tabulate(results, headers=headers, tablefmt="grid"))
        
        
    except Exception as e:
        print(f"Wystąpił błąd: {str(e)}")

if __name__ == "__main__":
    main() 