import numpy as np
import json
import os

def calculate_baseline():
    # Ścieżka do pliku z danymi kalibracji stereo
    stereo_calibration_file = 'stereo_calibration_data.json'
    
    # Sprawdź czy plik z kalibracją istnieje
    if not os.path.exists(stereo_calibration_file):
        print("Błąd: Brak pliku z danymi kalibracji stereo.")
        print("Najpierw wykonaj kalibrację stereo używając stereo.py")
        return None
    
    try:
        # Wczytaj dane kalibracji
        with open(stereo_calibration_file, 'r') as file:
            stereo_data = json.load(file)
        
        # Pobierz wektor translacji
        translation_vector = np.array(stereo_data["translation_vector"])
        
        # Oblicz baseline jako normę wektora translacji
        baseline = np.linalg.norm(translation_vector)
        
        # Wyświetl szczegółowe informacje
        print("\nSzczegóły obliczenia baseline:")
        print(f"Wektor translacji T: {translation_vector.flatten()}")
        print(f"Baseline (norma wektora T): {baseline:.6f} jednostek")
        
        # Wyświetl komponenty baseline
        print("\nKomponenty baseline:")
        print(f"X: {abs(translation_vector[0][0]):.6f}")
        print(f"Y: {abs(translation_vector[1][0]):.6f}")
        print(f"Z: {abs(translation_vector[2][0]):.6f}")
        
        return baseline
        
    except Exception as e:
        print(f"Wystąpił błąd podczas obliczania baseline: {str(e)}")
        return None

if __name__ == "__main__":
    print("Obliczanie odległości bazowej (baseline) między kamerami...")
    baseline = calculate_baseline()
    
