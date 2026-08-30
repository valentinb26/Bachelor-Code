# Calibrate and Validate Homography
import cv2
import numpy as np
from config.config import USE_CALIBRATION, HOMOGRAPHY_PATH

USE_CALIBRATION = False

# ---------------------------------------------------------------------------
# 1. ENTZERRUNGSDATEN LADEN & SYSTEMEINSTELLUNGEN
# ---------------------------------------------------------------------------
try:
    data = np.load("camera_distortion_matrixA.npz")
    mtx = data['mtx']
    dist = data['dist']
    USE_CALIBRATION = True
    print("Kameramatrix und Verzerrungskoeffizienten erfolgreich geladen.")
except FileNotFoundError:
    print("Fehler: 'camera_distortion_matrix.npz' wurde nicht gefunden.")
    #exit(1)

camera_pixels = []

robot_coordinates = np.array([
    [-22.8,  202.6], #1
    [89.8,   277.0], #2
    [156.4,  174.8], #3
    [258.8,  102.2], #4
    [300.9,  -10.0], #5
    [174.1,  -95.5], #6
    [55.3,  -194.8], #7
    [-67.7, -284.0]  #8
], dtype=np.float32)

# ---------------------------------------------------------------------------
# 2. BERECHNUNG UND VALIDIERUNG
# ---------------------------------------------------------------------------
def calculate_and_validate():
    """Berechnet die Homographie-Matrix und validiert diese direkt."""
    global camera_pixels, robot_coordinates
    
    camera_pixels_np = np.array(camera_pixels, dtype=np.float32)
    
    # Homographie berechnen
    homography_matrix, _ = cv2.findHomography(camera_pixels_np, robot_coordinates)
    
    print("\nTransformationsmatrix erfolgreich berechnet!")
    print(homography_matrix)
    
    # Speichern der Matrix
    np.save(HOMOGRAPHY_PATH, homography_matrix)
    print("\nMatrix als 'homography_matrix.npy' gespeichert.")

    # Validierung durchführen
    print("\n=== HOMOGRAPHIE-VALIDIERUNG ===\n")
    print("Transformiere alle 8 Kalibrierungs-Pixel zurück zu Roboter-Koordinaten:\n")

    errors = []
    for i, (pixel, robot) in enumerate(zip(camera_pixels_np, robot_coordinates)):
        calculated = cv2.perspectiveTransform(
            np.array([[[pixel[0], pixel[1]]]], dtype=np.float32),
            homography_matrix
        )[0][0]
        
        error_x = abs(calculated[0] - robot[0])
        error_y = abs(calculated[1] - robot[1])
        error_total = np.sqrt(error_x**2 + error_y**2)
        
        errors.append(error_total)
        
        print(f"Punkt {i+1}:")
        print(f"  Pixel:     ({int(pixel[0])}, {int(pixel[1])})")
        print(f"  Erwartet:  ({robot[0]:.1f}, {robot[1]:.1f})")
        print(f"  Berechnet: ({calculated[0]:.2f}, {calculated[1]:.2f})")
        print(f"  Fehler:    {error_total:.2f}mm (ΔX={error_x:.2f}, ΔY={error_y:.2f})\n")

    print(f"--- STATISTIK ---")
    print(f"Durchschnittlicher Fehler: {np.mean(errors):.2f}mm")
    print(f"Max-Fehler:                {np.max(errors):.2f}mm")
    print(f"Min-Fehler:                {np.min(errors):.2f}mm")
    print("\nDu kannst das Fenster jetzt schließen (Beliebige Taste drücken).")

# ---------------------------------------------------------------------------
# 3. MAUS-CALLBACK
# ---------------------------------------------------------------------------
def click_event(event, x, y, flags, params):
    """Maus-Callback-Funktion zum Sammeln der Pixelpunkte."""
    global camera_pixels, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        camera_pixels.append([x, y])
        punkt_nummer = len(camera_pixels)
        
        print(f"Punkt {punkt_nummer} registriert: Pixel (X={x}, Y={y}) -> Zugeordnet zu Roboter: {robot_coordinates[punkt_nummer-1]}")
        
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image, str(punkt_nummer), (x + 10, y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Kalibrierungs-Fenster", image)
        
        if len(camera_pixels) == 8:
            print("\n--- Alle 8 Punkte gesammelt! ---")
            calculate_and_validate()

# ---------------------------------------------------------------------------
# 4. HAUPTPROGRAMM (LIVE-BILD & OPTISCHE ENTZERRUNG)
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit(1)

print("Ziehe das Fenster groß und drücke 's' um ein Standbild für die Kalibrierung einzufrieren...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Kamerabildes.")
        break

    # Entzerren des Live-Frames
    if USE_CALIBRATION:
        frame = cv2.undistort(frame, mtx, dist, None, mtx)
    else:
        frame = frame.copy()
    cv2.imshow("Live-Vorschau (Drücke 's' für Standbild)", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        image = frame.copy()
        cap.release()
        cv2.destroyAllWindows()
        break

print("\nANLEITUNG:")
print("Bitte klicke nacheinander auf die 8 Markierungen im entzerrten Bild.")
print("Achte exakt auf die Reihenfolge, die oben bei 'robot_coordinates' definiert ist!\n")

cv2.namedWindow("Kalibrierungs-Fenster")
cv2.setMouseCallback("Kalibrierungs-Fenster", click_event)

cv2.imshow("Kalibrierungs-Fenster", image)
cv2.waitKey(0)
cv2.destroyAllWindows()