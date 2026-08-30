"""
validate_homography.py

Validiert die bestehende Homographie-Matrix auf unabhaengigen Testpunkten,
die NICHT zur urspruenglichen Kalibrierung verwendet wurden.

Ablauf:
    1. TEST_POINTS unten mit deinen echten Roboterkoordinaten befuellen
       (die du bereits per DobotLab/get_pose() aufgeschrieben hast).
    2. Skript im Modus 'capture' starten -> Kamera oeffnet sich, du klickst
       Punkt fuer Punkt an. Ergebnis wird in captured_points.json gespeichert.
    3. Skript im Modus 'calc' starten -> laedt captured_points.json + die
       bestehende Homographie, berechnet Fehler, druckt Tabelle + Statistik.

Nutzung:
    python validate_homography.py capture
    python validate_homography.py calc

Wichtig: Die Kamera darf sich seit der urspruenglichen Kalibrierung nicht
bewegt haben (Position, Neigung, Fokus). Sonst ist die Validierung wertlos.
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# KONFIGURATION -- HIER ANPASSEN
# ============================================================

# Pfad zur bestehenden Homographie-Matrix (NICHT neu berechnen, nur laden)
HOMOGRAPHY_PATH = Path("data/homography_matrix.npy")

# Wohin die geklickten Pixelkoordinaten gespeichert werden
CAPTURE_OUTPUT_PATH = Path("captured_points.json")

# Kamera-Index (0 = Standard-Webcam, ggf. anpassen falls mehrere Kameras)
CAMERA_INDEX = 0

# Deine 10 Testpunkte MIT den bereits notierten echten Roboterkoordinaten.
# label: frei waehlbar (z.B. P1, P2, ...)
# region: "rand" oder "mitte" -- fuer die spaetere Auswertung
# real_x, real_y: die von dir per DobotLab/get_pose() abgelesenen mm-Werte
#
# WICHTIG: Reihenfolge hier = Reihenfolge, in der du im Capture-Modus klickst.
TEST_POINTS = [
    {"label": "P1",  "region": "rechts/rand",  "real_x": -53.4, "real_y": 326.4},
    {"label": "P2",  "region": "rechts/mitte",  "real_x": 84.7, "real_y": 191.5},
    {"label": "P3",  "region": "rechts/rand",  "real_x": 250.0, "real_y": 218.9},
    {"label": "P4",  "region": "mitte/rand",  "real_x": 329.3, "real_y": 29.4},
    {"label": "P5",  "region": "mitte/mitte",  "real_x": 197.4, "real_y": 15.0},
    {"label": "P6",  "region": "mitte/mitte", "real_x": 242.6, "real_y": -136.5},
    {"label": "P7",  "region": "links/mitte", "real_x": 124.3, "real_y": -152.5},
    {"label": "P8",  "region": "links/mitte", "real_x": 109.8, "real_y": -240.4},
    {"label": "P9",  "region": "links/rand", "real_x": -10.4, "real_y": -314.0},
    {"label": "P10", "region": "links/rand", "real_x": -86.9, "real_y": -220.8},
]

# Aus deiner Trainingsdaten-Messung (main.tex, tab:homographie_validierung),
# fuer den direkten Vergleich im Ergebnisprint.
TRAINING_MEAN_ERROR_MM = 2.08


# ============================================================
# KERNFUNKTION (identisch zu utils.py, damit dieses Skript
# unabhaengig von deiner Projektstruktur lauffaehig ist)
# ============================================================

def pixel_to_robot(pixel_x: int, pixel_y: int, homography_matrix):
    """Transformiert Pixel-Koordinaten mithilfe der Homographie-Matrix in Roboter-Millimeter."""
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return float(transformed_point[0][0][0]), float(transformed_point[0][0][1])


# ============================================================
# MODUS 1: CAPTURE -- Klicks sammeln
# ============================================================

class ClickCapturer:
    def __init__(self):
        self.clicked_point = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)


def run_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("FEHLER: Kamera konnte nicht geoeffnet werden.")
        sys.exit(1)

    window_name = "Homographie-Validierung: Testpunkte anklicken"
    cv2.namedWindow(window_name)
    capturer = ClickCapturer()
    cv2.setMouseCallback(window_name, capturer.on_mouse)

    results = []
    frozen_frame = None
    idx = 0

    print(f"\n{len(TEST_POINTS)} Testpunkte zu erfassen.")
    print("Ablauf pro Punkt: Roboter/Marker manuell zum naechsten Punkt bewegen,")
    print("Kamera live -> 's' druecken zum Einfrieren -> Punkt anklicken -> 'n' zum Bestaetigen.")
    print("'r' zum erneuten Einfrieren falls verklickt. 'q' zum Abbrechen.\n")

    while idx < len(TEST_POINTS):
        point_info = TEST_POINTS[idx]
        print(f"--> Aktueller Punkt: {point_info['label']} ({point_info['region']})")

        # Live-Phase bis 's' gedrueckt wird
        frozen_frame = None
        capturer.clicked_point = None

        while frozen_frame is None:
            ret, frame = cap.read()
            if not ret:
                print("FEHLER: Kein Kamerabild erhalten.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)

            display = frame.copy()
            cv2.putText(display, f"Punkt {point_info['label']} - 's' zum Einfrieren",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                frozen_frame = frame.copy()
            elif key == ord('q'):
                print("Abgebrochen.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        # Eingefrorene Phase: warten auf Klick + Bestaetigung
        while True:
            display = frozen_frame.copy()
            if capturer.clicked_point:
                cv2.circle(display, capturer.clicked_point, 6, (0, 0, 255), -1)
                cv2.putText(display, f"Pixel: {capturer.clicked_point} - 'n' bestaetigen, 'r' neu",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Punkt anklicken", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n') and capturer.clicked_point:
                results.append({
                    "label": point_info["label"],
                    "region": point_info["region"],
                    "real_x": point_info["real_x"],
                    "real_y": point_info["real_y"],
                    "pixel_x": capturer.clicked_point[0],
                    "pixel_y": capturer.clicked_point[1],
                })
                print(f"    Gespeichert: Pixel {capturer.clicked_point}\n")
                idx += 1
                break
            elif key == ord('r'):
                frozen_frame = None
                capturer.clicked_point = None
                break
            elif key == ord('q'):
                print("Abgebrochen.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    with open(CAPTURE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Alle {len(results)} Punkte erfasst und in '{CAPTURE_OUTPUT_PATH}' gespeichert.")
    print("Weiter mit: python validate_homography.py calc")


# ============================================================
# MODUS 2: CALC -- Fehler berechnen und auswerten
# ============================================================

def run_calc():
    if not HOMOGRAPHY_PATH.exists():
        print(f"FEHLER: Homographie-Matrix nicht gefunden unter '{HOMOGRAPHY_PATH}'.")
        sys.exit(1)
    if not CAPTURE_OUTPUT_PATH.exists():
        print(f"FEHLER: '{CAPTURE_OUTPUT_PATH}' nicht gefunden. Erst 'capture' ausfuehren.")
        sys.exit(1)

    homography_matrix = np.load(HOMOGRAPHY_PATH)

    with open(CAPTURE_OUTPUT_PATH, "r", encoding="utf-8") as f:
        captured = json.load(f)

    rows = []
    for entry in captured:
        calc_x, calc_y = pixel_to_robot(entry["pixel_x"], entry["pixel_y"], homography_matrix)
        error = float(np.hypot(calc_x - entry["real_x"], calc_y - entry["real_y"]))
        rows.append({
            **entry,
            "calc_x": calc_x,
            "calc_y": calc_y,
            "error_mm": error,
        })

    # ---- Tabelle drucken ----
    print("\n" + "=" * 100)
    print(f"{'Punkt':<6}{'Region':<8}{'Pixel':<16}{'Erwartet (mm)':<22}{'Berechnet (mm)':<22}{'Fehler (mm)':<10}")
    print("-" * 100)
    for r in rows:
        pixel_str = f"({r['pixel_x']}, {r['pixel_y']})"
        erwartet_str = f"({r['real_x']:.1f}, {r['real_y']:.1f})"
        berechnet_str = f"({r['calc_x']:.2f}, {r['calc_y']:.2f})"
        print(f"{r['label']:<6}{r['region']:<8}{pixel_str:<16}{erwartet_str:<22}{berechnet_str:<22}{r['error_mm']:.2f}")
    print("=" * 100)

    errors = [r["error_mm"] for r in rows]
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    min_error = float(np.min(errors))

    print(f"\nGesamt (n={len(errors)}):")
    print(f"  Mittelwert: {mean_error:.2f} mm")
    print(f"  Max:        {max_error:.2f} mm")
    print(f"  Min:        {min_error:.2f} mm")
    print(f"\n  Vergleich Trainingsdaten (main.tex): {TRAINING_MEAN_ERROR_MM:.2f} mm")
    diff = mean_error - TRAINING_MEAN_ERROR_MM
    print(f"  Differenz Test- vs. Trainingsfehler: {diff:+.2f} mm")

    # ---- Rand vs. Mitte, nur falls beide Gruppen vorhanden sind ----
    rand_errors = [r["error_mm"] for r in rows if r["region"] == "rand"]
    mitte_errors = [r["error_mm"] for r in rows if r["region"] == "mitte"]
    if rand_errors and mitte_errors:
        print(f"\n  Rand  (n={len(rand_errors)}):  Mittelwert {np.mean(rand_errors):.2f} mm")
        print(f"  Mitte (n={len(mitte_errors)}):  Mittelwert {np.mean(mitte_errors):.2f} mm")

    # ---- LaTeX-Tabellenzeilen zum direkten Einfuegen ----
    print("\n--- LaTeX-Tabellenzeilen (zum Kopieren) ---")
    for r in rows:
        print(f"{r['label']} & ({r['pixel_x']}, {r['pixel_y']}) & "
              f"({r['real_x']:.1f}, {r['real_y']:.1f}) & "
              f"({r['calc_x']:.2f}, {r['calc_y']:.2f}) & {r['error_mm']:.2f} \\\\")

    # ---- CSV speichern ----
    csv_path = Path("validation_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,region,pixel_x,pixel_y,real_x,real_y,calc_x,calc_y,error_mm\n")
        for r in rows:
            f.write(f"{r['label']},{r['region']},{r['pixel_x']},{r['pixel_y']},"
                    f"{r['real_x']},{r['real_y']},{r['calc_x']:.4f},{r['calc_y']:.4f},{r['error_mm']:.4f}\n")
    print(f"\nErgebnisse zusaetzlich gespeichert in '{csv_path}'.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("capture", "calc"):
        print("Nutzung: python validate_homography.py [capture|calc]")
        sys.exit(1)

    if sys.argv[1] == "capture":
        run_capture()
    else:
        run_calc()