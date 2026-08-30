import cv2

# Webcam initialisieren (0 ist meist die eingebaute Standard-Kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Webcam konnte nicht geöffnet werden.")
    exit()

counter = 1

print("--- Webcam-Steuerung ---")
print("LEERTASTE  : Bild speichern")
print("'q' oder ESC : Programm beenden")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Empfangen des Kamerabildes.")
        break

    # Live-Bild anzeigen
    cv2.imshow("Webcam (Leertaste = Foto, Q = Beenden)", frame)

    # Auf Tastendruck warten (1 Millisekunde)
    key = cv2.waitKey(1) & 0xFF

    # Leertaste gedrückt (ASCII-Code 32)
    if key == 32:
        #filename = f"bild{counter}.jpg"
        filename = f"/Users/v/Desktop/Coding_Workspaces/ba_code/gdino/src/eval/eval_images/bild{counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Gespeichert: {filename}")
        counter += 1

    # 'q' oder ESC (ASCII-Code 27) zum Beenden
    elif key == ord('q') or key == 27:
        print("Programm beendet.")
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()