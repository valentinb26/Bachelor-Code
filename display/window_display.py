# window_display.py
# Zeigt den Live-Kamera-Feed mit den zuletzt gespeicherten Detections an.
# WICHTIG: run() muss im Main-Thread aufgerufen werden (macOS erlaubt keine
# cv2-Fenster aus Nebenthreads, Windows/Linux sind hier toleranter, aber wir
# halten uns an die strengste Regel).
import cv2
import logging

logger = logging.getLogger(__name__)

class WindowDisplay:
    """Simples Debug-Fenster: aktueller Frame + letzte Bounding Boxes + Mittelpunkt."""

    WINDOW_NAME = "Live Feed - Detections"

    def __init__(self, detector):
        self.detector = detector
        self._stop_requested = False

    def stop(self):
        """Kann aus jedem Thread aufgerufen werden (z. B. aus shutdown())."""
        self._stop_requested = True

    def run(self):
        """
        Blockierende Schleife, MUSS im Main-Thread laufen.
        Endet entweder durch Taste 'q' im Fenster oder durch einen externen stop()-Aufruf.
        """
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        while not self._stop_requested and self.detector.inference_active:
            frame, result, error_msg = self._get_latest_frame_and_result()

            if frame is not None:
                display_frame = self._draw_overlay(frame, result, error_msg)
                cv2.imshow(self.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._stop_requested = True
                break

        cv2.destroyWindow(self.WINDOW_NAME)
        logger.info("Display beendet.")


    def _get_latest_frame_and_result(self):
        with self.detector.frame_lock:
            frame = self.detector.current_frame_bgr
            result = self.detector.current_result
            error_msg = self.detector.inference_error
            frame = None if frame is None else frame.copy()
        return frame, result, error_msg


    def _draw_overlay(self, frame, result, error_msg):
        if result is None:
            return frame
        
        # Fehler oben rechts, falls vorhanden
        if error_msg:
            cv2.putText(frame, "INFERENCE ERROR!", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, error_msg, (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for box, score, label in zip(result["boxes_xyxy"], result["scores"], result["labels"]):
            x1, y1, x2, y2 = box

            # Bottom-Edge-Fix: Mittelpunkt sitzt auf der Unterkante, nicht in der Box-Mitte
            # (korrigiert Parallaxe-Fehler durch die schräge Kamera-Perspektive)
            cx, cy = int((x1 + x2) // 2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return frame