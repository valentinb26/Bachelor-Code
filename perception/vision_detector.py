# vision_detector.py
# Kapselt Kamera, Bild-Entzerrung, KI-Modell und Koordinaten-Transformation.
import sys
import time
import torch
import cv2
import threading
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any

from utils.utils import (
    filter_oversized_detections, pixel_to_robot, find_candidates, get_object_config
)
from config.config import (
    GROUNDING_DINO_CONFIG, 
    GROUNDING_DINO_CHECKPOINT, 
    HOMOGRAPHY_PATH, 
    USE_CALIBRATION,
    CALIBRATION_PATH,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    Z_GROUND,
    GRASP_STRATEGIES
    )

logger = logging.getLogger(__name__)


class VisionDetector:
    def __init__(self, target_prompt: str):
        # Lädt Kalibrierungsdaten und das GDINO-Modell
        # Startet den Video-Thread und den Inferenz-Thread im Hintergrund
        self.cap = cv2.VideoCapture(0)

        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.homography_matrix = None
        self.mtx = None
        self.dist = None
        
        # Thread-Zustände 
        self.frame_lock = threading.Lock()

        self.current_frame_bgr = None
        self.current_result = None
        self.current_prompt = target_prompt
        logger.debug(f"Master Prompt: {self.current_prompt}")
        self.inference_active = True
        self.inference_error= None

        # Automatisches laden
        self._load_calibration()
        self._load_model()

        # Threads direkt im Hintergrund starten
        self.video_thread = threading.Thread(target=self._video_thread_loop, daemon=False)
        self.inference_thread = threading.Thread(target=self._inference_thread_loop, daemon=False)
        self.video_thread.start()
        self.inference_thread.start()


    def _load_calibration(self): 
        """Lädt Homographie- und Kameramatrizen."""
        try:
            self.homography_matrix = np.load(HOMOGRAPHY_PATH)
            logger.info("Homographie-Matrix geladen.")
        except FileNotFoundError:
            logger.error(f"'{HOMOGRAPHY_PATH}' nicht gefunden! Bitte Kalibirierungsskript durchführen! Skript wird beendet.")
            #sys.exit(1)

        try:
            if USE_CALIBRATION:
                calib_data = np.load(CALIBRATION_PATH)
                self.mtx, self.dist = calib_data["mtx"], calib_data["dist"]
                logger.info(f"{CALIBRATION_PATH}' geladen!")
            else:
                logger.info("Kamerakalibrierung ist laut Config deaktiviert. Fahre ohne Entzerrung fort.")
                self.mtx, self.dist = None, None
                return
        except FileNotFoundError:
            logger.warning("Kamerakalibrierung nicht gefunden. Fahre ohne Entzerrung fort.")
            self.mtx, self.dist = None, None


    def _load_model(self): 
        """Prüft die Verfügbarkeit von GroundingDINO."""
        try:
            global load_model, predict, T, box_ops
            from groundingdino.util.inference import load_model, load_image, predict
            import groundingdino.datasets.transforms as T
            from groundingdino.util import box_ops
        except ImportError:
            logger.critical("GroundingDINO nicht installiert --> pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
            sys.exit(1)

        """Lädt GroundedDINO-Modell und gibt (model, device) zurück."""
        if not GROUNDING_DINO_CONFIG.exists():
            logger.critical(f"Config nicht gefunden: {GROUNDING_DINO_CONFIG}")
            sys.exit(1)
        if not GROUNDING_DINO_CHECKPOINT.exists():
            logger.critical(f"Checkpoint nicht gefunden: {GROUNDING_DINO_CHECKPOINT}!  Checkpoint herunterladen:\n wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth")
            sys.exit(1)

        logger.info(f"Lade Modell auf {self.device.upper()}...")
        t0 = time.perf_counter()
        self.model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, device=self.device)
        self.model.eval()
        t1 = time.perf_counter()
        logger.debug(f"Modell geladen in {t1 - t0:.2f}s")


    def _video_thread_loop(self):     
        """Läuft eigenständig und füllt immer das aktuellste Frame in die globale Variable."""
        logger.info("Gestartet, lese Frames...")

        while self.inference_active:
            ret, frame = self.cap.read()
            if not ret:
                logger.critical("Kamera-Fehler, breche ab.")
                self.inference_active = False
                break
            
            with self.frame_lock:
                self.current_frame_bgr = frame.copy()


    def _inference_thread_loop(self): 
        """Läuft eigenständig und macht Inferenz auf dem aktuellsten Frame."""
        logger.info("Gestartet, warte auf Frames...")
        processing_cache = {"newcameramtx": None, "roi": None}
        
        while self.inference_active:
            frame_raw = None

            with self.frame_lock:
                if self.current_frame_bgr is not None:
                    frame_raw = self.current_frame_bgr.copy()
                prompt = self.current_prompt
                
            if frame_raw is None or prompt is None:   
                time.sleep(0.01) # außerhalb des locks warten
                continue
                
            try:
                # Entzerrung
                if USE_CALIBRATION:
                    frame = self.undistort_frame(frame_raw, processing_cache)
                else:
                    frame = frame_raw
                
                # Inferenz (NICHT unter Lock, blocking!)
                t0 = time.perf_counter()
                result = self.run_inference_on_frame(frame, prompt)
                t1 = time.perf_counter()
                logger.debug(f"Zeit für '{prompt}' war: {t1-t0:.2f}")
                
                with self.frame_lock:
                    self.current_result = result
                    self.inference_error = None  # Error gelöscht, wenn erfolgreich
                    
            except Exception as e:
                logger.error(f"{type(e).__name__}: {e}")
                with self.frame_lock:
                    self.inference_error = f"{type(e).__name__}: {str(e)[:60]}"

    
# ---------------------------------------------------------------------------
#  HAUTPFUNKTION 
# ---------------------------------------------------------------------------     

    
    def get_robot_coordinates_for_object(self, expected_label: str, timeout: float = 10.0, poll_interval: float = 0.3, center: bool = False ) -> Optional[Tuple[float, float]]:
        """Wartet bis ein passendes Objekt im aktuellen Frame gefunden wird.
        Prompt muss vorher mit set_prompt() gesetzt sein."""
        # Rechnet bei Erfolg Pixel in Roboter-Millimeter um 
        # Gibt (robot_x, robot_y) zurück oder None

        t0 = time.time()
        while time.time() - t0 < timeout:

            with self.frame_lock:
                result = self.current_result
                frame = self.current_frame_bgr.copy()
                
            # Filtern
            candidates = find_candidates(result, expected_label)
            if candidates:
                # Besten Kandidaten nach höchsten Score wählen
                best_candidate = max(candidates, key=lambda c: c[1])
                box, score, label = best_candidate
                
                # 1. Segmentiermaske
                #mask, bbox = self.get_segmentation_mask(
                #    frame, 
                #    result["boxes_xyxy"],  # <-- numpy arrays aus result
                #    result["scores"],
                #    result["labels"],
                #    expected_label
                #)
                
                # 2. Orientierung
                #angle_deg, conf = self.estimate_2d_orientation(mask, bbox)

                # 3. Komplette Pose
                #detection_dict = {
                #    "bbox": bbox,
                #    "class": label,
                #    "confidence": score,
                #    "image_point": (int((box[0] + box[2]) // 2), int(box[3]))  # Centroid
                #}
                
                #grasp_pose = self.compute_grasp_pose(detection_dict, mask, label)
            
                #logger.info(f"Pose: x={grasp_pose['x_mm']:.1f}, y={grasp_pose['y_mm']:.1f}, "
                #       f"z={grasp_pose['z_mm']:.1f}, rot={grasp_pose['gripper_rotation_deg']:.1f}°")
                
                #return grasp_pose  # <-- Komplette Pose, nicht nur XY

                # KOMMT WEG
                
                
                x1, y1, x2, y2 = box
                if center:
                    # Mittelpunkt der Bounding Box
                    cx, cy = int((x1 + x2) // 2), int((y1 +y2) // 2)
                else:
                    # Mittelpunkt untere Höhe
                    cx, cy = int((x1 + x2) // 2), int(y2)
                

                if self.homography_matrix is None:
                    logger.error("Keine Homographie-Matrix geladen. Transformation unmöglich!")
                    return None
                
                robot_x, robot_y = pixel_to_robot(cx, cy, self.homography_matrix)
                logger.info(f"    -> Gefunden: {label} ({score:.2f}) bei Pixel ({cx},{cy}) -> Roboter ({robot_x:.1f}, {robot_y:.1f})")
                return robot_x, robot_y
        
            time.sleep(poll_interval)

        logger.warning(f"  -> [TIMEOUT] '{expected_label}' wurde innerhalb von {timeout}s nicht erkannt.")
        return None

# ---------------------------------------------------------------------------
#  SPEZIFISCHE HILFSMETHODEN 
# ---------------------------------------------------------------------------        

    def set_prompt(self, new_prompt: str):
        # Aktuelle nicht verwendet, könnte aber später wichtig werden..
        """Wechselt den aktuellen Suchprompt während der Laufzeit."""
        with self.frame_lock:
            self.current_prompt = new_prompt
    

    def undistort_frame(self, frame, cache):
        """Entzerrt das Kamerabild basierend auf den Kalibrierungsdaten."""
        if self.mtx is None or self.dist is None:
            cv2.putText(frame, "Linse Normal (Unkalibriert)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        h, w = frame.shape[:2]
        if cache.get("newcameramtx") is None:
            cache["newcameramtx"], cache["roi"] = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        return cv2.undistort(frame, self.mtx, self.dist, None, cache["newcameramtx"])


    def run_inference_on_frame(self, frame, prompt):
        """Führt die Inferenz auf einem OpenCV-Frame aus."""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        # convert frame to a PIL object in RGB space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_source = Image.fromarray(frame_rgb)
        #image_source.save("debug_frame.jpg")

        # convert the PIL image object to a transform object
        image_transformed, _ = transform(image_source, None)
        #ToPILImage()(image_transformed).save("transformed_frame.jpg")


        # predict boxes, logits, phrases
        t0 = time.perf_counter()
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model, 
                image=image_transformed, 
                caption=prompt, 
                box_threshold=BOX_THRESHOLD, 
                text_threshold=TEXT_THRESHOLD,
                device=self.device)  
        inference_ms = (time.perf_counter() - t0) * 1000

        w, h = image_source.size
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])

        boxes_xyxy_np = boxes_xyxy.cpu().numpy().astype(int)
        logits_np = logits.cpu().numpy()

        # Filter oversized detections
        boxes_filtered, logits_filtered, phrases_filtered = filter_oversized_detections(boxes_xyxy_np, logits_np, phrases, w, h)
        
        return {
            "boxes_xyxy":   boxes_filtered,
            "scores":       logits_filtered,
            "labels":       phrases_filtered,
            "inference_ms": inference_ms,
            "num_detections": len(phrases),
        }


    def get_segmentation_mask(image, boxes, logits, phrases, target_phrase):
        """
        Extrahiere Segmentiermaske für ein erkanntes Objekt.
        
        Args:
            image: PIL Image oder numpy array (H, W, 3)
            boxes: tensor (N, 4) in normalized coordinates [x1, y1, x2, y2]
            logits: tensor (N,) confidence scores
            phrases: list of str, erkannte Klassen
            target_phrase: str, z.B. "bumper"
        
        Returns:
            mask: numpy array (H, W) binary mask
            bbox: (x1, y1, x2, y2) in Pixel
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        H, W = img_np.shape[:2]
        
        # Finde Index des target Objekts
        idx = None
        for i, phrase in enumerate(phrases):
            if target_phrase.lower() in phrase.lower():
                idx = i
                break
        
        if idx is None:
            return None, None
        
        # Denormalize bbox
        box = boxes[idx]  # [x1, y1, x2, y2] normalized
        x1, y1, x2, y2 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
        
        # Einfache binäre Maske aus BBox (crude, aber funktioniert für Szenario A)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Optional: Morphological operations für bessere Maske
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask, (x1, y1, x2, y2)
    
    
    def estimate_2d_orientation(mask, bbox):
        """
        Schätze 2D-Orientierung (Winkel um Z-Achse) aus Segmentiermaske.
        
        Args:
            mask: binary mask (H, W)
            bbox: (x1, y1, x2, y2)
        
        Returns:
            angle_deg: float, Winkel in Grad (0-360)
            confidence: float (0-1), wie sicher die Schätzung ist
        """
        # Konturen finden
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or len(contours[0]) < 5:
            # Zu kleine Maske, fallback auf BBox-Winkel
            return 0.0, 0.1
        
        # Größte Kontur nehmen
        cnt = max(contours, key=cv2.contourArea)
        
        # Fitting ellipse (gibt längste Achse)
        if len(cnt) < 5:
            return 0.0, 0.1
        
        ellipse = cv2.fitEllipse(cnt)
        # ellipse = (center, (major_axis, minor_axis), angle)
        center, axes, angle = ellipse
        
        # Angle ist schon in Grad, aber relativ zur Bildachse
        # Normalisiere auf 0-360
        angle_deg = float(angle) % 360
        
        # Confidence basierend auf Achsen-Verhältnis
        # Je deutlicher die Asymmetrie, desto sicherer die Orientierung
        major, minor = axes
        if major > 0:
            aspect_ratio = minor / major
            confidence = min(1.0, 1.0 - aspect_ratio)  # 0 = rund, 1 = lang
        else:
            confidence = 0.1
        
        return angle_deg, confidence


    def compute_grasp_pose(self, detection, mask, object_class):
        """
        Berechne Greif-Pose basierend auf Objekt-Klasse und Strategie.
        
        Args:
            detection: {
                "bbox": (x1, y1, x2, y2),
                "class": str (z.B. "bumper"),
                "confidence": float,
                "image_point": (cx, cy)  # Centroid aus GDINO
            }
            mask: binary segmentation mask
            object_class: str, Klassen-Name
        
        Returns:
            grasp_pose: {
                "x_mm": float,
                "y_mm": float,
                "z_mm": float,
                "gripper_rotation_deg": float,
                "strategy": str
            }
        """
        strategy = GRASP_STRATEGIES.get(object_class.lower())
        if not strategy:
            return None
        
        # 1. Position: Centroid aus Detection (in Pixel)
        cx, cy = detection["image_point"]
        
        # 2. Homographie anwenden (deine bestehende 8-point Kalibrierung)
        # (diese Funktion hast du bereits)
        x_mm, y_mm = pixel_to_robot(cx, cy)
        
        # 3. Z-Koordinate: Z_GROUND + Objekthöhe
        bbox = detection["bbox"]
        #obj_height_px = bbox[3] - bbox[1]  # Pixel-Höhe
        z_mm = Z_GROUND + get_object_config(object_class).get("height", 20)  # Fallback 20mm
        #z_mm = Z_GROUND + (obj_height_px * MM_PER_PIXEL)  # deine Kalibrierung
        
        # 4. Gripper-Rotation (wenn Teil nicht symmetrisch)
        if strategy["use_orientation"]:
            angle_deg, confidence = self.estimate_2d_orientation(mask, bbox)
            gripper_rotation = angle_deg
        else:
            gripper_rotation = strategy["gripper_rotation"]
        
        return {
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "gripper_rotation_deg": gripper_rotation,
            "strategy_name": strategy["name"],
            "strategy_desc": strategy["description"],
            "confidence": detection["confidence"]
        }


    def _get_object_height(self, object_name: str) -> float:
        """Ermittelt die feste Position aus der Objekt-Konfiguration für einen Objekt-Namen."""
        return self._get_object_config(object_name).get("height")  # Gibt die Höhe in mm zurück


    def shutdown(self):
        """Räumt alle Ressourcen sauber auf."""

        logger.info("Fahre Vision-System herunter...")
        self.inference_active = False
        
        logger.info("Warte auf Video-Thread (max 3s)...")
        self.video_thread.join(timeout=3)
        if self.video_thread.is_alive():
            logger.warning("Video-Thread antwortet nicht (aber daemon=False, wird erzwungen)")
            sys.exit(1)
        
        logger.info("Warte auf Inference-Thread (max 3s)...")
        self.inference_thread.join(timeout=3)
        if self.inference_thread.is_alive():
            logger.warning("Inference-Thread antwortet nicht (aber daemon=False, wird erzwungen)")
            sys.exit(1)
        
        if self.cap.isOpened():
            self.cap.release()
        logger.info("Erfolgreich heruntergefahren.")