# utils.py
# unspezifische Helferfunktionen
import logging
import os
import re
import json
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def load_plan_from_json(filename: str):
    if not os.path.exists(filename):
        logger.error(f"Datei {filename} nicht gefunden.")
        return []
        
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not data.get("solved", False):
        logger.error("Der Plan war nicht erfolgreich/gelöst.")
        return []
        
    steps = []
    for step in data.get("actions", []):
        action = step["action"]
        obj = step["object"]
        slots = step.get("slots", [])
        steps.append((action, obj, slots))
        
    logger.info(f"Plan erfolgreich importiert! ({len(steps)} Schritte)")
    return steps


def strip_instance(object_name: str) -> str:
    """
    Entfernt Instanz-Suffixe wie '_1' oder '_green' am Ende eines Namens.
    Beispiel: 'blue_cover_1' -> 'blue_cover'
    """
    return re.sub(r'_\d*$', '', object_name)


def filter_oversized_detections(boxes, logits, phrases, frame_w, frame_h):
    """Filtert Erkennungen, die >95% der Framegröße haben."""
    max_box_size_ratio = 0.95
    frame_area = frame_w * frame_h
    max_allowed_area = frame_area * (max_box_size_ratio ** 2)
    
    valid_indices = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < max_allowed_area:
            valid_indices.append(i)
    
    return (
        boxes[valid_indices],
        logits[valid_indices],
        [phrases[i] for i in valid_indices]
    )


def pixel_to_robot(pixel_x: int, pixel_y: int, homography_matrix):
    """Transformiert Pixel-Koordinaten mithilfe der Homographie-Matrix in Roboter-Millimeter."""
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return float(transformed_point[0][0][0]), float(transformed_point[0][0][1])


def find_candidates(result: dict, expected_label: str) -> list:
    """Filtert Detections nach einem Label, das zum erwarteten label/objekt passt."""
    if result is None:
        return []
    candidates = []
    for box, score, label in zip(result["boxes_xyxy"], result["scores"], result["labels"]):
        # Grober Substring-Match
        if expected_label.lower() in label.lower() or label.lower() in expected_label.lower():
            candidates.append((box, score, label))
    return candidates
