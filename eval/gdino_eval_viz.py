# gdino_eval_with_viz.py
# Wie gdino_eval.py, aber zeichnet Boxen und Labels ins Bild

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import numpy as np
import sys

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    GROUNDING_DINO_CONFIG,
    GROUNDING_DINO_CHECKPOINT,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)
from ground_truth import GROUND_TRUTH

IMAGE_DIR = "eval/eval_images"
RESULT_DIR = "eval/eval_results"

CLASS_PROMPTS = {
    "wheel_small": "black wheel",
    "screw_long": "long screw",
    "blue_cover": "blue plastic cover",
}

MASTER_PROMPT = " . ".join(CLASS_PROMPTS.values())

# Farben für Boxen pro Klasse (RGB)
CLASS_COLORS = {
    "wheel_small": (0, 255, 0),      # Grün
    "screw_long": (255, 0, 0),       # Rot
    "blue_cover": (0, 0, 255),       # Blau
}


def load_gdino():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, device=device)
    model.eval()
    return model, device


def run_inference(model, device, image_path):
    frame = cv2.imread(str(image_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_source = Image.fromarray(frame_rgb)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_source, None)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=MASTER_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

    # Boxes sind normalized (0-1), müssen zurück zu Pixel-Koordinaten
    h, w = frame_rgb.shape[:2]
    boxes_pixel = boxes * torch.tensor([w, h, w, h])
    
    return phrases, logits, boxes_pixel.cpu().numpy()


def draw_boxes_on_image(image_path, phrases, logits, boxes_pixel, gt_counts):
    """Zeichnet erkannte Boxen ins Bild und speichert es."""
    frame = cv2.imread(str(image_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image_pil)

    # Versuche eine Schriftart zu laden, fallback auf default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    # Erkannte Boxen zeichnen
    for phrase, logit, box in zip(phrases, logits, boxes_pixel):
        x1, y1, x2, y2 = box.astype(int)
        
        # Validierung: Box muss im Bild sein
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue  # Überspringe ungültige Boxen
        
        # Bestimme Farbe basierend auf Label
        color = (200, 200, 200)  # Default grau
        for cls_name, cls_color in CLASS_COLORS.items():
            if any(word in phrase.lower() for word in CLASS_PROMPTS[cls_name].lower().split()):
                color = cls_color
                break
        
        # Box zeichnen
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Label mit Confidence
        label = f"{phrase} ({logit:.2f})"
        draw.text((x1, y1 - 20), label, fill=color, font=font)

    # Ground Truth Info als Text oben links
    gt_text = " | ".join([f"{cls}={cnt}" for cls, cnt in gt_counts.items()])
    draw.text((10, 10), f"GT: {gt_text}", fill=(255, 255, 255), font=font)

    return image_pil


def main():
    print("DEBUG: Starte Skript...")
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"RESULT_DIR: {RESULT_DIR}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"RESULT_DIR erstellt/existiert")

    print(f"GROUND_TRUTH hat {len(GROUND_TRUTH)} Einträge")
    print("DEBUG: Imports OK")
    
    model, device = load_gdino()

    totals = {cls: {"gt": 0, "korrekt": 0, "erkannt_gesamt": 0} for cls in CLASS_PROMPTS}

    for image_name, gt_counts in GROUND_TRUTH.items():
        image_path = Path(IMAGE_DIR) / image_name
        if not image_path.exists():
            print(f"WARNUNG: {image_path} nicht gefunden, ueberspringe.")
            continue

        print(f"Verarbeite {image_name}...", end=" ")
        
        phrases, logits, boxes_pixel = run_inference(model, device, image_path)

        # Visualisierung zeichnen
        image_annotated = draw_boxes_on_image(image_path, phrases, logits, boxes_pixel, gt_counts)
        
        # Speichern
        result_path = Path(RESULT_DIR) / f"annotated_{image_name}"
        image_annotated.save(result_path)
        print(f"gespeichert in {result_path}")

        # Metriken berechnen (wie vorher)
        for cls in CLASS_PROMPTS:
            gt = gt_counts.get(cls, 0)
            pred_count = sum(1 for phrase in phrases 
                            if any(word in phrase.lower() for word in CLASS_PROMPTS[cls].lower().split()))
            korrekt = min(gt, pred_count)

            totals[cls]["gt"] += gt
            totals[cls]["korrekt"] += korrekt
            totals[cls]["erkannt_gesamt"] += pred_count

    print("\n" + "=" * 50)
    print("GESAMTERGEBNIS")
    print("=" * 50)
    for cls, vals in totals.items():
        gt = vals["gt"]
        korrekt = vals["korrekt"]
        erkannt = vals["erkannt_gesamt"]
        recall = korrekt / gt if gt > 0 else float("nan")
        precision = korrekt / erkannt if erkannt > 0 else float("nan")
        print(f"{cls}: Recall={recall:.2f} ({korrekt}/{gt}), Precision={precision:.2f} ({korrekt}/{erkannt})")

    with open(Path(RESULT_DIR) / "results.json", "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert in {RESULT_DIR}/results.json")
    
if __name__ == "__main__":
    main()