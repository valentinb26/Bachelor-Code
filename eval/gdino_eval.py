# gdino_eval.py
# Läuft unabhängig vom Hauptsystem, nutzt aber dieselbe Inferenz wie vision_detector.py
# (gleiches Modell, gleiche Thresholds), damit die Zahlen für die Thesis gültig sind.

import os
import json
from pathlib import Path
from PIL import Image
import cv2
import torch
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

# TODO: Trag hier deine echten Prompts aus config.py OBJECT_CONFIG ein
CLASS_PROMPTS = {
    "wheel_small": "small black wheel",
    "screw_long": "long red screw screw bolt",
    "blue_cover": "blue large cover",
}

MASTER_PROMPT = " . ".join(CLASS_PROMPTS.values())


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
    return phrases, logits


def count_detections_per_class(phrases):
    """Zaehlt wie oft welches Klassen-Label in den erkannten Phrases vorkommt.
    Einfacher Substring-Match, kein IoU noetig fuer die Zaehl-Methode."""
    counts = {cls: 0 for cls in CLASS_PROMPTS}
    for phrase in phrases:
        for cls, prompt in CLASS_PROMPTS.items():
            # einfacher Check: taucht ein Wort aus dem Prompt in der Erkennung auf?
            if any(word in phrase.lower() for word in prompt.lower().split()):
                counts[cls] += 1
    return counts


def main():
    model, device = load_gdino()

    totals = {cls: {"gt": 0, "korrekt": 0, "erkannt_gesamt": 0} for cls in CLASS_PROMPTS}

    for image_name, gt_counts in GROUND_TRUTH.items():
        image_path = Path(IMAGE_DIR) / image_name
        if not image_path.exists():
            print(f"WARNUNG: {image_path} nicht gefunden, ueberspringe.")
            continue

        phrases, logits = run_inference(model, device, image_path)
        pred_counts = count_detections_per_class(phrases)

        print(f"\n{image_name}")
        print(f"  Erkannt (roh): {list(zip(phrases, [round(float(l), 2) for l in logits]))}")

        for cls in CLASS_PROMPTS:
            gt = gt_counts.get(cls, 0)
            pred = pred_counts.get(cls, 0)
            korrekt = min(gt, pred)  # simple: nicht mehr Treffer zaehlen als es GT gibt

            totals[cls]["gt"] += gt
            totals[cls]["korrekt"] += korrekt
            totals[cls]["erkannt_gesamt"] += pred

            if gt > 0 or pred > 0:
                print(f"  {cls}: GT={gt}, erkannt={pred}")

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

    # Fuer die Thesis speichern
    with open("eval/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2, ensure_ascii=False)
    print("\nGespeichert in eval_results.json")


if __name__ == "__main__":
    main()