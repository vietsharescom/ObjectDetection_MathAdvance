"""Track B — run the fine-tuned RF-DETR on one image and save an annotated copy.

Usage:
    python predict.py --weights runs/rfdetr_finetune/checkpoint_best_total.pth --image "path/to/image.jpg"
"""

import sys
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

import supervision as sv  # noqa: E402
from PIL import Image  # noqa: E402
from rfdetr import RFDETRNano  # noqa: E402

from common.dataset_paths import CLASS_NAMES  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to a fine-tuned checkpoint, e.g. checkpoint_best_total.pth")
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=str(_THIS_DIR / "runs" / "prediction.jpg"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = RFDETRNano.from_checkpoint(args.weights)

    image = Image.open(args.image).convert("RGB")
    detections = model.predict(image, threshold=args.threshold)

    labels = [f"{CLASS_NAMES[c]} {s:.2f}" for c, s in zip(detections.class_id, detections.confidence)]
    annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
    annotated.save(args.out)
    print(f"saved: {args.out} ({len(detections)} boxes)")
