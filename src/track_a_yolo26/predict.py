"""Track A — run the fine-tuned YOLO26 on one image and save an annotated copy.

Usage:
    python predict.py --weights runs/yolo26_finetune/weights/best.pt --image "path/to/image.jpg"
"""

import sys
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(_THIS_DIR / "runs" / "yolo26_finetune" / "weights" / "best.pt"))
    p.add_argument("--image", required=True)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", default=str(_THIS_DIR / "runs" / "prediction.jpg"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(args.image, conf=args.conf)
    results[0].save(filename=args.out)
    print(f"saved: {args.out} ({len(results[0].boxes)} boxes)")
