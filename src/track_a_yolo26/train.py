"""Track A — fine-tune a pretrained YOLO26 (Ultralytics) on the full 14k-image dataset.

Usage:
    python train.py
    python train.py --model yolo26s.pt --epochs 100 --batch 8
"""

import sys
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from ultralytics import YOLO  # noqa: E402

from common.dataset_paths import DATA_YAML  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n.pt", help="pretrained checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=float, default=-1, help="-1 = Ultralytics AutoBatch (dò batch size an toàn theo VRAM)")
    p.add_argument("--name", default="yolo26_finetune")
    p.add_argument("--workers", type=int, default=4, help="giảm nếu bị MemoryError trong DataLoader worker (RAM hệ thống, không phải VRAM)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.model)  # auto-downloads pretrained COCO weights on first use
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(_THIS_DIR / "runs"),
        name=args.name,
        workers=args.workers,
    )
