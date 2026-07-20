"""Track B — fine-tune a pretrained RF-DETR (transformer detector) on the full dataset.

RF-DETR (Roboflow, 2025/2026) reads our existing YOLO-format dataset directly --
it auto-detects a valid YOLO layout (data.yaml + train/valid with images/labels
subfolders), so no COCO conversion is needed; class count is aligned from the
dataset automatically.

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 4
"""

import sys
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from rfdetr import RFDETRNano  # noqa: E402

from common.dataset_paths import DATASET_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4, help="RTX 2060 6GB -> keep small; RF-DETR is a transformer")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", default=str(_THIS_DIR / "runs" / "rfdetr_finetune"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = RFDETRNano()  # pretrained checkpoint auto-downloaded on first use
    model.train(
        dataset_dir=str(DATASET_ROOT),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
