"""Track A — mAP50 / mAP50-95 / precision / recall on the test split (Ultralytics built-in)."""

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
    p.add_argument("--weights", default=str(_THIS_DIR / "runs" / "yolo26_finetune" / "weights" / "best.pt"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=str(DATA_YAML), split=args.split)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
