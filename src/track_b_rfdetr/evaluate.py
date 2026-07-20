"""Track B — mAP50 / mAP50-95 on the test split, via supervision's MeanAveragePrecision.

Usage:
    python evaluate.py --weights runs/rfdetr_finetune/checkpoint_best_total.pth
"""

import sys
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402
from PIL import Image  # noqa: E402
from supervision.metrics import MeanAveragePrecision  # noqa: E402
from rfdetr import RFDETRNano  # noqa: E402

from common.dataset_paths import TEST_IMAGES, TEST_LABELS  # noqa: E402
from track_c_scratch.dataset import list_image_label_pairs, read_yolo_labels  # noqa: E402


def ground_truth_detections(label_path, img_w, img_h):
    boxes, class_ids = [], []
    for cls_id, xc, yc, w, h in read_yolo_labels(label_path):
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        class_ids.append(cls_id)
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(xyxy=np.array(boxes, dtype=np.float32), class_id=np.array(class_ids, dtype=int))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--max-images", type=int, default=None, help="cap eval set size for a quick run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = RFDETRNano.from_checkpoint(args.weights)

    pairs = list_image_label_pairs(TEST_IMAGES, TEST_LABELS)
    if args.max_images:
        pairs = pairs[: args.max_images]
    print(f"evaluating on {len(pairs)} test images")

    metric = MeanAveragePrecision()
    predictions, targets = [], []
    for img_path, label_path in pairs:
        image = Image.open(img_path).convert("RGB")
        preds = model.predict(image, threshold=args.threshold)
        gt = ground_truth_detections(label_path, image.width, image.height)
        predictions.append(preds)
        targets.append(gt)

    result = metric.update(predictions, targets).compute()
    print(f"mAP50:    {result.map50:.4f}")
    print(f"mAP50-95: {result.map50_95:.4f}")
