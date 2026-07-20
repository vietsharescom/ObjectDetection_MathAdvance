"""Track C — precision/recall @ IoU 0.5 on the validation subset.

Note: this is a simplified metric (not full COCO mAP with a recall-sweep) --
appropriate here because Track C is a small hand-rolled detector. Tracks A/B
use Ultralytics' / RF-DETR's built-in mAP50 / mAP50-95 evaluation, which is
the number that should anchor the report's cross-track comparison.
"""

import sys
import json
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402

from dataset import HeadPersonSubsetDataset, read_yolo_labels  # noqa: E402
from model import ScratchDetector  # noqa: E402
from utils import decode_predictions, nms, iou_xyxy, yolo_to_xyxy  # noqa: E402
from common.dataset_paths import VALID_IMAGES, VALID_LABELS  # noqa: E402


def evaluate(model, dataset, device, conf_thresh=0.3, iou_match_thresh=0.5):
    tp, fp, fn = 0, 0, 0
    for idx in range(len(dataset)):
        img_path, label_path = dataset.pairs[idx]
        img_t, _ = dataset[idx]
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_t)[0]
        pred_boxes = nms(
            decode_predictions(pred, grid_size=dataset.grid_size, conf_thresh=conf_thresh),
            iou_thresh=0.4,
        )

        gt_boxes = [(cls_id, *yolo_to_xyxy(xc, yc, w, h))
                    for cls_id, xc, yc, w, h in read_yolo_labels(label_path)]
        matched = set()
        for x1, y1, x2, y2, score, cls_id in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, (gcls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if gcls != cls_id or j in matched:
                    continue
                iou = iou_xyxy([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_match_thresh:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(_THIS_DIR / "runs" / "scratch_detector.pt"))
    p.add_argument("--subset-size", type=int, default=200)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--conf-thresh", type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = HeadPersonSubsetDataset(
        VALID_IMAGES, VALID_LABELS, subset_size=args.subset_size,
        img_size=args.img_size, grid_size=args.grid_size, seed=123,
    )
    model = ScratchDetector(grid_size=args.grid_size).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    metrics = evaluate(model, val_ds, device, conf_thresh=args.conf_thresh)
    print(json.dumps(metrics, indent=2))
