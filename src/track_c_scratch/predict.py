"""Track C — run the from-scratch detector on one image and save an annotated copy.

Usage:
    python predict.py --image "path/to/image.jpg" --out prediction.jpg
"""

import sys
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_SRC_DIR))

import cv2  # noqa: E402
import torch  # noqa: E402

from model import ScratchDetector  # noqa: E402
from utils import decode_predictions, nms  # noqa: E402
from common.dataset_paths import CLASS_NAMES  # noqa: E402

BOX_COLORS = [(255, 0, 0), (0, 200, 0)]  # Head, Person


def load_model(weights_path, grid_size=8, device="cpu"):
    model = ScratchDetector(grid_size=grid_size).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict_image(model, img_path, img_size=128, grid_size=8, conf_thresh=0.3, iou_thresh=0.4, device="cpu"):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig = img.copy()
    resized = cv2.resize(img, (img_size, img_size))
    img_t = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_t = img_t.to(device)

    with torch.no_grad():
        pred = model(img_t)[0]
    boxes = decode_predictions(pred, grid_size=grid_size, conf_thresh=conf_thresh)
    boxes = nms(boxes, iou_thresh=iou_thresh)
    return orig, boxes


def draw_boxes(image, boxes):
    h, w = image.shape[:2]
    img = image.copy()
    for x1, y1, x2, y2, score, cls_id in boxes:
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        color = BOX_COLORS[cls_id]
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.putText(img, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(_THIS_DIR / "runs" / "scratch_detector.pt"))
    p.add_argument("--image", required=True)
    p.add_argument("--conf-thresh", type=float, default=0.3)
    p.add_argument("--out", default=str(_THIS_DIR / "runs" / "prediction.jpg"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device=device)
    orig, boxes = predict_image(model, args.image, conf_thresh=args.conf_thresh, device=device)
    result = draw_boxes(orig, boxes)
    cv2.imwrite(args.out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"saved: {args.out} ({len(boxes)} boxes)")
