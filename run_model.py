"""Run a trained detector: score it, or draw predictions on images.

One CLI over both model families, so Track A and Track C are driven the same way
and — importantly — scored by the same metric (metrics.py). Track B/RF-DETR is not
included here because its checkpoint loader needs the rfdetr package and a CUDA
build; use the notebook for that track.

    # score a fine-tuned YOLO on the test split
    python run_model.py evaluate --track a \
        --weights ../outputs/track_a/finetune/weights/best.pt

    # score the from-scratch CNN, capped at 200 images for a quick pass
    python run_model.py evaluate --track c \
        --weights ../outputs/track_c/scratch_detector.pt --limit 200

    # draw predictions on some images
    python run_model.py predict --track a \
        --weights ../outputs/track_a/finetune/weights/best.pt \
        --source ../../Person\\ Detection\\ with\\ head.v2i.yolov8/test/images \
        --out ../outputs/preds --limit 12

Torch is imported lazily so `--help` and `--check` work without it installed.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from metrics import evaluate_detections, yolo_labels_to_xyxy, iou_matrix, CLASS_NAMES  # noqa: E402

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")
IMG_SIZE, GRID_SIZE = 128, 8
COLOURS = [(66, 133, 244), (219, 132, 82)]  # Head, Person — matches the notebook


# ---------------------------------------------------------------- dataset ----
def find_dataset_root(start: Path | None = None) -> Path | None:
    """Walk up looking for the extracted Roboflow export."""
    start = start or _THIS
    for parent in [start, *start.parents]:
        for cand in parent.glob("*"):
            if cand.is_dir() and (cand / "data.yaml").exists():
                return cand
        if (parent / "data.yaml").exists():
            return parent
    return None


def list_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for ext in IMG_EXTS:
        for img in sorted(Path(images_dir).glob(ext)):
            lbl = Path(labels_dir) / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return sorted(pairs)


def list_images(source: Path):
    source = Path(source)
    if source.is_file():
        return [source]
    out = []
    for ext in IMG_EXTS:
        out.extend(sorted(source.glob(ext)))
    return out


def load_rgb(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def empty_pred():
    return {"boxes": np.zeros((0, 4), np.float32),
            "scores": np.zeros(0, np.float32),
            "labels": np.zeros(0, int)}


# ------------------------------------------------------------------ device ---
def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------- track A ---
def build_track_a(weights: str, conf: float):
    from ultralytics import YOLO
    model = YOLO(weights)

    def predict(img_rgb):
        # Ultralytics expects BGR for numpy input.
        r = model.predict(img_rgb[:, :, ::-1], conf=conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return empty_pred()
        return {
            "boxes": r.boxes.xyxy.cpu().numpy().astype(np.float32),
            "scores": r.boxes.conf.cpu().numpy().astype(np.float32),
            "labels": r.boxes.cls.cpu().numpy().astype(int),
        }

    return predict


# ----------------------------------------------------------------- track C ---
def _scratch_model(num_classes=2, grid_size=GRID_SIZE, dropout=0.1):
    import torch.nn as nn

    def conv_block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.MaxPool2d(2),
        )

    class ScratchDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid_size = grid_size
            self.backbone = nn.Sequential(
                conv_block(3, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128),
            )
            self.head = nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True), nn.Dropout2d(dropout),
                nn.Conv2d(128, 5 + num_classes, 1),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    return ScratchDetector()


def decode_predictions(pred, grid_size=GRID_SIZE, conf_thresh=0.25):
    """(5+C,S,S) logits -> [x1,y1,x2,y2,score,cls] rows in normalised [0,1] coords."""
    import torch
    S = grid_size
    xy, wh = torch.sigmoid(pred[0:2]), torch.sigmoid(pred[2:4])
    obj, cls = torch.sigmoid(pred[4]), torch.softmax(pred[5:], dim=0)

    cls_score, cls_id = cls.max(dim=0)
    score = obj * cls_score
    keep = score >= conf_thresh
    if not keep.any():
        return np.zeros((0, 6), np.float32)

    gj, gi = torch.nonzero(keep, as_tuple=True)
    xc = (gi.float() + xy[0][keep]) / S
    yc = (gj.float() + xy[1][keep]) / S
    w, h = wh[0][keep], wh[1][keep]
    out = torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2,
                       score[keep], cls_id[keep].float()], dim=1)
    return out.cpu().numpy().astype(np.float32)


def nms(boxes, iou_thresh=0.45):
    """Greedy per-class NMS over [x1,y1,x2,y2,score,cls] rows."""
    if len(boxes) == 0:
        return boxes
    keep = []
    for cls_id in np.unique(boxes[:, 5]):
        cb = boxes[boxes[:, 5] == cls_id]
        cb = cb[np.argsort(-cb[:, 4])]
        while len(cb):
            best, cb = cb[0], cb[1:]
            keep.append(best)
            if len(cb):
                cb = cb[iou_matrix(best[None, :4], cb[:, :4])[0] < iou_thresh]
    return np.array(keep, dtype=np.float32)


def build_track_c(weights: str, conf: float):
    import cv2
    import torch

    device = pick_device()
    model = _scratch_model().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    def predict(img_rgb):
        h, w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        t = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            raw = model(t)[0]
        boxes = nms(decode_predictions(raw, conf_thresh=conf))
        if len(boxes) == 0:
            return empty_pred()
        # normalised -> absolute pixels of the ORIGINAL image
        return {
            "boxes": boxes[:, :4] * np.array([w, h, w, h], np.float32),
            "scores": boxes[:, 4],
            "labels": boxes[:, 5].astype(int),
        }

    return predict


BUILDERS = {"a": build_track_a, "c": build_track_c}

_MISSING = {
    "torch": "pip install torch torchvision",
    "ultralytics": "pip install ultralytics",
    "cv2": "pip install opencv-python",
}


def build_predictor(track: str, weights: str, conf: float):
    """Construct a predictor, turning the two common setup failures into
    actionable messages instead of a traceback."""
    if not Path(weights).exists():
        raise SystemExit(
            f"weights not found: {weights}\n"
            f"Train track {track.upper()} first — see Kazim/notebooks/"
            "Person_Head_Detection_Kazim.ipynb"
        )
    try:
        return BUILDERS[track](weights, conf)
    except ImportError as exc:
        pkg = (exc.name or "").split(".")[0]
        hint = _MISSING.get(pkg, f"pip install {pkg}")
        raise SystemExit(f"missing dependency '{pkg}' — install it with:\n    {hint}")


# ---------------------------------------------------------------- drawing ----
def draw(img_rgb, boxes, labels, scores=None):
    import cv2
    out = img_rgb.copy()
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        colour = COLOURS[int(lab) % len(COLOURS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = CLASS_NAMES[int(lab)]
        if scores is not None:
            text += f" {scores[i]:.2f}"
        cv2.putText(out, text, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------- commands ----
def cmd_evaluate(args):
    root = Path(args.dataset) if args.dataset else find_dataset_root()
    if root is None:
        raise SystemExit("dataset not found — pass --dataset /path/to/'Person Detection with head.v2i.yolov8'")

    split = args.split
    if not (root / split).is_dir():
        print(f"[warn] no '{split}' split; falling back to 'valid'")
        split = "valid"

    pairs = list_pairs(root / split / "images", root / split / "labels")
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(f"no image/label pairs found under {root / split}")

    # Build first: fail on bad weights / missing deps before printing a banner.
    predict = build_predictor(args.track, args.weights, args.conf)

    print(f"dataset : {root}")
    print(f"split   : {split}  ({len(pairs)} images)")
    print(f"track   : {args.track.upper()}  weights={args.weights}")
    print(f"conf    : {args.conf}\nscoring ...")

    preds, gts = [], []
    for n, (img_path, label_path) in enumerate(pairs, 1):
        img = load_rgb(img_path)
        h, w = img.shape[:2]
        preds.append(predict(img))
        gts.append(yolo_labels_to_xyxy(label_path, w, h))
        if n % 100 == 0:
            print(f"  {n}/{len(pairs)}")

    res = evaluate_detections(preds, gts, CLASS_NAMES)

    print(f"\n{'':<10}{'AP50':>10}{'AP50-95':>10}")
    print("-" * 30)
    for cls_name, ap in res["per_class"].items():
        print(f"{cls_name:<10}{ap['AP50']:>10.4f}{ap['AP50_95']:>10.4f}")
    print("-" * 30)
    print(f"{'mAP':<10}{res['mAP50']:>10.4f}{res['mAP50_95']:>10.4f}")

    total_pred = sum(len(p['boxes']) for p in preds)
    total_gt = sum(len(g['boxes']) for g in gts)
    print(f"\npredicted {total_pred} boxes vs {total_gt} ground truth "
          f"({total_pred / max(total_gt, 1):.2f}x)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"track": args.track, "weights": args.weights, "split": split,
                   "images": len(pairs), "conf": args.conf, **res}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print("saved ->", args.out)


def cmd_predict(args):
    import cv2

    images = list_images(args.source)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"no images found at {args.source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predict = build_predictor(args.track, args.weights, args.conf)

    print(f"track {args.track.upper()} -> {len(images)} images -> {out_dir}")
    total = 0
    for img_path in images:
        img = load_rgb(img_path)
        p = predict(img)
        total += len(p["boxes"])
        annotated = draw(img, p["boxes"], p["labels"], p["scores"])
        dest = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(dest), annotated[:, :, ::-1])  # back to BGR for imwrite
        print(f"  {img_path.name:<52} {len(p['boxes']):>3} boxes")
    print(f"\n{total} boxes over {len(images)} images "
          f"({total / len(images):.1f} per image)")


def cmd_check(args):
    """Verify wiring without needing torch or trained weights."""
    root = find_dataset_root()
    print("dataset root :", root or "NOT FOUND")
    if root:
        for split in ("train", "valid", "test"):
            d = root / split
            if d.is_dir():
                pairs = list_pairs(d / "images", d / "labels")
                print(f"  {split:<6} {len(pairs):>6} image/label pairs")

    # metric + NMS round-trip on synthetic data — no torch required
    gt = [{"boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]], np.float32),
           "labels": np.array([0, 1])}]
    pr = [{"boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]], np.float32),
           "scores": np.array([0.9, 0.8], np.float32), "labels": np.array([0, 1])}]
    res = evaluate_detections(pr, gt, CLASS_NAMES)
    assert abs(res["mAP50"] - 1.0) < 1e-6, res
    print(f"metric       : OK (perfect prediction -> mAP50={res['mAP50']:.4f})")

    boxes = np.array([[10, 10, 50, 50, 0.9, 0],
                      [12, 12, 52, 52, 0.8, 0],
                      [200, 200, 260, 260, 0.7, 1]], np.float32)
    kept = nms(boxes, 0.45)
    assert len(kept) == 2, kept
    print(f"nms          : OK (3 boxes -> {len(kept)} after suppression)")
    print("\nWiring OK. Train a model, then run 'evaluate' or 'predict'.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--track", choices=["a", "c"], required=True,
                        help="a = YOLO (Ultralytics), c = from-scratch CNN")
        sp.add_argument("--weights", required=True)
        sp.add_argument("--conf", type=float, default=0.25)
        sp.add_argument("--limit", type=int, default=0, help="cap image count (0 = all)")

    e = sub.add_parser("evaluate", help="score a model with the shared mAP metric")
    common(e)
    e.add_argument("--dataset", default=None)
    e.add_argument("--split", default="test", choices=["train", "valid", "test"])
    e.add_argument("--out", default=None, help="write results JSON here")
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("predict", help="draw predictions onto images")
    common(d)
    d.add_argument("--source", required=True, help="image file or folder")
    d.add_argument("--out", default="predictions")
    d.set_defaults(func=cmd_predict)

    c = sub.add_parser("check", help="verify dataset + metric wiring (no torch needed)")
    c.set_defaults(func=cmd_check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
