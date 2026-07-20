"""Track C — small-subset dataset loader for the from-scratch grid detector.

Only ~800 train / ~200 val images are sampled on purpose: this track exists to
demonstrate training a CNN detector from scratch (batch norm, dropout) on a
budget, not to reach competitive accuracy. Tracks A/B use the full 14k images.
"""

import sys
import random
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

import cv2
import torch
from torch.utils.data import Dataset

from common.dataset_paths import CLASS_NAMES, NUM_CLASSES  # noqa: E402


def list_image_label_pairs(images_dir, labels_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    pairs = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def read_yolo_labels(label_path):
    """Parse a YOLO-format label file: 'class_id x_center y_center width height' (normalized 0-1)."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = parts
            boxes.append((int(cls_id), float(xc), float(yc), float(w), float(h)))
    return boxes


class HeadPersonSubsetDataset(Dataset):
    """YOLO-format subset dataset, encoded into an SxS grid target for Track C.

    Each grid cell can hold at most 1 object (single-box-per-cell, YOLOv1-lite
    style) -- a deliberate simplification for a from-scratch teaching model.
    If several ground-truth boxes fall in the same cell, only the first one
    kept; the rest are dropped (documented limitation, see report).
    """

    def __init__(self, images_dir, labels_dir, subset_size=800, img_size=128, grid_size=8, seed=42):
        self.pairs = list_image_label_pairs(images_dir, labels_dir)
        rng = random.Random(seed)
        rng.shuffle(self.pairs)
        if subset_size is not None:
            self.pairs = self.pairs[:subset_size]
        self.img_size = img_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        S = self.grid_size
        target = torch.zeros((5 + NUM_CLASSES, S, S), dtype=torch.float32)
        for cls_id, xc, yc, w, h in read_yolo_labels(label_path):
            gi = min(int(xc * S), S - 1)
            gj = min(int(yc * S), S - 1)
            if target[4, gj, gi] == 1.0:
                continue  # cell already occupied -- single-box-per-cell limitation
            target[0, gj, gi] = xc * S - gi  # tx: offset within cell, [0,1]
            target[1, gj, gi] = yc * S - gj  # ty
            target[2, gj, gi] = w            # tw: image-relative width, [0,1]
            target[3, gj, gi] = h            # th
            target[4, gj, gi] = 1.0          # objectness
            target[5 + cls_id, gj, gi] = 1.0  # one-hot class
        return img_t, target


if __name__ == "__main__":
    from common.dataset_paths import TRAIN_IMAGES, TRAIN_LABELS

    ds = HeadPersonSubsetDataset(TRAIN_IMAGES, TRAIN_LABELS, subset_size=20)
    img, target = ds[0]
    print(f"dataset size: {len(ds)}")
    print(f"image tensor: {tuple(img.shape)}, target tensor: {tuple(target.shape)}")
    print(f"objects in sample 0: {int(target[4].sum().item())}")
