"""Track C — train the from-scratch CNN detector on a small image subset.

Usage:
    python train.py
    python train.py --subset-size 1000 --epochs 30
"""

import sys
import json
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_SRC_DIR))

import torch
from torch.utils.data import DataLoader

from dataset import HeadPersonSubsetDataset  # noqa: E402
from model import ScratchDetector  # noqa: E402
from loss import ScratchDetectorLoss  # noqa: E402
from common.dataset_paths import TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS  # noqa: E402


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = HeadPersonSubsetDataset(
        TRAIN_IMAGES, TRAIN_LABELS, subset_size=args.subset_size,
        img_size=args.img_size, grid_size=args.grid_size, seed=42,
    )
    val_ds = HeadPersonSubsetDataset(
        VALID_IMAGES, VALID_LABELS, subset_size=max(100, args.subset_size // 4),
        img_size=args.img_size, grid_size=args.grid_size, seed=123,
    )
    print(f"train images: {len(train_ds)}, val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ScratchDetector(grid_size=args.grid_size, dropout=args.dropout).to(device)
    criterion = ScratchDetectorLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                running += criterion(model(imgs), targets).item() * imgs.size(0)
        val_loss = running / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:>3}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "scratch_detector.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved model + history -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset-size", type=int, default=800)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out-dir", default=str(_THIS_DIR / "runs"))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
