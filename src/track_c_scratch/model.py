"""Track C — simple CNN grid detector, trained from scratch.

Deliberately small and simple (YOLOv1-lite): a conv backbone with BatchNorm +
Dropout (the specific techniques the assignment mentions for the from-scratch
track), downsampling a 128x128 image to an 8x8 grid. Each grid cell predicts
one box (tx, ty, tw, th) + objectness + class logits.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 2


def conv_block(in_ch, out_ch, dropout=0.1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.MaxPool2d(2),
    )


class ScratchDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, grid_size=8, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        # 128 -> 64 -> 32 -> 16 -> 8 (4 downsampling blocks -> 8x8 grid)
        self.backbone = nn.Sequential(
            conv_block(3, 16, dropout),
            conv_block(16, 32, dropout),
            conv_block(32, 64, dropout),
            conv_block(64, 128, dropout),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 5 + num_classes, kernel_size=1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)  # (B, 5+C, S, S) raw logits


if __name__ == "__main__":
    model = ScratchDetector()
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(out.shape)}")
    print(f"trainable params: {n_params:,}")
