"""Track C — YOLOv1-style sum-squared-error loss for the from-scratch detector."""

import torch
import torch.nn as nn

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5


class ScratchDetectorLoss(nn.Module):
    """Coordinate + objectness + no-objectness + class loss, cell-masked by target objectness."""

    def forward(self, pred, target):
        # pred, target: (B, 5+C, S, S)
        pred_txty = torch.sigmoid(pred[:, 0:2])
        pred_twth = torch.sigmoid(pred[:, 2:4])
        pred_obj = torch.sigmoid(pred[:, 4])
        pred_cls = pred[:, 5:]

        target_txty = target[:, 0:2]
        target_twth = target[:, 2:4]
        target_obj = target[:, 4]
        target_cls = target[:, 5:]

        obj_mask = target_obj
        noobj_mask = 1.0 - obj_mask

        coord_loss = LAMBDA_COORD * (
            (obj_mask.unsqueeze(1) * (pred_txty - target_txty) ** 2).sum()
            + (obj_mask.unsqueeze(1) * (pred_twth - target_twth) ** 2).sum()
        )
        obj_loss = (obj_mask * (pred_obj - target_obj) ** 2).sum()
        noobj_loss = LAMBDA_NOOBJ * (noobj_mask * pred_obj ** 2).sum()
        cls_loss = (obj_mask.unsqueeze(1) * (pred_cls.sigmoid() - target_cls) ** 2).sum()

        total = coord_loss + obj_loss + noobj_loss + cls_loss
        return total / pred.shape[0]


if __name__ == "__main__":
    from model import ScratchDetector

    model = ScratchDetector()
    criterion = ScratchDetectorLoss()
    dummy_img = torch.randn(2, 3, 128, 128)
    dummy_target = torch.zeros(2, 7, 8, 8)
    dummy_target[0, 4, 3, 3] = 1.0
    dummy_target[0, 6, 3, 3] = 1.0

    pred = model(dummy_img)
    loss = criterion(pred, dummy_target)
    print(f"loss: {loss.item():.4f}")
