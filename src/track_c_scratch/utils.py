"""Track C — decode raw grid predictions into boxes, IoU, and greedy NMS."""

import torch


def decode_predictions(pred, grid_size=8, conf_thresh=0.3):
    """Decode one image's raw output (5+C, S, S) into [x1, y1, x2, y2, score, cls_id] boxes
    (all coordinates normalized to [0, 1] image space)."""
    S = grid_size
    pred_txty = torch.sigmoid(pred[0:2])
    pred_twth = torch.sigmoid(pred[2:4])
    pred_obj = torch.sigmoid(pred[4])
    pred_cls = torch.softmax(pred[5:], dim=0)

    boxes = []
    for gj in range(S):
        for gi in range(S):
            obj = pred_obj[gj, gi].item()
            if obj < conf_thresh:
                continue
            tx, ty = pred_txty[0, gj, gi].item(), pred_txty[1, gj, gi].item()
            w, h = pred_twth[0, gj, gi].item(), pred_twth[1, gj, gi].item()
            xc = (gi + tx) / S
            yc = (gj + ty) / S
            cls_id = int(torch.argmax(pred_cls[:, gj, gi]).item())
            cls_score = pred_cls[cls_id, gj, gi].item()
            score = obj * cls_score
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            boxes.append([x1, y1, x2, y2, score, cls_id])
    return boxes


def iou_xyxy(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, iou_thresh=0.4):
    """Greedy per-class NMS. boxes: list of [x1,y1,x2,y2,score,cls_id]."""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if b[5] != best[5] or iou_xyxy(b, best) < iou_thresh]
    return keep


def yolo_to_xyxy(xc, yc, w, h):
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
