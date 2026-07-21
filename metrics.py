"""Shared detection metrics — one mAP implementation for all three tracks.

Why this exists: the repo evaluates each track with a different tool. Track A uses
Ultralytics' internal validator, Track B uses `supervision.metrics`, and Track C
(src/track_c_scratch/evaluate.py) reports only precision/recall/F1 at a single
confidence threshold. Those numbers are not comparable, so the cross-track table
in the report was comparing apples to oranges.

This module computes VOC-style AP (all-point interpolation, the same definition
COCO uses) from plain numpy arrays, so every track can be scored the same way:
feed it whatever boxes the model produced, in absolute xyxy pixel coordinates.

Self-check:
    python metrics.py
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Head", "Person"]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. boxes_a (N,4), boxes_b (M,4), both xyxy -> (N,M)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]  # (N,1,4)
    b = boxes_b[None, :, :]  # (1,M,4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)

    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP: make precision monotonically decreasing, then
    integrate over recall. This is the COCO/VOC2010+ definition."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Sweep right-to-left so each precision becomes the max precision at >= that recall.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision_for_class(
    pred_boxes: list[np.ndarray],
    pred_scores: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_threshold: float,
) -> float:
    """AP for a single class at a single IoU threshold.

    Each list is per-image and must be the same length. A ground-truth box may be
    matched at most once; extra detections onto an already-matched box count as
    false positives, which is what stops duplicate-box spam from inflating recall.
    """
    n_gt = int(sum(len(g) for g in gt_boxes))
    if n_gt == 0:
        return float("nan")  # class absent from this split — excluded from the mean

    # Flatten every detection across images, tagged with its image index.
    img_ids, scores, boxes = [], [], []
    for img_id, (b, s) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(b, s):
            img_ids.append(img_id)
            scores.append(score)
            boxes.append(box)

    if not boxes:
        return 0.0

    img_ids = np.array(img_ids)
    scores = np.array(scores, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)

    order = np.argsort(-scores)  # highest confidence first
    img_ids, boxes = img_ids[order], boxes[order]

    matched = {i: np.zeros(len(g), dtype=bool) for i, g in enumerate(gt_boxes)}
    tp = np.zeros(len(boxes), dtype=np.float32)
    fp = np.zeros(len(boxes), dtype=np.float32)

    for k, (img_id, box) in enumerate(zip(img_ids, boxes)):
        gt = gt_boxes[img_id]
        if len(gt) == 0:
            fp[k] = 1
            continue

        ious = iou_matrix(box[None, :], gt)[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not matched[img_id][best]:
            tp[k] = 1
            matched[img_id][best] = True
        else:
            fp[k] = 1

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return _average_precision(recall, precision)


def evaluate_detections(
    predictions: list[dict],
    ground_truths: list[dict],
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """Compute mAP50, mAP50-95 and per-class AP.

    predictions:   per-image dicts {"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,)}
    ground_truths: per-image dicts {"boxes": (M,4) xyxy, "labels": (M,)}

    Coordinates must be absolute pixels in the *same* space for preds and GT.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths ({len(ground_truths)}) "
            "must describe the same images, in the same order"
        )

    thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50:0.05:0.95, the COCO sweep
    per_class_ap = {}

    for class_id, class_name in enumerate(class_names):
        p_boxes, p_scores, g_boxes = [], [], []
        for pred, gt in zip(predictions, ground_truths):
            p_lab = np.asarray(pred["labels"]).reshape(-1)
            g_lab = np.asarray(gt["labels"]).reshape(-1)
            p_box = np.asarray(pred["boxes"], dtype=np.float32).reshape(-1, 4)
            g_box = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            p_scr = np.asarray(pred["scores"], dtype=np.float32).reshape(-1)

            p_mask, g_mask = p_lab == class_id, g_lab == class_id
            p_boxes.append(p_box[p_mask])
            p_scores.append(p_scr[p_mask])
            g_boxes.append(g_box[g_mask])

        aps = [average_precision_for_class(p_boxes, p_scores, g_boxes, t) for t in thresholds]
        per_class_ap[class_name] = {
            "AP50": aps[0],
            "AP50_95": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan"),
        }

    valid = [v for v in per_class_ap.values() if not np.isnan(v["AP50"])]
    return {
        "mAP50": float(np.mean([v["AP50"] for v in valid])) if valid else float("nan"),
        "mAP50_95": float(np.mean([v["AP50_95"] for v in valid])) if valid else float("nan"),
        "per_class": per_class_ap,
    }


def yolo_labels_to_xyxy(label_path, img_w: int, img_h: int) -> dict:
    """Read a YOLO .txt label file into absolute-pixel xyxy ground truth."""
    boxes, labels = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
            boxes.append([
                (xc - w / 2) * img_w, (yc - h / 2) * img_h,
                (xc + w / 2) * img_w, (yc + h / 2) * img_h,
            ])
            labels.append(cls_id)

    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=int),
    }


if __name__ == "__main__":
    # A perfect prediction must score 1.0; a shifted/degraded one must score lower.
    gt = [{"boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]]), "labels": np.array([0, 1])}]

    perfect = [{
        "boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]]),
        "scores": np.array([0.9, 0.8]),
        "labels": np.array([0, 1]),
    }]
    res = evaluate_detections(perfect, gt)
    assert abs(res["mAP50"] - 1.0) < 1e-6, res
    assert abs(res["mAP50_95"] - 1.0) < 1e-6, res
    print(f"perfect      -> mAP50={res['mAP50']:.4f}  mAP50-95={res['mAP50_95']:.4f}")

    # A duplicate box outranking a still-undetected GT must cost AP. (A duplicate
    # that lands *after* recall is already 1.0 correctly does not — that is VOC AP,
    # not a bug, so the duplicate here is deliberately mid-ranking.)
    gt_two_heads = [{"boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]]), "labels": np.array([0, 0])}]
    dupes = [{
        "boxes": np.array([[10, 10, 50, 50], [11, 11, 51, 51], [60, 60, 90, 90]]),
        "scores": np.array([0.9, 0.85, 0.7]),  # middle box duplicates the first
        "labels": np.array([0, 0, 0]),
    }]
    res_d = evaluate_detections(dupes, gt_two_heads)
    head_ap = res_d["per_class"]["Head"]["AP50"]
    assert head_ap < 1.0, res_d  # expected 0.5*1.0 + 0.5*(2/3) = 0.8333
    print(f"duplicate box -> Head AP50={head_ap:.4f} (penalised, correct)")

    # Wrong class => zero AP for both classes.
    wrong = [{
        "boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]]),
        "scores": np.array([0.9, 0.8]),
        "labels": np.array([1, 0]),
    }]
    res_w = evaluate_detections(wrong, gt)
    assert res_w["mAP50"] == 0.0, res_w
    print(f"wrong class  -> mAP50={res_w['mAP50']:.4f}")

    # An absent class must be excluded from the mean, not scored as 0.
    gt_head_only = [{"boxes": np.array([[10, 10, 50, 50]]), "labels": np.array([0])}]
    pred_head_only = [{
        "boxes": np.array([[10, 10, 50, 50]]), "scores": np.array([0.9]), "labels": np.array([0]),
    }]
    res_a = evaluate_detections(pred_head_only, gt_head_only)
    assert np.isnan(res_a["per_class"]["Person"]["AP50"]), res_a
    assert abs(res_a["mAP50"] - 1.0) < 1e-6, res_a
    print(f"absent class -> mAP50={res_a['mAP50']:.4f} (Person excluded, not zeroed)")

    print("\nAll metric self-checks passed.")
