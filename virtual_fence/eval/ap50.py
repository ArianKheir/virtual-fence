# Simple AP50 evaluator for a single class (person). Predictions and GT in YOLO format.
import os
import numpy as np
from pathlib import Path
from ..utils.boxes import iou_xyxy

def load_yolo_labels(label_dir, img_shape_lookup):
    # returns dict img_id -> [ (x1,y1,x2,y2) ... ]
    lab = {}
    p = Path(label_dir)
    for txt in p.glob('*.txt'):
        img_id = txt.stem
        lines = txt.read_text().strip().splitlines()
        boxes = []
        if len(lines) == 0:
            lab[img_id] = boxes
            continue
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts[:5])
            if int(cls) != 0:
                continue
            H, W = img_shape_lookup[img_id]
            x = cx * W; y = cy * H; bw = w * W; bh = h * H
            x1 = x - bw/2.0; y1 = y - bh/2.0; x2 = x + bw/2.0; y2 = y + bh/2.0
            boxes.append([x1,y1,x2,y2])
        lab[img_id] = boxes
    return lab

def ap50(ground_truth: dict, predictions: dict, iou_thr=0.5):
    # ground_truth: img_id -> [gt boxes]
    # predictions: img_id -> [ (box, score) ]
    all_scores = []
    all_matches = []
    total_gt = 0
    for img_id, gt in ground_truth.items():
        total_gt += len(gt)
        preds = predictions.get(img_id, [])
        boxes = [p[0] for p in preds]
        scores = [p[1] for p in preds]
        order = np.argsort(scores)[::-1]
        gt_used = np.zeros(len(gt), dtype=bool)
        for idx in order:
            b = boxes[idx]
            best_iou = 0.0; best_j = -1
            for j, g in enumerate(gt):
                if gt_used[j]:
                    continue
                iou = iou_xyxy(b, g)[0,0]
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_iou >= iou_thr:
                all_matches.append(1); gt_used[best_j] = True
            else:
                all_matches.append(0)
            all_scores.append(scores[idx])

    if len(all_scores) == 0:
        return 0.0

    order = np.argsort(all_scores)[::-1]
    tp = np.cumsum(np.array(all_matches)[order] == 1)
    fp = np.cumsum(np.array(all_matches)[order] == 0)
    recalls = tp / max(1, total_gt)
    precisions = tp / np.maximum(1, tp + fp)

    # 11-point interpolated AP (VOC 2007 style)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = 0.0
        if np.any(recalls >= t):
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return float(ap)
