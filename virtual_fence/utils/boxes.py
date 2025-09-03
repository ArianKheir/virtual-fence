import numpy as np

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=float)

def iou_xyxy(a, b):
    # a, b: [N, 4], [M, 4]
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    N, M = a.shape[0], b.shape[0]
    ious = np.zeros((N, M), dtype=float)
    for i in range(N):
        ax1, ay1, ax2, ay2 = a[i]
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j in range(M):
            bx1, by1, bx2, by2 = b[j]
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = a_area + b_area - inter + 1e-9
            ious[i, j] = inter / union
    return ious

def nms(boxes, scores, iou_thr=0.45):
    # simple NMS
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thr]
    return keep
