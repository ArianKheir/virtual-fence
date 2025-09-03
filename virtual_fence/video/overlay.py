import cv2
import time
import math

def _to_float(v):
    # Accept Python numbers, NumPy scalars, Torch scalars, and length-1 containers
    try:
        if hasattr(v, "item"):
            v = v.item()
        elif isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return float(v)
    except Exception:
        return float("nan")

def _to_int(v):
    f = _to_float(v)
    if not math.isfinite(f):
        return 0
    return int(round(f))

def _as_xyxy(box):
    # supports [x1,y1,x2,y2] or [x,y,w,h]; returns Python ints
    x1 = _to_float(box[0])
    y1 = _to_float(box[1])
    x2 = _to_float(box[2])
    y2 = _to_float(box[3])
    # Convert xywh → xyxy if needed

    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return 0, 0, 0, 0
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)

    return _to_int(x1), _to_int(y1), _to_int(x2), _to_int(y2)


def draw_overlays(frame, boxes, ids, zone, count, fps=None):
    # boxes: list of (x1,y1,x2,y2,conf)
    for i, b in enumerate(boxes):
        # Normalize every coord/label to basic Python types
        if len(b) < 4:
            continue
        conf = _to_float(b[4]) if len(b) >= 5 else 1.0
        x1, y1, x2, y2 = _as_xyxy(b)
        color = (0, 255, 0)
        pt1 = (_to_int(x1), _to_int(y1))
        pt2 = (_to_int(x2), _to_int(y2))
        try:
            cv2.rectangle(frame, pt1, pt2, color, 2)
        except Exception:
            # Skip any pathological box rather than crashing the whole run
            continue
        tid_text = f'id:{int(ids[i])} ' if i < len(ids) else ''
        label = f'{tid_text}conf:{conf:.2f}'
        cv2.putText(frame, label, (pt1[0], max(0, pt1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    zx1, zy1, zx2, zy2 = map(_to_int, zone)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)
    cv2.putText(frame, "ZONE", (zx1, max(0, zy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f'Count: {count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    if fps is not None and math.isfinite(float(fps)):
        cv2.putText(frame, f'FPS: {float(fps):.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame
