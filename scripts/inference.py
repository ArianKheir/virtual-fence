import argparse, os, time
import cv2
import numpy as np

from virtual_fence.detectors.yolo import YOLODetector
from virtual_fence.detectors.vlm_owl import VLMOWLDetector
from virtual_fence.detectors.rcnn_resnet50_fpn_v2 import RCNNResNet50FPNv2Detector  # NEW
from virtual_fence.tracking.centroid_tracker import CentroidTracker
from virtual_fence.tracking.sort_tracker import SORT
from virtual_fence.counting.zone import ZoneCounter
from virtual_fence.video.overlay import draw_overlays

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='input video path')
    ap.add_argument('--output', required=True, help='output mp4 path')
    ap.add_argument('--zone', required=True, help='x1,y1,x2,y2')
    ap.add_argument('--detector', required=True, choices=['yolo','vlm','rcnn'])  # UPDATED
    ap.add_argument('--tracker', default='sort', choices=['centroid','sort'])
    ap.add_argument('--weights', default=None, help='weights path (YOLO only)')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--vlm-prompts', default='person,pedestrian',
                    help='Comma-separated list of OWLv2 prompts')
    ap.add_argument('--text-thr', type=float, default=0.25,
                    help='OWLv2 text_threshold (similarity threshold)')
    ap.add_argument('--ct-max-disappeared', type=int, default=30)
    ap.add_argument('--ct-max-distance', type=float, default=100.0)
    ap.add_argument('--ct-min-iou', type=float, default=0.1)
    ap.add_argument('--ct-min-hits', type=int, default=2)
    ap.add_argument('--ct-no-predict', action='store_true')
    ap.add_argument('--ct-predict-miss-horizon', type=int, default=1,
                    help='Predict for at most this many consecutive miss frames')
    ap.add_argument('--ct-vel-damping', type=float, default=0.6,
                    help='Velocity damping factor during prediction (0..1)')
    ap.add_argument('--ct-max-vel-px', type=float, default=50.0,
                    help='Clamp per-axis velocity in pixels/frame')
    ap.add_argument('--ct-draw-miss-budget', type=int, default=8,
                    help='Only draw a track if last seen within this many frames')
    ap.add_argument('--ct-reacquire-iou', type=float, default=0.2,
                    help='Stronger IoU required to re-acquire after a miss')
    ap.add_argument('--ct-reg-min-conf', type=float, default=0.25,
                    help='Minimum det confidence to start a new track')

    return ap.parse_args()

def main():
    args = parse_args()
    x1,y1,x2,y2 = map(int, args.zone.split(','))
    zone = (x1,y1,x2,y2)

    # Detector
    if args.detector == 'yolo':
        det = YOLODetector(weights=args.weights or 'yolov8n.pt', conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    elif args.detector == 'vlm':
        prompts = [p.strip() for p in args.vlm_prompts.split(',') if p.strip()]
        det = VLMOWLDetector(
            prompts=prompts,
            conf=args.conf,
            text_threshold=args.text_thr
        )
    else:  # rcnn
        det = RCNNResNet50FPNv2Detector(conf=args.conf)

    # Tracker
    if args.tracker == 'centroid':
        tracker = CentroidTracker(
            max_disappeared=args.ct_max_disappeared,
            max_distance=args.ct_max_distance,
            min_iou=args.ct_min_iou,
            min_hits=args.ct_min_hits,
            use_prediction=(not args.ct_no_predict),
            predict_miss_horizon=args.ct_predict_miss_horizon,
            vel_damping=args.ct_vel_damping,
            max_vel_px=args.ct_max_vel_px,
            draw_miss_budget=args.ct_draw_miss_budget,
            reacquire_iou=args.ct_reacquire_iou,
            reg_min_conf=args.ct_reg_min_conf
        )
    else:
        tracker = SORT(max_age=30, min_hits=3, iou_threshold=0.2)

    # IO
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open input video: {args.input}')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out = cv2.VideoWriter(args.output, fourcc, fps_in, (W,H))

    zc = ZoneCounter(zone)
    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = det.detect(frame)  # [x1,y1,x2,y2,conf,cls]
        rects = [[d[0], d[1], d[2], d[3]] for d in detections]
        confs = [d[4] for d in detections]
        dets5 = np.array([[*r, confs[i]] for i, r in enumerate(rects)], dtype=float)
        if dets5.size == 0:
            dets5 = np.zeros((0, 5), dtype=float)
        # Track
        if args.tracker == 'centroid':
            # NEW: feed scores, and get id->(cx,cy,box) directly
            objs = tracker.update(rects, confs)
            ids, boxes = [], []
            for tid, (cx, cy, box) in objs.items():
                ids.append(tid)
                # ensure [x1,y1,x2,y2,conf]
                if len(box) == 4:
                    boxes.append([box[0], box[1], box[2], box[3], 1.0])
                else:
                    boxes.append(box[:5])
        else:
            res = tracker.update(dets5)
            ids = []
            boxes = []
            for tid, (cx,cy,box) in res.items():
                ids.append(tid)
                # map box + conf estimate (best matching original det if available)
                conf = 1.0
                if len(rects)>0:
                    dists = [ (abs((r[0]+r[2])/2.0 - cx) + abs((r[1]+r[3])/2.0 - cy), i) for i,r in enumerate(rects) ]
                    _, ridx = min(dists, key=lambda t:t[0])
                    conf = confs[ridx]
                boxes.append([box[0],box[1],box[2],box[3], conf])

        # Count
        tracked_centroids = {
            ids[i]: ((boxes[i][0] + boxes[i][2]) / 2.0, (boxes[i][1] + boxes[i][3]) / 2.0)
            for i in range(len(ids))
        }
        total = zc.update(tracked_centroids)

        # FPS
        frame_idx += 1
        elapsed = time.time() - t0
        fps = frame_idx / max(1e-9, elapsed)

        # Draw (sanitize boxes to plain Python floats to keep OpenCV happy)
        clean_boxes = []
        for b in boxes:
            if len(b) >= 5:
                x1 = float(b[0]); y1 = float(b[1]); x2 = float(b[2]); y2 = float(b[3]); conf = float(b[4])
                clean_boxes.append([x1, y1, x2, y2, conf])
        overlay = draw_overlays(frame.copy(), clean_boxes, ids, zone, total, fps=fps)
        out.write(overlay)

    cap.release(); out.release()
    print(f'Output written to: {args.output} | Count: {zc.total}')

if __name__ == '__main__':
    main()