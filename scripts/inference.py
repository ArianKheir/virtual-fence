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
        tracker = CentroidTracker(max_disappeared=30)
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
            objs = tracker.update(rects)  # id->(cx,cy)
            # Build outputs
            ids = []
            boxes = []
            for tid, (cx,cy) in objs.items():
                # find closest rect for viz
                if len(rects)>0:
                    dists = [ (abs((r[0]+r[2])/2.0 - cx) + abs((r[1]+r[3])/2.0 - cy), i) for i,r in enumerate(rects) ]
                    _, ridx = min(dists, key=lambda t:t[0])
                    boxes.append([*rects[ridx], confs[ridx]])
                    ids.append(tid)
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