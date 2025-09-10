# Raspberry Pi–optimized ONNXRuntime inference for YOLOv8 (person-focused)
# - CPUExecutionProvider + aggressive graph opts
# - Minimal allocations per frame (preallocated input tensor)
# - Vectorized NMS in NumPy (or fast path if model already exports NMS)
# - Optional detect-interval to skip detector on intermediate frames
# - Lightweight centroid tracker with gating + short persistence
# - Optional GStreamer hardware H.264 encoding pipeline on Pi
#
# Usage (Pi-friendly):
#   python inference_onnx_pi.py --weights yolov8n.onnx --input input.mp4 --output out.mp4 \
#       --zone 244,91,767,431 --imgsz 416 --conf 0.3 --iou 0.5 \
#       --detect-interval 2 --ct-min-hits 2 --ct-max-disappeared 30
#
import argparse, os, time, sys
import cv2
import numpy as np
import onnxruntime as ort

# ----------------------------
# Utils: pre/post processing
# ----------------------------
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """
    Resize + pad to square keeping ratio. Returns image, ratio (r_w, r_h), and padding (dw, dh).
    """
    shape = im.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # ratios to map back
    r_w = new_unpad[0] / (shape[1] + 1e-9)
    r_h = new_unpad[1] / (shape[0] + 1e-9)
    return im, (r_w, r_h), (left, top)

def xywh2xyxy(x):
    # x: [..., 4] in (cx,cy,w,h)
    y = np.empty_like(x)
    y[...,0] = x[...,0] - x[...,2] / 2
    y[...,1] = x[...,1] - x[...,3] / 2
    y[...,2] = x[...,0] + x[...,2] / 2
    y[...,3] = x[...,1] + x[...,3] / 2
    return y

def nms_numpy(boxes, scores, iou_thr=0.5, top_k=300):
    """
    Pure NumPy NMS. boxes: [N,4], scores:[N]
    Returns indices kept.
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int32)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)).astype(np.float32)
    order = scores.argsort()[::-1]
    if top_k is not None:
        order = order[:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

def decode_yolov8(raw, conf_thr=0.25, iou_thr=0.5, class_id=None, max_det=300):
    """
    Accepts YOLOv8 ONNX outputs in either shape:
      - (1, num_boxes, 4+nc)   or
      - (1, 4+nc, num_boxes)
    Returns: boxes [N,4] in xyxy on letterboxed image scale, scores [N], classes [N]
    """
    if isinstance(raw, (list, tuple)):
        out = raw[0]
    else:
        out = raw
    arr = np.array(out)
    if arr.ndim != 3:
        raise ValueError(f'Unexpected YOLO output shape: {arr.shape}')
    if arr.shape[1] in (4,5,6):  # unlikely, but guard
        arr = arr.transpose(0,2,1)
    if arr.shape[1] > arr.shape[2]:  # (1, 84, 8400) -> (1, 8400, 84)
        arr = arr.transpose(0,2,1)
    # now arr: (1, num_boxes, 4+nc)
    arr = arr[0]
    num_cols = arr.shape[1]
    if num_cols < 6:
        raise ValueError('Model output must be [x,y,w,h,classes...]')
    boxes_cxcywh = arr[:, :4]
    cls_scores = arr[:, 4:]  # [N, nc]
    if class_id is None:
        cls_idx = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
        classes = cls_idx.astype(np.int32)
    else:
        if class_id < 0 or class_id >= cls_scores.shape[1]:
            raise ValueError(f'class_id {class_id} out of range [0,{cls_scores.shape[1]-1}]')
        scores = cls_scores[:, class_id]
        classes = np.full((cls_scores.shape[0],), class_id, dtype=np.int32)
    # filter by conf
    m = scores >= conf_thr
    if not np.any(m):
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    boxes_cxcywh = boxes_cxcywh[m]
    scores = scores[m]
    classes = classes[m]
    # to xyxy
    boxes = xywh2xyxy(boxes_cxcywh)
    # NMS (single-class or per-class unified since we filter one class)
    keep = nms_numpy(boxes, scores, iou_thr=iou_thr, top_k=max_det)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32), classes[keep].astype(np.int32)

# ----------------------------
# Lightweight centroid tracker
# ----------------------------
class Track:
    __slots__ = ("id","box","cx","cy","hits","miss","score")
    def __init__(self, tid, box, score):
        self.id = tid
        self.box = box.astype(np.float32)
        self.cx = float((box[0]+box[2])/2)
        self.cy = float((box[1]+box[3])/2)
        self.hits = 1
        self.miss = 0
        self.score = float(score)

class CentroidTracker:
    def __init__(self, max_disappeared=30, min_hits=2, max_distance=80.0, min_iou=0.1):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = int(max_disappeared)
        self.min_hits = int(min_hits)
        self.max_distance = float(max_distance)
        self.min_iou = float(min_iou)

    @staticmethod
    def _iou(a, b):
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        w = max(0.0, xx2-xx1); h = max(0.0, yy2-yy1)
        inter = w*h
        ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) - inter + 1e-9
        return inter / ua

    def update(self, boxes, scores):
        """
        boxes: [N,4], scores:[N]
        Returns dict {id: (box[4], score)}
        """
        # if no detections: decay and keep alive a short while (no prediction)
        if boxes.shape[0] == 0:
            to_del = []
            for tid, t in self.tracks.items():
                t.miss += 1
                if t.miss > self.max_disappeared:
                    to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]
            return {tid: (t.box, t.score) for tid, t in self.tracks.items() if t.hits >= self.min_hits and t.miss == 0}

        # build cost by L1 centroid distance; gate by distance or IoU
        centroids = np.vstack(((boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2)).T
        unmatched_dets = list(range(boxes.shape[0]))
        matched_det_for_track = {}

        # simple greedy (few objects): fine for Pi
        for tid, t in list(self.tracks.items()):
            # find nearest detection that passes gate
            dists = np.abs(centroids[:,0] - t.cx) + np.abs(centroids[:,1] - t.cy)
            idx_sorted = np.argsort(dists)
            assigned = False
            for j in idx_sorted:
                if j not in unmatched_dets:
                    continue
                dist_ok = dists[j] <= self.max_distance
                iou_ok = self._iou(t.box, boxes[j]) >= self.min_iou
                if dist_ok or iou_ok:
                    matched_det_for_track[tid] = j
                    unmatched_dets.remove(j)
                    assigned = True
                    break
            if not assigned:
                t.miss += 1
            else:
                j = matched_det_for_track[tid]
                t.box = boxes[j].astype(np.float32)
                t.cx = float(centroids[j,0]); t.cy = float(centroids[j,1])
                t.score = float(scores[j])
                t.hits += 1
                t.miss = 0

        # new tracks for the rest (avoid tiny conf)
        for j in unmatched_dets:
            b = boxes[j]
            sc = float(scores[j])
            nt = Track(self.next_id, b, sc)
            self.tracks[self.next_id] = nt
            self.next_id += 1

        # output confirmed, currently visible tracks
        out = {}
        for tid, t in self.tracks.items():
            if t.hits >= self.min_hits and t.miss == 0:
                out[tid] = (t.box.copy(), t.score)
        return out

# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser('Raspberry Pi–optimized YOLOv8 ONNX inference')
    ap.add_argument('--weights', required=True, help='Path to .onnx model')
    ap.add_argument('--input', required=True, help='Video file path or camera index (int)')
    ap.add_argument('--output', required=True, help='Output video file path')
    ap.add_argument('--zone', required=True, help='x1,y1,x2,y2 for counting region')
    ap.add_argument('--imgsz', type=int, default=416, help='Inference size (square)')
    ap.add_argument('--conf', type=float, default=0.30, help='Confidence threshold')
    ap.add_argument('--iou', type=float, default=0.50, help='NMS IoU threshold')
    ap.add_argument('--class-id', type=int, default=0, help='Class to keep (0=person for COCO)')
    ap.add_argument('--max-det', type=int, default=300, help='Max detections before NMS top_k')
    ap.add_argument('--detect-interval', type=int, default=2, help='Run detector every N frames (>=1)')
    # Tracker (very light)
    ap.add_argument('--ct-min-hits', type=int, default=2)
    ap.add_argument('--ct-max-disappeared', type=int, default=30)
    ap.add_argument('--ct-max-distance', type=float, default=80.0)
    ap.add_argument('--ct-min-iou', type=float, default=0.10)
    # ORT / threads
    ap.add_argument('--intra-threads', type=int, default=max(1, (os.cpu_count() or 4)//2))
    ap.add_argument('--inter-threads', type=int, default=1)
    ap.add_argument('--opt-level', type=int, default=3, choices=[0,1,2,3], help='ORT graph optimization level')
    ap.add_argument('--providers', nargs='*', default=['CPUExecutionProvider'])
    # Video I/O
    ap.add_argument('--use-gstreamer', action='store_true', help='Use GStreamer pipeline for output (Pi HW H.264)')
    ap.add_argument('--gst-out', type=str, default='', help='Custom GStreamer pipeline for output writer')
    ap.add_argument('--fps', type=float, default=0.0, help='Force FPS for writer (0=from input)')
    ap.add_argument('--fourcc', type=str, default='mp4v', help='Fallback FOURCC if not using GStreamer')
    ap.add_argument('--show-fps', action='store_true', help='Draw FPS on frames')
    return ap.parse_args()

def build_session(weights, providers, intra, inter, opt_level):
    sess_options = ort.SessionOptions()
    levels = [
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ]
    sess_options.graph_optimization_level = levels[opt_level]
    sess_options.intra_op_num_threads = int(intra)
    sess_options.inter_op_num_threads = int(inter)
    # Make runs more deterministic on small devices
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(weights, sess_options=sess_options, providers=providers)
    input_name = sess.get_inputs()[0].name
    return sess, input_name

def open_video(input_arg):
    try:
        cam_index = int(input_arg)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(input_arg)
    return cap

def make_writer(out_path, w, h, fps, use_gst=False, gst_pipeline='', fourcc='mp4v'):
    if use_gst:
        if not gst_pipeline:
            # Generic H.264 HW encode on Pi via v4l2h264enc (modify if missing on your image)
            gst_pipeline = (
                f'appsrc ! videoconvert ! v4l2h264enc extra-controls="encode,frame_level_rate_control_enable=1,video_bitrate=4000000" ! '
                f'h264parse ! mp4mux ! filesink location={out_path}'
            )
        return cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, (w, h))
    else:
        four = cv2.VideoWriter_fourcc(*fourcc)
        return cv2.VideoWriter(out_path, four, fps, (w, h))

def draw_boxes(frame, tracks_dict, zone, count, show_fps=False, fps_val=0.0):
    for tid, (box, score) in tracks_dict.items():
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'ID:{tid} {score:.2f}', (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,220,50), 2)
    x1,y1,x2,y2 = zone
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.putText(frame, f'Count:{count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if show_fps:
        cv2.putText(frame, f'{fps_val:.1f} FPS', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

def main():
    args = parse_args()

    # OpenCV threading – avoid fighting ORT on small CPUs
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Parse zone
    zx1,zy1,zx2,zy2 = map(int, args.zone.split(','))
    zone = (zx1,zy1,zx2,zy2)

    # Video input
    cap = open_video(args.input)
    if not cap.isOpened():
        print('ERROR: cannot open input', file=sys.stderr)
        sys.exit(1)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps <= 0 or np.isnan(in_fps):
        in_fps = 25.0
    out_fps = args.fps if args.fps > 0 else in_fps

    # Output
    writer = make_writer(args.output, in_w, in_h, out_fps, args.use_gstreamer, args.gst_out, args.fourcc)
    if not writer.isOpened():
        print('WARNING: fallback to software writer (fourcc).')
        writer = make_writer(args.output, in_w, in_h, out_fps, False, '', args.fourcc)
        if not writer.isOpened():
            print('ERROR: cannot open VideoWriter', file=sys.stderr)
            sys.exit(2)

    # ORT session
    sess, input_name = build_session(args.weights, args.providers, args.intra_threads, args.inter_threads, args.opt_level)

    # Preallocate input tensor
    imgsz = int(args.imgsz)
    blob = np.empty((1, 3, imgsz, imgsz), dtype=np.float32)

    # Tracker & counting
    tracker = CentroidTracker(max_disappeared=args.ct_max_disappeared,
                              min_hits=args.ct_min_hits,
                              max_distance=args.ct_max_distance,
                              min_iou=args.ct_min_iou)
    counted = set()
    total_count = 0

    frame_i = 0
    t0 = time.time()
    det_boxes_prev = np.zeros((0,4), dtype=np.float32)
    det_scores_prev = np.zeros((0,), dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        orig = frame

        run_detector = (frame_i % max(1, args.detect_interval) == 1)
        if run_detector:
            # preprocess (reuse blob)
            img, (rw, rh), (dw, dh) = letterbox(orig, new_shape=imgsz)
            # HWC BGR -> CHW, normalize to 0..1 into preallocated blob
            # Avoid extra allocations:
            np.divide(img.transpose(2,0,1), 255.0, out=blob[0], dtype=np.float32)

            # inference
            out = sess.run(None, {input_name: blob})

            # decode
            boxes_ltr, scores, classes = decode_yolov8(out, conf_thr=args.conf, iou_thr=args.iou,
                                                        class_id=args.class_id, max_det=args.max_det)

            # map boxes back to original frame
            if boxes_ltr.shape[0] > 0:
                # reverse letterbox: subtract pad, divide by ratio
                boxes = boxes_ltr.copy()
                boxes[:, [0,2]] -= dw
                boxes[:, [1,3]] -= dh
                boxes[:, [0,2]] /= (rw + 1e-9)
                boxes[:, [1,3]] /= (rh + 1e-9)
                # clip
                boxes[:,0::2] = np.clip(boxes[:,0::2], 0, orig.shape[1]-1)
                boxes[:,1::2] = np.clip(boxes[:,1::2], 0, orig.shape[0]-1)
            else:
                boxes = boxes_ltr
            det_boxes_prev = boxes
            det_scores_prev = scores
        else:
            boxes = det_boxes_prev
            scores = det_scores_prev

        # update tracker only when we "saw" detections (on detect frames)
        if boxes.shape[0] > 0 and run_detector:
            tracks = tracker.update(boxes, scores)
        else:
            # decay miss for all tracks but don't predict (keeps CPU low)
            tracks = tracker.update(np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32))

        # counting in zone (use track center; count once)
        for tid, (b, sc) in tracks.items():
            cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
            if tid in counted:
                continue
            if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]:
                counted.add(tid); total_count += 1

        # draw & write
        draw_boxes(orig, tracks, zone, total_count, args.show_fps,
                   fps_val=(frame_i / max(1e-6, time.time() - t0)))
        writer.write(orig)

    cap.release(); writer.release()
    print(f'Done. Frames: {frame_i}, Avg FPS: {frame_i / max(1e-6, time.time() - t0):.2f}')

if __name__ == '__main__':
    main()
