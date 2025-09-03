# ONNXRuntime inference for YOLOv8 (person class only). Uses OpenCV pre/post.
import argparse, cv2, numpy as np, onnxruntime as ort, time, os

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def nms(boxes, scores, iou_thr=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        if len(idxs) == 1: break
        iou = iou_xyxy(boxes[i], boxes[idxs[1:]])[0]
        idxs = idxs[1:][iou < iou_thr]
    return keep

def iou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1: a = a[None, :]
    b = np.asarray(b, dtype=float)
    if b.ndim == 1: b = b[None, :]
    ious = np.zeros(b.shape[0], dtype=float)
    ax1, ay1, ax2, ay2 = a[0]
    a_area = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    for j in range(b.shape[0]):
        bx1, by1, bx2, by2 = b[j]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
        inter = iw*ih
        b_area = max(0.0, bx2-bx1) * max(0.0, by2-by1)
        ious[j] = inter / (a_area + b_area - inter + 1e-9)
    return ious

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--zone', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.weights, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out = cv2.VideoWriter(args.output, fourcc, fps, (W,H))
    x1,y1,x2,y2 = map(int, args.zone.split(','))
    count = 0; counted_ids=set(); next_id=0
    tracks = {}

    while True:
        ret, frame = cap.read(); if not ret: break
        img, r, (dw,dh) = letterbox(frame, new_shape=args.imgsz)
        x = img.transpose(2,0,1)[None].astype(np.float32) / 255.0
        preds = sess.run(None, {in_name: x})[0]  # [bs, no, 85] for COCO
        pred = preds[0]
        boxes = pred[:, :4]; scores = pred[:,4]; cls = pred[:,5:]
        cls_id = np.argmax(cls, axis=1); cls_score = np.max(cls, axis=1)
        obj = scores * cls_score
        keep = obj > args.conf
        boxes, obj, cls_id = boxes[keep], obj[keep], cls_id[keep]
        # xywh to xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:,0] = boxes[:,0] - boxes[:,2]/2
        xyxy[:,1] = boxes[:,1] - boxes[:,3]/2
        xyxy[:,2] = boxes[:,0] + boxes[:,2]/2
        xyxy[:,3] = boxes[:,1] + boxes[:,3]/2

        # scale back
        xyxy[:,[0,2]] -= dw; xyxy[:,[1,3]] -= dh
        xyxy /= r

        # filter person class (0)
        m = cls_id == 0
        xyxy = xyxy[m]; obj = obj[m]

        # naive tracking: assign ids by proximity
        ids=[]; vis_boxes=[]
        for i, b in enumerate(xyxy):
            cx = (b[0]+b[2])/2; cy=(b[1]+b[3])/2
            match = None; best=1e9
            for tid, t in tracks.items():
                tx,ty = t['centroid']
                d = abs(tx-cx)+abs(ty-cy)
                if d<best and d<80: best=d; match=tid
            if match is None:
                match = next_id; next_id+=1
            tracks[match]={'centroid':(cx,cy),'box':b,'score':float(obj[i])}
            ids.append(match); vis_boxes.append([*b, float(obj[i])])

        # count
        for tid, t in tracks.items():
            cx,cy = t['centroid']
            if tid in counted_ids: continue
            if x1<=cx<=x2 and y1<=cy<=y2:
                counted_ids.add(tid); count+=1

        # draw
        for i,b in enumerate(vis_boxes):
            cv2.rectangle(frame,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),2)
            cv2.putText(frame, f'id:{ids[i]} {b[4]:.2f}',(int(b[0]),max(0,int(b[1]-6))),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2); cv2.putText(frame,f'Count:{count}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        out.write(frame)
    cap.release(); out.release(); print('Done')

if __name__ == '__main__':
    main()
