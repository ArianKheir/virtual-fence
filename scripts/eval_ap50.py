import argparse, json, os
import cv2
from pathlib import Path
from virtual_fence.eval.ap50 import load_yolo_labels, ap50

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='images dir (to read sizes)')
    ap.add_argument('--labels', required=True, help='ground-truth labels dir (YOLO format)')
    ap.add_argument('--pred', required=True, help='predictions dir (txt per image: x1 y1 x2 y2 score)')
    args = ap.parse_args()

    imgdir = Path(args.images)
    labdir = Path(args.labels)
    preddir = Path(args.pred)

    # Build shape lookup
    shapes = {}
    for imgp in imgdir.glob('*.*'):
        im = cv2.imread(str(imgp))
        if im is None: continue
        H, W = im.shape[:2]
        shapes[imgp.stem] = (H, W)

    gt = load_yolo_labels(labdir, shapes)

    preds = {}
    for predp in preddir.glob('*.txt'):
        img_id = predp.stem
        boxes = []
        for ln in predp.read_text().strip().splitlines():
            x1,y1,x2,y2,sc = map(float, ln.strip().split()[:5])
            boxes.append(([x1,y1,x2,y2], sc))
        preds[img_id] = boxes

    score = ap50(gt, preds, iou_thr=0.5)
    print(f'AP50: {score:.4f}')

if __name__ == '__main__':
    main()
