from .base import BaseDetector
import numpy as np
import cv2
from typing import List
import time

class YOLODetector(BaseDetector):
    def __init__(self, weights='yolov8n.pt', conf=0.25, iou=0.45, imgsz=640, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model.fuse()  # speed
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device  # ultralytics will choose automatically if None
        self.person_class = 0  # COCO

    def detect(self, frame) -> list:
        # Run predict on a single frame
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False, device=self.device)[0]
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), cf, c in zip(xyxy, confs, clss):
                if c == self.person_class:
                    detections.append([float(x1),float(y1),float(x2),float(y2), float(cf), int(c)])
        return detections
