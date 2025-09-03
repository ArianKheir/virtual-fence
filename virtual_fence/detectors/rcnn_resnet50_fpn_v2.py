# virtual_fence/detectors/rcnn_resnet50_fpn_v2.py
from .base import BaseDetector
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import cv2


class RCNNResNet50FPNv2Detector(BaseDetector):
    def __init__(self, conf=0.5, device=None):
        super().__init__()
        self.conf = conf
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model with pretrained weights
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

        # COCO class names (person is index 1)
        self.class_names = ['person']  # We only care about person class

    def detect(self, frame) -> list:
        # Preprocess image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        # Filter detections
        detections = []
        for i in range(len(predictions['boxes'])):
            score = predictions['scores'][i].item()
            label = predictions['labels'][i].item()

            if score >= self.conf and label == 1:  # Only person class (COCO index 1)
                box = predictions['boxes'][i].cpu().numpy()
                x1, y1, x2, y2 = box
                detections.append([float(x1), float(y1), float(x2), float(y2), float(score), 0])  # cls=0 for person

        return detections