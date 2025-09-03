# virtual_fence/detectors/vlm_owl.py
# OWLv2 zero-shot detector for "person" only (and synonyms).
# Returns [x1,y1,x2,y2,score,cls] with cls=0.

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# If you still have BaseDetector, keep the inheritance; otherwise remove it.
try:
    from .base import BaseDetector
    _BASE = BaseDetector
except Exception:
    _BASE = object

class VLMOWLDetector(_BASE):
    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16",
        prompts=None,
        conf: float = 0.40,           # a bit higher gives cleaner boxes
        text_threshold: float = 0.30, # similarity threshold
        nms_iou: float = 0.50,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
        # Ensure prompts is a list[str] (never a single comma-joined string)
        default_prompts = ["person", "pedestrian"]  # minimal & stable
        if prompts is None:
            self.prompts = default_prompts
        else:
            if isinstance(prompts, str):
                # Accept "person,pedestrian" -> ["person","pedestrian"]
                prompts = [p.strip() for p in prompts.split(",")]
            self.prompts = [str(p).strip() for p in prompts if str(p).strip()]
            if not self.prompts:
                self.prompts = default_prompts

        self.conf = float(conf)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)

    def detect(self, frame_bgr: np.ndarray) -> list[list[float]]:
        # Convert BGR->RGB
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # OWLv2 expects a *batch* of class lists; use [self.prompts]
        inputs = self.processor(
            images=image_rgb,
            text=[self.prompts],               # <-- IMPORTANT: list-of-list
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [image_rgb.shape[:2]], device=self.device
        )  # (H,W) per image

        try:
            # OWLv2 post-process (scores already per phrase)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.conf,
                text_threshold=self.text_threshold
            )[0]
            boxes = results["boxes"]     # (N,4)
            scores = results["scores"]   # (N,)
            # results["labels"] are phrase indices, but we unify to "person"
        except Exception:
            # Fallback for older transformers (no grounded postprocess)
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.conf, target_sizes=[image_rgb.shape[:2]]
            )[0]
            boxes = results["boxes"]
            scores = results["scores"]

        if boxes.numel() == 0:
            return []

        # Final NMS across all phrases to remove duplicates
        keep = nms(boxes, scores, self.nms_iou)
        boxes = boxes[keep]
        scores = scores[keep]

        b = boxes.detach().float().cpu().numpy()
        s = scores.detach().float().cpu().numpy()

        out = []
        for i in range(b.shape[0]):
            x1, y1, x2, y2 = map(float, b[i].tolist())
            out.append([x1, y1, x2, y2, float(s[i]), 0.0])  # cls=0 for person
        return out
