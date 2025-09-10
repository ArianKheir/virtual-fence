# centroid_tracker.py
import numpy as np
from collections import OrderedDict

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    w = max(0.0, xB - xA); h = max(0.0, yB - yA)
    inter = w * h
    if inter <= 0: return 0.0
    areaA = max(0.0, (boxA[2]-boxA[0])) * max(0.0, (boxA[3]-boxA[1]))
    areaB = max(0.0, (boxB[2]-boxB[0])) * max(0.0, (boxB[3]-boxB[1]))
    union = areaA + areaB - inter + 1e-6
    return inter / union

class TrackState:
    __slots__ = ("id","cx","cy","box","prev_cx","prev_cy","vx","vy",
                 "disappeared","miss_streak","hits","age","score")
    def __init__(self, tid, cx, cy, box, score=1.0):
        self.id = tid
        self.cx = cx; self.cy = cy
        self.box = list(map(float, box))
        self.prev_cx = cx; self.prev_cy = cy
        self.vx = 0.0; self.vy = 0.0
        self.disappeared = 0
        self.miss_streak = 0
        self.hits = 1
        self.age = 1
        self.score = float(score)

class CentroidTracker:
    """
    Box-aware, greedy matcher with gating and bounded persistence.
    Key anti-ghost settings:
      - predict_miss_horizon: predict at most this many consecutive miss frames
      - vel_damping: reduce velocity on prediction to prevent drift
      - max_vel_px: clamp per-axis velocity (pixels/frame)
      - draw_miss_budget: only draw if last seen within this many frames
      - reacquire_iou: stricter IoU when re-acquiring after a miss
      - reg_min_conf: don't start tracks from very low-confidence detections
    """
    def __init__(self,
                 max_disappeared=30,
                 max_distance=100.0,
                 min_iou=0.1,
                 min_hits=2,
                 use_prediction=True,
                 predict_miss_horizon=1,
                 vel_damping=0.6,
                 max_vel_px=50.0,
                 draw_miss_budget=8,
                 reacquire_iou=0.2,
                 reg_min_conf=0.25):
        self.next_id = 0
        self.tracks = OrderedDict()
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)
        self.min_iou = float(min_iou)
        self.min_hits = int(min_hits)
        self.use_prediction = bool(use_prediction)
        self.predict_miss_horizon = int(predict_miss_horizon)
        self.vel_damping = float(vel_damping)
        self.max_vel_px = float(max_vel_px)
        self.draw_miss_budget = int(draw_miss_budget)
        self.reacquire_iou = float(reacquire_iou)
        self.reg_min_conf = float(reg_min_conf)

    def register(self, box, score=1.0):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        t = TrackState(self.next_id, cx, cy, box, score=score)
        self.tracks[self.next_id] = t
        self.next_id += 1

    def deregister(self, oid):
        if oid in self.tracks:
            del self.tracks[oid]

    def _gate(self, track: TrackState, det_box, det_cx, det_cy):
        # If trying to re-acquire after a miss, require stronger IoU.
        if track.disappeared > 0 or track.miss_streak > 0:
            return iou(track.box, det_box) >= self.reacquire_iou
        # Otherwise allow either IoU or distance gate.
        if self.min_iou > 0 and iou(track.box, det_box) >= self.min_iou:
            return True
        if self.max_distance < float("inf"):
            dist = abs(track.cx - det_cx) + abs(track.cy - det_cy)
            if dist <= self.max_distance:
                return True
        return False

    def _predict_if_missing(self, track: TrackState):
        if not self.use_prediction:
            return
        if track.miss_streak > self.predict_miss_horizon:
            return  # freeze after short horizon to avoid drifting ghosts
        # damped, clamped constant-velocity shift
        dx = max(-self.max_vel_px, min(self.max_vel_px, track.vx * self.vel_damping))
        dy = max(-self.max_vel_px, min(self.max_vel_px, track.vy * self.vel_damping))
        track.prev_cx = track.cx; track.prev_cy = track.cy
        track.cx += dx; track.cy += dy
        track.box[0] += dx; track.box[1] += dy
        track.box[2] += dx; track.box[3] += dy

    def update(self, rects, scores=None):
        if scores is None:
            scores = [1.0] * len(rects)
        else:
            scores = [float(s) for s in scores]

        # No detections: mark disappear and optionally predict a tiny step
        if len(rects) == 0:
            for oid, t in list(self.tracks.items()):
                t.disappeared += 1
                t.miss_streak += 1
                t.age += 1
                if t.disappeared > self.max_disappeared:
                    self.deregister(oid)
                else:
                    self._predict_if_missing(t)
            return {t.id: (float(t.cx), float(t.cy), [*map(float, t.box), float(t.score)])
                    for t in self.tracks.values()
                    if t.hits >= self.min_hits and t.miss_streak <= self.draw_miss_budget}

        det_boxes = [list(map(float, b)) for b in rects]
        det_cxcy = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in det_boxes]
        det_used = set()

        # Bootstrap: register only decent detections
        if len(self.tracks) == 0:
            for b, sc in zip(det_boxes, scores):
                if sc >= self.reg_min_conf:
                    self.register(b, sc)
            return {t.id: (float(t.cx), float(t.cy), [*map(float, t.box), float(t.score)])
                    for t in self.tracks.values()
                    if t.hits >= self.min_hits and t.miss_streak <= self.draw_miss_budget}

        # Candidate matches (prefer higher IoU, then smaller distance)
        pairs = []
        for oid, t in self.tracks.items():
            for j, b in enumerate(det_boxes):
                cx, cy = det_cxcy[j]
                if not self._gate(t, b, cx, cy):
                    continue
                j_iou = iou(t.box, b)
                dist = abs(t.cx - cx) + abs(t.cy - cy)
                pairs.append((-j_iou, dist, oid, j))
        pairs.sort()

        used_tracks = set()
        for _, _, oid, j in pairs:
            if oid in used_tracks or j in det_used:
                continue
            # assign detection j to track oid
            t = self.tracks[oid]
            old_cx, old_cy = t.cx, t.cy
            t.cx, t.cy = det_cxcy[j]
            t.vx, t.vy = (t.cx - old_cx), (t.cy - old_cy)  # update velocity only on hits
            t.prev_cx, t.prev_cy = old_cx, old_cy
            t.box = det_boxes[j][:]
            t.disappeared = 0
            t.miss_streak = 0
            t.hits += 1
            t.age += 1
            t.score = scores[j]
            used_tracks.add(oid)
            det_used.add(j)

        # Unmatched tracks: keep but don't let them drift indefinitely
        for oid, t in list(self.tracks.items()):
            if oid in used_tracks:
                continue
            t.disappeared += 1
            t.miss_streak += 1
            t.age += 1
            self._predict_if_missing(t)
            if t.disappeared > self.max_disappeared:
                self.deregister(oid)

        # Unmatched detections: register if confident
        for j, b in enumerate(det_boxes):
            if j in det_used:
                continue
            if scores[j] >= self.reg_min_conf:
                self.register(b, scores[j])

        # Export only confirmed + recently seen tracks to avoid ghosts
        result = {}
        for t in self.tracks.values():
            if t.hits >= self.min_hits and t.miss_streak <= self.draw_miss_budget and t.disappeared <= self.max_disappeared:
                result[t.id] = (float(t.cx), float(t.cy), [*map(float, t.box), float(t.score)])
        return result
