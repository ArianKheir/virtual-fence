# Minimal SORT implementation (no external dependencies).
# Original paper: Simple Online and Realtime Tracking (Bewley et al. 2016).

import numpy as np
from scipy.optimize import linear_sum_assignment

# Kalman filter per track (state: [x, y, s, r, vx, vy, vs])
# where (x,y) is box center, s is area, r is aspect ratio.

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # bbox: x1,y1,x2,y2
        self._init_kalman(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def _init_kalman(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2.0
        y = y1 + h/2.0
        s = max(1e-6, w * h)
        r = w / (h + 1e-9)

        self.x = np.array([x, y, s, r, 0, 0, 0], dtype=float)  # state
        self.P = np.eye(7) * 10.0
        self.F = np.eye(7)
        dt = 1.0
        for i in range(4):
            if i < 3:
                self.F[i, i+4] = dt
        self.Q = np.eye(7) * 0.01
        self.R = np.eye(4) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.x

    def update(self, bbox):
        # measurement z: [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        z = np.array([x1 + w/2.0, y1 + h/2.0, max(1e-6, w*h), w/(h+1e-9)], dtype=float)

        H = np.zeros((4, 7), dtype=float)
        H[0,0] = 1; H[1,1]=1; H[2,2]=1; H[3,3]=1

        # innovation
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        x, y, s, r, vx, vy, vs = self.x
        s = max(1e-9, s)
        r = max(1e-6, r)
        w = np.sqrt(s * r)
        h = s / (w + 1e-9)
        x1 = x - w/2.0
        y1 = y - h/2.0
        x2 = x + w/2.0
        y2 = y + h/2.0
        return np.array([x1, y1, x2, y2], dtype=float)


def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w*h
    denom = ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
             (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-9)
    return wh / denom


class SORT:
    """
    SORT tracker that accepts Nx4 (xyxy) OR Nx5 (xyxy, score) detections.

    Args:
        max_age:    frames to keep a track without update
        min_hits:   frames before a track is emitted
        iou_threshold: match threshold
        det_score_gate: minimum detection score to spawn a new track (if Nx5)
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.2, det_score_gate=0.30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_score_gate = det_score_gate
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Args:
            dets: ndarray of shape (N,4) [x1,y1,x2,y2] or (N,5) [x1,y1,x2,y2,score]
                  Empty input allowed.
        Returns:
            dict: id -> (cx, cy, box)
        """
        self.frame_count += 1

        # Normalize input
        dets = np.asarray(dets, dtype=float)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        if dets.size == 0:
            boxes = np.zeros((0, 4), dtype=float)
            scores = np.zeros((0,), dtype=float)
        else:
            if dets.shape[1] >= 5:
                boxes = dets[:, :4].astype(float)
                scores = dets[:, 4].astype(float)
            else:
                boxes = dets[:, :4].astype(float)
                scores = np.ones((boxes.shape[0],), dtype=float)

        # Predict all trackers
        for t in self.trackers:
            t.predict()

        # Associate
        if len(self.trackers) == 0 or len(boxes) == 0:
            matches, unmatched_dets, unmatched_trks = [], list(range(len(boxes))), list(range(len(self.trackers)))
        else:
            iou_matrix = np.zeros((len(boxes), len(self.trackers)), dtype=float)
            for d, det in enumerate(boxes):
                for t, trk in enumerate(self.trackers):
                    iou_matrix[d, t] = iou(det, trk.get_state())

            # Hungarian: maximize IOU -> minimize (-IOU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matches = []
            unmatched_dets = list(set(range(len(boxes))) - set(row_ind))
            unmatched_trks = list(set(range(len(self.trackers))) - set(col_ind))
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    unmatched_dets.append(r)
                    unmatched_trks.append(c)
                else:
                    matches.append((r, c))

        # update matched trackers
        for d, t in matches:
            self.trackers[t].update(boxes[d])

        # create new trackers (gate by detection score if provided)
        for i in unmatched_dets:
            if scores.shape[0] == 0 or scores[i] >= self.det_score_gate:
                self.trackers.append(KalmanBoxTracker(boxes[i]))

        # remove dead trackers & build return
        alive = []
        ret = []
        for t in self.trackers:
            if t.time_since_update < self.max_age:
                alive.append(t)
                # Emit if warm-up satisfied
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    box = t.get_state()
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    ret.append((t.id, cx, cy, box))
        self.trackers = alive

        result = {}
        for tid, cx, cy, box in ret:
            result[int(tid)] = (float(cx), float(cy), box.astype(float))
        return result
