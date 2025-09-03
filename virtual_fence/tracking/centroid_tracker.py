import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = OrderedDict()   # id -> (cx, cy)
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of [x1, y1, x2, y2]
        if len(rects) == 0:
            # mark disappearances
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype=float)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i, 0] = (x1 + x2) / 2.0
            input_centroids[i, 1] = (y1 + y2) / 2.0

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype=float)

        # distance matrix
        D = np.linalg.norm(object_centroids[:, None, :] - input_centroids[None, :, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self.objects[oid] = tuple(input_centroids[col])
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # mark disappeared / register new
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        unused_cols = set(range(input_centroids.shape[0])) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects
