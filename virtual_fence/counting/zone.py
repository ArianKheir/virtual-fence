from dataclasses import dataclass, field
from typing import Tuple, Set, Dict

@dataclass
class ZoneCounter:
    zone: Tuple[int, int, int, int]  # x1, y1, x2, y2
    counted_ids: Set[int] = field(default_factory=set)
    total: int = 0

    def in_zone(self, cx: float, cy: float) -> bool:
        x1, y1, x2, y2 = self.zone
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def update(self, tracked: Dict[int, tuple]):
        # tracked: id -> (cx, cy)
        for tid, (cx, cy) in tracked.items():
            if tid in self.counted_ids:
                continue
            if self.in_zone(cx, cy):
                self.counted_ids.add(tid)
                self.total += 1
        return self.total
