from abc import ABC, abstractmethod
from typing import List, Tuple

# Return format for detect(): List[ (x1,y1,x2,y2, conf, cls_id) ]
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> list:
        pass
