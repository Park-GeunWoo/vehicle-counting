import cv2
import numpy as np
from typing import Tuple, List

class LineZone:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = start
        self.end = end

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def is_crossing(self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> bool:
        A, B = self.start, self.end
        C, D = prev_pos, curr_pos
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)


class LineZoneAnnotator:
    def __init__(self, line_color=(0, 255, 255), thickness=2):
        self.line_color = line_color
        self.thickness = thickness

    def annotate(self, frame: np.ndarray, line_zone: LineZone) -> np.ndarray:
        cv2.line(frame, line_zone.start, line_zone.end, self.line_color, self.thickness)
        return frame
