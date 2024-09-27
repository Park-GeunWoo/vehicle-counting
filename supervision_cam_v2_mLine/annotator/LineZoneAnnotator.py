import cv2
import numpy as np
from line_zone import LineZone


class LineZoneAnnotator:
    def __init__(self, line_color=(0, 255, 255), thickness=2):
        self.line_color = line_color
        self.thickness = thickness

    def annotate(self, frame: np.ndarray, line_zone: LineZone) -> np.ndarray:
        cv2.line(frame, line_zone.start, line_zone.end, self.line_color, self.thickness)
        return frame