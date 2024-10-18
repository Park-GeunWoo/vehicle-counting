import cv2
import numpy as np
from typing import Tuple
from line_zone import LineZone


class LineZoneAnnotator:
    def __init__(self, line_color=(0, 255, 255), thickness=2):
        self.line_color = line_color
        self.thickness = thickness

    def annotate(self, frame: np.ndarray, line_zone: LineZone) -> np.ndarray:
        cv2.line(frame, line_zone.start, line_zone.end, self.line_color, self.thickness)
        return frame

class TraceAnnotator:
    def __init__(self, trace_length=10, line_color_detected=(255, 0, 0), line_color_predicted=(0, 0, 255), thickness=8):
        self.trace_length = trace_length
        self.trace_data = {}
        self.line_color_detected = line_color_detected
        self.line_color_predicted = line_color_predicted
        self.thickness = thickness

    def update_trace(self, tracker_id: int, current_position: Tuple[int, int], predicted=False):
        if tracker_id not in self.trace_data:
            self.trace_data[tracker_id] = []
        self.trace_data[tracker_id].append((current_position, predicted))
        if len(self.trace_data[tracker_id]) > self.trace_length:
            self.trace_data[tracker_id].pop(0)

    def remove_trace(self, tracker_id: int):
        if tracker_id in self.trace_data:
            del self.trace_data[tracker_id]

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        for tracker_id, trace_points in self.trace_data.items():
            for i in range(1, len(trace_points)):
                prev_point, prev_predicted = trace_points[i - 1]
                curr_point, curr_predicted = trace_points[i]
                if curr_predicted:
                    color = self.line_color_predicted
                    cv2.line(frame, prev_point, curr_point, color, self.thickness)
                # color = self.line_color_predicted if curr_predicted else self.line_color_detected
                # cv2.line(frame, prev_point, curr_point, color, self.thickness)
        return frame

class PolygonAnnotator:
    def __init__(self,line_color=(0,0,0),thickness=2):
        self.line_color = line_color
        self.thickness = thickness
    
    def annotate(self, frame: np.ndarray, line_zone: LineZone) -> np.ndarray:
        cv2.line(frame, line_zone.start, line_zone.end, self.line_color, self.thickness)
        return frame