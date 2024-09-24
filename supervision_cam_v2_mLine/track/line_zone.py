import cv2
import numpy as np
from typing import Tuple, List
from data.count import in_count, out_count

counted_tracker_ids = set()


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


def check_line_crossing_multiple_zones(
    tracker_id: int,
    previous_coordinates: list,
    line_zones: list
    ):
    """여러개의 LineZone과 교차 여부를 확인하는 함수"""
    
    if tracker_id in counted_tracker_ids:
        return
    
    global in_count,out_count
    
    prev_x, prev_y = previous_coordinates[-3][0]
    curr_x, curr_y = previous_coordinates[-1][0]


    for line_zone in line_zones:
        #객가 교차했는지 확인
        if line_zone.is_crossing((prev_x, prev_y), (curr_x, curr_y)):
            #이전위치와 현재위치 비교
            curr_direction = (line_zone.end[1] - line_zone.start[1]) * (curr_x - line_zone.start[0]) - (line_zone.end[0] - line_zone.start[0]) * (curr_y - line_zone.start[1])
            prev_direction = (line_zone.end[1] - line_zone.start[1]) * (prev_x - line_zone.start[0]) - (line_zone.end[0] - line_zone.start[0]) * (prev_y - line_zone.start[1])

            #이전 방향과 현재 방향이 다르면 교차
            if prev_direction > 0 and curr_direction <= 0:
                #선 위에서 아래로->in
                
                in_count[0] += 1
                print(f'asdf{in_count}')
                
            elif prev_direction < 0 and curr_direction >= 0:
                #선 아래에서 위로->out
                
                out_count[0] += 1
                print(f'asdf{out_count}')

            #카운팅된 객체는 set에 추가
            counted_tracker_ids.add(tracker_id)
            break
        