import cv2
import numpy as np
from typing import Tuple, List

from data.count import in_count, out_count

from data.class_names import class_names

from supervision.tracker.byte_tracker.basetrack import TrackState
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

def process_detections(
        detections,
        tracker,
        trace_annotator,
        line_zones
    ):
    """
    감지된 객체의 추적 및 라인 교차 여부를 확인하는 함수.
    
    Parameters:
    - detections: 감지된 객체 목록
    - tracker: 트래커 객체
    - trace_annotator: 경로 어노테이터 객체
    - line_zones: 라인 존 객체 목록
    - check_line_crossing_multiple_zones: 라인 교차 여부를 확인하는 함수
    """
    # 감지된 객체에 대한 추적 정보 업데이트 및 라인 교차 확인
    for detection_idx in range(len(detections)):
        tracker_id = int(detections.tracker_id[detection_idx])
        x_min, y_min, x_max, y_max = detections.xyxy[detection_idx]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        
        # 경로 업데이트
        trace_annotator.update_trace(tracker_id, (center_x, center_y), predicted=False)
        previous_coordinates = trace_annotator.trace_data.get(tracker_id)

        # 이전 경로가 있고 길이가 2 이상일 때 라인 교차 확인
        if previous_coordinates and len(previous_coordinates) > 2:
            check_line_crossing_multiple_zones(tracker_id, previous_coordinates, line_zones)

    # 잃어버린 객체에 대한 추적 정보 처리
    for track in tracker.lost_tracks:
        if track.state == TrackState.Lost:
            predicted_coords = track.mean[:2]
            center_x, center_y = int(predicted_coords[0]), int(predicted_coords[1])
            
            # 경로 업데이트 (예측된 좌표로)
            trace_annotator.update_trace(track.external_track_id, (center_x, center_y), predicted=True)
            previous_coordinates = trace_annotator.trace_data.get(track.external_track_id)

            # 이전 경로가 있고 길이가 2 이상일 때 라인 교차 확인
            if previous_coordinates and len(previous_coordinates) > 2:
                check_line_crossing_multiple_zones(track.external_track_id, previous_coordinates, line_zones)

    # 추적이 중단된 객체 처리
    for track in tracker.removed_tracks:
        trace_annotator.remove_trace(track.external_track_id)
        
    labels = [f"#{tracker_id} {class_names.get(class_id, 'Unknown')} {confidence:.2f}" for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)]

    return labels

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
                
            elif prev_direction < 0 and curr_direction >= 0:
                #선 아래에서 위로->out
                
                out_count[0] += 1

            #카운팅된 객체는 set에 추가
            counted_tracker_ids.add(tracker_id)
            break
        