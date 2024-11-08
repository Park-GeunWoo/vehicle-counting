import cv2
import numpy as np
from typing import Tuple, List
from collections import defaultdict
from line_zone import LineZone
from supervision.tracker.byte_tracker.basetrack import TrackState

counted_tracker_ids = set()


def build_spatial_grid(line_zones, grid_size=100):
    """
    공간 분할을 위한 격자 데이터를 생성합니다.
    """
    grid = defaultdict(list)
    for line_zone in line_zones:
        min_x = min(line_zone.start[0], line_zone.end[0]) // grid_size
        min_y = min(line_zone.start[1], line_zone.end[1]) // grid_size
        max_x = max(line_zone.start[0], line_zone.end[0]) // grid_size
        max_y = max(line_zone.start[1], line_zone.end[1]) // grid_size

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                grid[(x, y)].append(line_zone)
    return grid


def get_relevant_zones(grid, x, y, grid_size=100):
    """
    객체가 속한 격자의 관련 라인 존을 반환합니다.
    """
    grid_key = (x // grid_size, y // grid_size)
    return grid.get(grid_key, [])


def check_line_crossing_multiple_zones(
    tracker_id: int,
    previous_coordinates: list,
    line_zones_grid,
    grid_size=100,
):
    """
    여러 LineZone과 교차 여부를 확인하는 함수 (격자 기반).
    """
    global count

    if tracker_id in counted_tracker_ids:
        return

    # 이전 및 현재 좌표 가져오기
    prev_coords = np.array(previous_coordinates[-3][0])
    curr_coords = np.array(previous_coordinates[-1][0])

    # 객체가 속한 격자의 라인 존만 확인
    relevant_zones = get_relevant_zones(line_zones_grid, curr_coords[0], curr_coords[1], grid_size)

    for line_zone in relevant_zones:
        if line_zone.is_crossing(prev_coords, curr_coords):
            curr_direction = -np.cross(line_zone.direction_vector, curr_coords - line_zone.start)
            prev_direction = -np.cross(line_zone.direction_vector, prev_coords - line_zone.start)

            if prev_direction > 0 and curr_direction <= 0:
                count[0] += 1
                counted_tracker_ids.add(tracker_id)
                break


def process_detections(detections, tracker, trace_annotator, line_zones):
    """
    감지된 객체의 추적 및 라인 교차 여부를 확인하는 함수.
    """
    # 격자 데이터 생성 (최초 한 번만 실행)
    line_zones_grid = build_spatial_grid(line_zones)

    # 감지된 객체 추적 및 경로 업데이트
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
            check_line_crossing_multiple_zones(tracker_id, previous_coordinates, line_zones_grid)

    # 잃어버린 객체 처리
    for track in tracker.lost_tracks:
        if track.state == TrackState.Lost:
            predicted_coords = track.mean[:2]
            center_x, center_y = int(predicted_coords[0]), int(predicted_coords[1])

            # 경로 업데이트 (예측된 좌표로)
            trace_annotator.update_trace(track.external_track_id, (center_x, center_y), predicted=True)
            previous_coordinates = trace_annotator.trace_data.get(track.external_track_id)

            if previous_coordinates and len(previous_coordinates) > 2:
                check_line_crossing_multiple_zones(track.external_track_id, previous_coordinates, line_zones_grid)

    # 추적이 중단된 객체 처리
    for track in tracker.removed_tracks:
        trace_annotator.remove_trace(track.external_track_id)
