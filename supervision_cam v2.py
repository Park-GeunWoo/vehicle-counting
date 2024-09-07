import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Tuple, Dict

from utils import getNextFilename
from class_names import class_names

from supervision.tracker.byte_tracker.basetrack import TrackState


in_count = 0
out_count = 0

object_state={} #객체 부호
counted_tracker_ids=set() #카운팅된 객체
        
class TraceAnnotator:
    def __init__(
        self,
        trace_length=10, 
        line_color_detected=(0, 255, 0), 
        line_color_predicted=(0, 0, 255),
        thickness=2
        ):
        self.trace_length = trace_length  # 최대 추적 좌표 길이
        self.trace_data = {}  # 트래킹된 좌표 기록
        self.line_color_detected = line_color_detected  # 감지된 상태의 색상 (초록색)
        self.line_color_predicted = line_color_predicted  # 예측된 상태의 색상 (빨간색)
        self.thickness = thickness


    def update_trace(self, tracker_id: int, current_position: Tuple[int, int], predicted=False):
        """현재 좌표를 추가하고 trace 길이를 유지하는 함수"""
        if tracker_id not in self.trace_data:
            self.trace_data[tracker_id] = []

        # 좌표 리스트에 현재 좌표 추가
        self.trace_data[tracker_id].append((current_position, predicted))

        # trace 길이가 초과하면 오래된 좌표 삭제
        if len(self.trace_data[tracker_id]) > self.trace_length:
            self.trace_data[tracker_id].pop(0)

    def remove_trace(self, tracker_id: int):
        """트래킹이 중단된 객체의 좌표 데이터를 삭제하는 함수"""
        if tracker_id in self.trace_data:
            del self.trace_data[tracker_id]
            
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """객체의 트레이스를 프레임에 그리는 함수"""
        for tracker_id, trace_points in self.trace_data.items():
            # 각 객체의 trace 좌표를 선으로 연결
            for i in range(1, len(trace_points)):
                prev_point, prev_predicted = trace_points[i - 1]
                curr_point, curr_predicted = trace_points[i]

                # 감지된 상태인지 예측된 상태인지 확인하고 색상 변경
                color = self.line_color_predicted if curr_predicted else self.line_color_detected

                cv2.line(
                    frame,
                    prev_point,  # 이전 좌표
                    curr_point,  # 현재 좌표
                    color,       # 상태에 따른 색상
                    self.thickness
                )
        return frame



class LineZone:
    
    def __init__(
        self,
        start:Tuple[int,int],
        end:Tuple[int,int]
        ):

        self.start=start
        self.end=end
    
    def _ccw(self, A, B, C):
        # 세 점이 시계방향인지 반시계방향인지를 판단하는 함수
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def is_crossing(self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> bool:
        # 객체의 이전 위치(prev_pos)와 현재 위치(curr_pos)를 선분으로 보고
        # 이 선분이 LineZone과 교차하는지 확인하는 함수
        A, B = self.start, self.end  # LineZone의 선분
        C, D = prev_pos, curr_pos  # 객체의 선분
        
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)


class LineZoneAnnotator:
    
    def __init__(
        self,
        line_color=(0,255,255),
        thickness=2
        ):
    
        self.line_color=line_color
        self.thickness=thickness
    
    def annotate(
        self,
        frame:np.ndarray,
        line_zone:LineZone
        )->np.ndarray:
        
        cv2.line(
            frame,
            line_zone.start,
            line_zone.end,
            self.line_color,
            self.thickness
        )
        return frame

def check_line_crossing(
    tracker_id: int, 
    previous_coordinates: list,
    line_zone: LineZone
    ):
    
    
    if tracker_id in counted_tracker_ids:
        return
    
    global in_count, out_count

    # 좌표는 tuple 형식의 첫 번째 요소에 저장되어 있으므로 [0]으로 좌표를 가져옴
    prev_x, prev_y = previous_coordinates[-3][0]  # 이전 프레임의 좌표
    curr_x, curr_y = previous_coordinates[-1][0]  # 현재 좌표

    
    # 객체가 LineZone을 교차했는지 확인
    if line_zone.is_crossing((prev_x, prev_y), (curr_x, curr_y)):
        
        if object_state.get(tracker_id, False):
            in_count += 1
        else:
            out_count += 1
        counted_tracker_ids.add(tracker_id)
    


def process_frame(
    frame: np.ndarray, 
    index: int,
    model, 
    tracker,
    conf_thres,
    line_zone,
    box_annotator,
    label_annotator,
    trace_annotator,
    line_annotator
    ) -> np.ndarray:
    
    global in_count,out_count,object_state
    #Load frame
    results = model(frame)[0]
    
    detections = sv.Detections.from_ultralytics(results) #Convert Yolo results to Detections
    detections=tracker.update_with_detections(detections)
    
    #no detected objects
    if detections.tracker_id is None:
        print(f"No objects detected")
        return frame
    
    #detections=detections[np.isin(detections.class_id, classes)]
    detections=detections[detections.confidence > conf_thres]
    print(counted_tracker_ids)
    # 감지된 객체에 대해 처리
    if detections.tracker_id is not None:
        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])

            # 바운딩 박스 중앙 좌표 계산
            x_min, y_min, x_max, y_max = detections.xyxy[detection_idx]
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            # 감지된 좌표로 TraceAnnotator 업데이트 (감지된 상태)
            trace_annotator.update_trace(tracker_id, (center_x, center_y), predicted=False)
            
                # 객체가 처음 감지된 경우에만 부호 할당
            if tracker_id not in object_state:
                # 직선 방정식 계산
                x_center = center_x
                y_center = center_y
                # 선의 방정식 (Ax + By + C = 0)을 사용한 부호 계산
                A = line_zone.end[1] - line_zone.start[1]  # y2 - y1
                B = line_zone.start[0] - line_zone.end[0]  # x1 - x2
                C = line_zone.end[0] * line_zone.start[1] - line_zone.start[0] * line_zone.end[1]  # x2 * y1 - x1 * y2

                # 객체가 처음 감지된 위치의 부호 할당 (True: in, False: out)
                sign = A * x_center + B * y_center + C
                object_state[tracker_id] = sign < 0  # 라인의 아래쪽에 있으면 True (in), 위쪽이면 False (out)

            previous_coordinates = trace_annotator.trace_data.get(tracker_id) #이전좌표들
            if previous_coordinates and len(previous_coordinates) > 2:
                check_line_crossing(tracker_id, previous_coordinates, line_zone)
                        
                        
    # 감지되지 않은 객체들에 대해 처리
    for track in tracker.lost_tracks:
        if track.state == TrackState.Lost:
            
            # 예측된 좌표로 TraceAnnotator 업데이트 (예측된 상태)
            predicted_coords = track.mean[:2]  # Kalman 필터에서 예측된 좌표
            center_x, center_y = int(predicted_coords[0]), int(predicted_coords[1])

            # 예측된 좌표로 TraceAnnotator 업데이트
            trace_annotator.update_trace(track.external_track_id, (center_x, center_y), predicted=True)

            # 이전 좌표 가져오기 및 교차 여부 확인 (예측된 객체)
            previous_coordinates = trace_annotator.trace_data.get(track.external_track_id)
            if previous_coordinates and len(previous_coordinates) > 2:
                check_line_crossing(track.external_track_id, previous_coordinates, line_zone)
                
                

    # 트래킹이 중단된 객체의 좌표 삭제
    for track in tracker.removed_tracks:
        trace_annotator.remove_trace(track.external_track_id)
        
    
    #label annotater str
    labels = [
        f"#{tracker_id} {class_names.get(class_id, 'Unknown')} {confidence:.2f}"
        for tracker_id, class_id, confidence 
        in zip(detections.tracker_id, detections.class_id, detections.confidence)
    ]
    
    
    for label in labels:
        print(f'{label}')

    #annotate
    annotated_frame = box_annotator.annotate(
        scene=frame, 
        detections=detections
    )
    annotated_frame=trace_annotator.annotate(
        frame=annotated_frame
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    
    annotated_frame = line_annotator.annotate(
        frame=annotated_frame,
        line_zone=line_zone
    )
     
    
#Show counts
    cv2.putText(
        annotated_frame,
        f"IN Count: {in_count} OUT Count: {out_count}",
        (80, 100), #location
        cv2.FONT_HERSHEY_SIMPLEX, #font
        2,
        (125, 0, 255), #text color
        2,
        cv2.LINE_AA
        )
    
    
    return annotated_frame

def main():
    weights = 'yolov8s.pt'
    
    model = YOLO(weights)
    tracker=sv.ByteTrack(
        track_activation_threshold=0.55,
        lost_track_buffer=60,  # 예측을 더 길게 하기 위해 설정
        minimum_matching_threshold=0.8,
        frame_rate=10,
        minimum_consecutive_frames=3
    )
    label_annotator=sv.LabelAnnotator()
    box_annotator=sv.BoxAnnotator()
    line_annotator=LineZoneAnnotator()
    trace_annotator = TraceAnnotator(trace_length=10)
    
    Cam = True
    input_path='C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/3_Input 남측_G88/16-19/alwa_20240529_185759_F.mp4' 
    
    output_filename = 'result'
    
    start_point, end_point = (585, 747), (1595, 392)
    line_zone=LineZone(start=start_point, end=end_point)
    
    conf_thres = 0.25
    classes = [2, 3, 5, 7]
    stride = 1
    

    if Cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print('Could not open webcam')
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # FPS를 못 가져오면 기본값 설정
        fps = 30
    frame_delay = int(1000 / fps)  # 각 프레임 사이의 대기 시간(ms)
    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'Webcam resolution: {int(width)}x{int(height)}')
    
    
    output_filename = getNextFilename(base_name=output_filename, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))
    
        
    index = 1
    prev_time = time.time()
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        if index%stride==0:
            annotated_frame = process_frame(
                frame,
                index,
                model,
                tracker,
                conf_thres,
                line_zone,
                box_annotator,
                label_annotator,
                trace_annotator,
                line_annotator
            )
        
            # FPS 계산
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # FPS 값을 화면에 표시
            cv2.putText(
                annotated_frame,
                f'FPS: {fps:.2f}', 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0),
                2
                )
        
        
            cv2.imshow('cv2', annotated_frame)
            out.write(annotated_frame)
        
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break
        
        index += 1
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
