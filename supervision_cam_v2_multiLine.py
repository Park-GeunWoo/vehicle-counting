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

counted_tracker_ids=set() #카운팅된 tracker id
        
class TraceAnnotator:
    def __init__(
        self,
        trace_length=10, 
        line_color_detected=(255, 0, 0), 
        line_color_predicted=(0, 0, 255),
        thickness=8
        ):
        self.trace_length = trace_length  #경로 길이
        self.trace_data = {}  #trace coordinate
        self.line_color_detected = line_color_detected  #detected line color
        self.line_color_predicted = line_color_predicted  #경로 색상
        self.thickness = thickness


    def update_trace(
        self,
        tracker_id: int, 
        current_position: Tuple[int, int], 
        predicted=False
        ):
        """현재 좌표를 추가하고 trace 길이를 유지하는 함수"""
        if tracker_id not in self.trace_data:
            self.trace_data[tracker_id] = []

        #리스트에 현재좌표 추가
        self.trace_data[tracker_id].append((current_position, predicted))

        #trace길이를 초과하면 오래된 좌표 삭제
        if len(self.trace_data[tracker_id]) > self.trace_length:
            self.trace_data[tracker_id].pop(0)

    def remove_trace(self, tracker_id: int):
        """트래킹이 중단된 객체의 좌표 데이터를 삭제하는 함수"""
        if tracker_id in self.trace_data:
            del self.trace_data[tracker_id]
            
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """객체의 경로를 프레임에 그리는 함수"""
        for tracker_id, trace_points in self.trace_data.items():
            #객체의 좌표들을 연결
            for i in range(1, len(trace_points)):
                prev_point, prev_predicted = trace_points[i - 1]
                curr_point, curr_predicted = trace_points[i]

                #김지된 객체인지 예측된 객체인지 판단
                color = self.line_color_predicted if curr_predicted else self.line_color_detected

                cv2.line(
                    frame,
                    prev_point,  #이전 좌표
                    curr_point,  #현재 좌표
                    color,       #상태에 따른 색
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
        """세 점이 시계방향인지 반시계방향인지를 판단하는 함수"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def is_crossing(self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> bool:
        """선분이 LineZone과 교차하는지 계산하는 함수"""
        A, B = self.start, self.end  #LineZone의 선분
        C, D = prev_pos, curr_pos  #객체의 선분
        
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)


def check_line_crossing(
    tracker_id: int, 
    previous_coordinates: list,
    line_zone: LineZone
    ):
    """객체의 선분이 Linezone과 교차하는지 확인하는 함수"""
    
    if tracker_id in counted_tracker_ids:
        return
    
    global in_count, out_count

    #좌표는 tuple로 [0] 인덱스에 저장되있음
    prev_x, prev_y = previous_coordinates[-3][0]  #이전 프레임 좌표
    curr_x, curr_y = previous_coordinates[-1][0]  #현재 프레임 좌표

    
    #객체가 LineZone과 교차했는지 확인
    if line_zone.is_crossing((prev_x, prev_y), (curr_x, curr_y)):
        
        if object_state.get(tracker_id, False):
            in_count += 1
        else:
            out_count += 1
            
        counted_tracker_ids.add(tracker_id)
        

def check_line_crossing_multiple_zones(
    tracker_id: int,
    previous_coordinates: list,
    line_zones: list
    ):
    """여러개의 LineZone과 교차 여부를 확인하는 함수"""
    
    if tracker_id in counted_tracker_ids:
        return
    
    global in_count, out_count
    
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
                in_count += 1
            elif prev_direction < 0 and curr_direction >= 0:
                #선 아래에서 위로->out
                out_count += 1

            #카운팅된 객체는 set에 추가
            counted_tracker_ids.add(tracker_id)
            break
    
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


def process_frame(
    frame: np.ndarray, 
    index: int,
    model, 
    tracker,
    conf_thres,
    line_zones,
    box_annotator,
    label_annotator,
    trace_annotator,
    line_annotator,
    classes,
    smoother
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
    
    detections=detections[np.isin(detections.class_id, classes)]
    detections=detections[detections.confidence > conf_thres]
    detections = smoother.update_with_detections(detections)
    

    """감지된 객체들의 처리"""
    for detection_idx in range(len(detections)):
        tracker_id = int(detections.tracker_id[detection_idx])

        #Center coordinates of the bounding box
        x_min, y_min, x_max, y_max = detections.xyxy[detection_idx]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)


        #TraceAnnotator 업데이트
        trace_annotator.update_trace(tracker_id, (center_x, center_y), predicted=False)
    

        previous_coordinates = trace_annotator.trace_data.get(tracker_id) #이전좌표들
        if previous_coordinates and len(previous_coordinates) > 2:
            check_line_crossing_multiple_zones(tracker_id, previous_coordinates, line_zones)
                    
                    
                        
    """감지되지 않으며 트래킹되고 있는 객체들의 처리"""
    for track in tracker.lost_tracks:
        if track.state == TrackState.Lost:
            
            predicted_coords = track.mean[:2]  #Kalman 필터에서 예측된 좌표
            center_x, center_y = int(predicted_coords[0]), int(predicted_coords[1])

            #TraceAnnotator 업데이트
            trace_annotator.update_trace(track.external_track_id, (center_x, center_y), predicted=True)

            #교차 여부 확인
            previous_coordinates = trace_annotator.trace_data.get(track.external_track_id)
            if previous_coordinates and len(previous_coordinates) > 2:
                check_line_crossing_multiple_zones(track.external_track_id, previous_coordinates, line_zones)
                
                

    #트래킹이 중단된 객체의 좌표 삭제
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
    
    # annotated_frame = line_annotator.annotate(
    #     frame=annotated_frame,
    #     line_zone=line_zones
    # )
    
    
    for line_zone in line_zones:
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
        2, #fontScale
        (125, 0, 255), #color
        2, #thickness
        )
    
    return annotated_frame


def main():
    weights = 'yolov8n.pt'
    
    model = YOLO(weights)
    tracker=sv.ByteTrack(
        track_activation_threshold=0.55, #트래킹 임계값
        lost_track_buffer=30,  # n 프레임만큼 트래킹
        minimum_matching_threshold=0.8, #두 객체의 매칭값
        frame_rate=10,  # 초당 트래킹할 프레임
        minimum_consecutive_frames=3 #트래킹이 시작되기 위한 최소 Detected되는 수 
    )
    
    label_annotator=sv.LabelAnnotator()
    box_annotator=sv.BoxAnnotator()
    line_annotator=LineZoneAnnotator()
    trace_annotator = TraceAnnotator(trace_length=10) 
    smoother = sv.DetectionsSmoother()
    
    Cam = False
    
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/8_Output 북측_G96/16-19/alwa_20240529_164029_F.mp4" #1
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/7_Output 남측_G95/16-19/alwa_20240529_181815_F.mp4" #2
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/6_Output 서측_G89/16-19/alwa_20240529_162522_F.mp4" #3
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/5_Output 동측_G91/16-19/alwa_20240529_181533_F.mp4" #4
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/4_Input 북측_G94/16-19/alwa_20240529_155919_F.mp4" #5
    input_path='C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/3_Input 남측_G88/16-19/alwa_20240529_185759_F.mp4' #6
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/2_Input 서측_G93/16-19/alwa_20240529_180041_F.mp4" #7
    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/1_Input 동측_G87/16-19/alwa_20240529_185838_F.mp4"#8
    
    output_filename = 'result'
    
    line_zones=[#1
        LineZone(start=(80,401),end=(1142,844)),
        LineZone(start=(263,357),end=(1323,612)),
        LineZone(start=(450,332),end=(1425,403))
    ]
    
    line_zones=[#2
        LineZone(start=(173,489),end=(873,991)),
        LineZone(start=(539,441),end=(1413,811)),
        LineZone(start=(742,401),end=(1749,613))
    ]
    
    line_zones=[#3
        LineZone(start=(122,304),end=(1037,915)),
        LineZone(start=(390,209),end=(1427,465)),
        LineZone(start=(587,153),end=(1564,200))
    ]
    line_zones=[#4
        LineZone(start=(50,320),end=(844,943)),
        LineZone(start=(319,266),end=(1176,628)),
        LineZone(start=(464,254),end=(1324,363))
    ]
    
    line_zones=[#5
        LineZone(start=(1879,468),end=(1456,991)),
        LineZone(start=(1742,421),end=(648,840)),
        LineZone(start=(1502,355),end=(574,657))
    ]
    
    line_zones=[#6
        LineZone(start=(324,653),end=(1356,387)),
        LineZone(start=(563,938),end=(1536,400)),
        LineZone(start=(1471,985),end=(1743,414))
    ]
    
    line_zones=[#7
        LineZone(start=(1367,316),end=(581,522)),
        LineZone(start=(1579,349),end=(912,830)),
        LineZone(start=(1762,380),end=(1653,978))
    ]
    
    line_zones=[#8
        LineZone(start=(1386,374),end=(542,570)),
        LineZone(start=(1546,393),end=(717,773)),
        LineZone(start=(1644,399),end=(1542,982))
    ]
    
    
    #roi 좌표 
    # y_start,y_end=(),()
    # x_start,x_end=(),()
    
    conf_thres = 0.25
    classes = [2, 3, 5, 7] # bicycle,car,motorcycle,bus,truck
    stride = 2
    

    if Cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print('Could not open webcam')
        return
    
    '''Fps 표시'''
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_delay = int(1000 / fps)
    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f'Webcam resolution: {width}x{height}')
    
    
    output_filename = getNextFilename(base_name=output_filename, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width,height))
    
    fdx=1
    index = 1
    prev_time = time.time()
    
    while True:
        success, frame = cap.read()
        # frame=frame[y_start:y_end,x_start:x_end]
        if not success:
            break
        
        if index%stride==0:
            annotated_frame = process_frame(
                frame,
                index,
                model,
                tracker, 
                conf_thres,
                line_zones,
                box_annotator,
                label_annotator,
                trace_annotator,
                line_annotator,
                classes,
                smoother
            )
        
            #Fps 계산
            current_time = time.time()
            fdx = stride / (current_time - prev_time)
            prev_time = current_time
            
            #Fps cv2
            cv2.putText(
                annotated_frame,
                f'FPS: {fdx:.2f}', 
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
        
        index+=1
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
