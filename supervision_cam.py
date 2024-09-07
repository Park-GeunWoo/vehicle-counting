import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Tuple, Dict

from class_names import class_names

in_count = 0
out_count = 0


def getNextFilename(base_name="result", extension="mp4"):
    i = 1
    while True:
        filename = f"{base_name}{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1


class customLineZone:
    def __init__(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int],
        crossed_ids:set
        ):
        
        self.start = start
        self.end = end
        self.tracker_state: Dict[int, bool] = {}
        self.crossed_ids = crossed_ids
        
    def _is_crossing(
        self, 
        bounding_box: Tuple[int, int, int, int],
        current_state: bool #Whether the object has previously crossed the line (True if crossed, False otherwise)
        ) -> bool:
        
        #BoundingBox center point calculation
        x_center = (bounding_box[0] + bounding_box[2]) / 2 #(x1+x2)/2
        y_center = (bounding_box[1] + bounding_box[3]) / 2 #(y1+y2)/2
        
        #Ax+By+C=0
        A = self.end[1] - self.start[1] # y_2 - y_1 
        B = self.start[0] - self.end[0] # x_1 - x_2
        C = self.end[0] * self.start[1] - self.start[0] * self.end[1] #(x_2 * y_1)-(x_1 * y_2)
        
        #sign
        sign = A * x_center + B * y_center + C
        
        #When first detected
        if current_state is None: 
            return sign < 0
        
        #sign conversion
        return sign < 0 if not current_state else sign > 0

    def trigger(
        self, 
        detections: Dict[int, Tuple[int, int, int, int]]  #{tracker_id: current_state val}
        ):
        global in_count, out_count
        
        for tracker_id, bounding_box in detections.items():
            
            if tracker_id in self.crossed_ids:
                continue  # 이미 라인을 넘은 객체는 무시
            

            current_state = self.tracker_state.get(tracker_id)
            
            #When first detected
            if current_state is None:
                self.tracker_state[tracker_id] = self._is_crossing(bounding_box, None)
                continue
            
            if self._is_crossing(bounding_box, current_state):
                
                if current_state:
                    out_count += 1
                else:
                    in_count += 1
                    
                self.tracker_state[tracker_id] = not current_state #sign reverse
                self.crossed_ids.add(tracker_id)  # 객체가 라인을 넘으면 ID를 저장

class customLineZoneAnnotator:
    
    def __init__(
        self,
        line_color=(0, 255, 255),
        thickness=2
        ):
        
        self.line_color = line_color
        self.thickness = thickness

    def annotate(
        self,
        frame: np.ndarray,
        line_zone: customLineZone
        ) -> np.ndarray:
        
        #Draw Line
        cv2.line(frame, line_zone.start, line_zone.end, self.line_color, self.thickness)
        
        return frame

def process_frame(
    frame: np.ndarray, 
    index: int,
    model, 
    tracker, 
    box_annotator,
    label_annotator,
    line_zone,
    line_annotator,
    conf_thres: float,
    classes: list
    ) -> np.ndarray:
    
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
    
    #dict {tracker_id:Bounding box}
    tracked_objects = {tracker_id: bbox for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy)}
    
    line_zone.trigger(tracked_objects)
        
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
        f"IN Count: {in_count}",
        (80, 100), #location
        cv2.FONT_HERSHEY_SIMPLEX, #font
        2,
        (0, 255, 0), #text color
        2,
        cv2.LINE_AA
        )
    
    cv2.putText(
        annotated_frame,
        f"OUT Count: {out_count}",
        (80, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        2,
        cv2.LINE_AA
        )
    
    return annotated_frame

def main():
    #Load model
    weights='yolov8s.pt'
    
    model = YOLO(weights)
    
    #init
    print('Initializing...')
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    input_filename='C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/3_Input 남측_G88/16-19/alwa_20240529_185759_F.mp4' #1
    input_filename="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/8_Output 북측_G96/16-19/alwa_20240529_164029_F.mp4" #1
    input_filename="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/7_Output 남측_G95/16-19/alwa_20240529_181815_F.mp4" #2
    input_filename="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/6_Output 서측_G89/16-19/alwa_20240529_162522_F.mp4" #3
    input_filename="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/5_Output 동측_G91/16-19/alwa_20240529_181533_F.mp4" #4
    input_filename="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/2_Input 서측_G93/16-19/alwa_20240529_180041_F.mp4"
    
    start_point, end_point=(596, 766),(1573, 522) #1
    start_point,end_point= (254, 440), (1151, 610) #1
    start_point,end_point= (626, 442), (1651, 795) #2
    start_point,end_point= (282, 295), (1395, 317) #3
    start_point,end_point= (199, 425), (1016, 688) #3
    start_point,end_point= (828, 634), (1622, 484) #3

    output_filename='result'
    
    conf_thres = 0.25  #confidence threshold
    classes = [2, 3, 5, 7]  #class filtering
    stride=3
        
    #capturing from webcam
    cap = cv2.VideoCapture(input_filename)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    crossed_ids = set()  # 모든 라인 존이 공유하는 crossed_ids 집합

    line_zones = customLineZone(start=start_point, end=end_point, crossed_ids=crossed_ids)
    line_annotator = customLineZoneAnnotator()
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    else:
        #frame width
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #frame height
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        print(f"Webcam resolution: {int(width)}x{int(height)}")
        
        output_filename = getNextFilename(base_name=output_filename, extension='mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  #Codec
        out = cv2.VideoWriter(output_filename, fourcc, 15.0, (width, height))
        
    
    index = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if index%stride==0:
            annotated_frame = process_frame(
                frame, 
                index,
                model, 
                tracker,
                box_annotator,
                label_annotator,
                line_zones,
                line_annotator,
                conf_thres,
                classes
            )
            
            cv2.imshow('Webcam', annotated_frame)
            
            out.write(annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        index += 1
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
