import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Tuple, Dict

from class_names import class_names

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
        end: Tuple[int, int]
        ):
        
        self.start = start
        self.end = end
        self.in_count = 0
        self.out_count = 0
        self.tracker_state: Dict[int, bool] = {}
        
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
        
        for tracker_id, bounding_box in detections.items():
            
            current_state = self.tracker_state.get(tracker_id)
            
            #When first detected
            if current_state is None:
                self.tracker_state[tracker_id] = self._is_crossing(bounding_box, None)
                continue
            
            if self._is_crossing(bounding_box, current_state):
                
                if current_state:
                    self.out_count += 1
                else:
                    self.in_count += 1
                    
                self.tracker_state[tracker_id] = not current_state #sign reverse

class customLineZoneAnnotator:
    
    def __init__(
        self,
        line_color=(0, 255, 255), 
        text_color=(255, 255, 255), 
        thickness=2,
        text_scale=1,
        text_thickness=4
        ):
        
        self.line_color = line_color
        self.text_color = text_color
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness

    def annotate(
        self,
        frame: np.ndarray,
        line_zone: customLineZone
        ) -> np.ndarray:
        
        #Draw Line
        cv2.line(frame, line_zone.start, line_zone.end, self.line_color, self.thickness)
        
        #Show counts
        cv2.putText(
            frame,
            f"IN Count: {line_zone.in_count}",
            (80, 100), #location
            cv2.FONT_HERSHEY_SIMPLEX, #font
            self.text_scale,
            (0, 255, 255), #text color
            self.text_thickness,
            cv2.LINE_AA
            )
        
        cv2.putText(
            frame,
            f"OUT Count: {line_zone.out_count}",
            (80, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            (0, 0, 255),
            self.text_thickness,
            cv2.LINE_AA
            )
        
        return frame

def process_frame(
    frame: np.ndarray, 
    index: int,
    model, 
    tracker, 
    smoother, 
    box_annotator,
    label_annotator,
    line_zone,
    line_annotator,
    conf_thres: float,
    #classes: list
    ) -> np.ndarray:
    
    #Load frame
    results = model(frame)[0]
    
    detections = sv.Detections.from_ultralytics(results) #Convert Yolo results to Detections
    
    detections=tracker.update_with_detections(detections)
    
    #no detected objects
    if detections.tracker_id is None:
        print(f"No objects detected")
        return frame
    
    #detections=detections[np.isin(detections.class_id, classes) ]
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
    
    return annotated_frame

def main():
    #Load model
    weights='yolov5xu.pt'
    
    model = YOLO(weights)
    
    #init
    print('Initializing...')
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    #capturing from webcam
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    output_filename='result'
    #Line position
    start, end = (640, 100), (640, 620)
    
    conf_thres = 0.45  #confidence threshold
    classes = [2, 3, 5, 7]  #class filtering
    
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
        fourcc = cv2.VideoWriter_fourcc(*'X264')  #Codec
        out = cv2.VideoWriter(output_filename, fourcc, 15.0, (width, height))
        
        
    line_zone = customLineZone(start=start, end=end)
    line_annotator = customLineZoneAnnotator()
    
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = process_frame(
            frame, 
            index,
            model, 
            tracker, 
            smoother, 
            box_annotator,
            label_annotator,
            line_zone,
            line_annotator,
            conf_thres,
            #classes
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
