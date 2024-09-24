import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
import numpy as np
import supervision as sv
from utils.file_utils import getNextFilename

from data.class_names import class_names
from data.count import in_count,out_count

from models.model_loader import load_model
from track.trace_annotator import TraceAnnotator
from track.line_zone import LineZone, LineZoneAnnotator, check_line_crossing_multiple_zones

from supervision.tracker.byte_tracker.basetrack import TrackState

        
def process_frame(
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
    smoother,
    roi,
    roi_points
    ):
    
    if roi:
        x_min, y_min, x_max, y_max = roi_points
        frame = frame[y_min:y_max, x_min:x_max]
    else:
        frame = frame
        
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is None:
        return frame

    #detections = detections[np.isin(detections.class_id, classes)]
    detections = detections[detections.confidence > conf_thres]
    detections = smoother.update_with_detections(detections)

    for detection_idx in range(len(detections)):
        tracker_id = int(detections.tracker_id[detection_idx])
        x_min, y_min, x_max, y_max = detections.xyxy[detection_idx]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        trace_annotator.update_trace(tracker_id, (center_x, center_y), predicted=False)
        previous_coordinates = trace_annotator.trace_data.get(tracker_id)

        if previous_coordinates and len(previous_coordinates) > 2:
            for line_zone in line_zones:
                check_line_crossing_multiple_zones(tracker_id, previous_coordinates, line_zones)

    for track in tracker.lost_tracks:
        if track.state == TrackState.Lost:
            predicted_coords = track.mean[:2]
            center_x, center_y = int(predicted_coords[0]), int(predicted_coords[1])
            trace_annotator.update_trace(track.external_track_id, (center_x, center_y), predicted=True)
            previous_coordinates = trace_annotator.trace_data.get(track.external_track_id)
            if previous_coordinates and len(previous_coordinates) > 2:
                for line_zone in line_zones:
                    check_line_crossing_multiple_zones(track.external_track_id, previous_coordinates, line_zones)

    for track in tracker.removed_tracks:
        trace_annotator.remove_trace(track.external_track_id)

    labels = [f"#{tracker_id} {class_names.get(class_id, 'Unknown')} {confidence:.2f}" for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)]
    
    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = trace_annotator.annotate(annotated_frame)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    for line_zone in line_zones:
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)

    
    return annotated_frame


def main():
    model = load_model()
    tracker = sv.ByteTrack(
        track_activation_threshold= 0.55,
        lost_track_buffer= 30,
        minimum_matching_threshold= 0.8,
        frame_rate= 10,
        minimum_consecutive_frames = 1
    )
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()
    trace_annotator = TraceAnnotator(trace_length=5)
    line_annotator = LineZoneAnnotator()
    smoother = sv.DetectionsSmoother()
    
    global in_count, out_count
    
    Cam=False
    width,height=1920,1800
    Cam_fps=30 #1080 30, 720 60
    
    input_path ="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/1_Input 동측_G87/16-19/alwa_20240529_185838_F.mp4"
    output_filename = 'result'
    
    roi=True
    roi_points=(577,377,1904,1042)
    x1,y1,x2,y2=roi_points
    
    line_zone_points=[
        (1897, 611,1161, 945),
        (925,805,1620,478),
        (729,612,1397,411)
        ]
    
    line_zones = []
    
    for point in line_zone_points:
        
        start_x, start_y, end_x, end_y = point
        if roi:
            line_zone = LineZone(
                start=(start_x - x1, start_y - y1),
                end=(end_x - x1, end_y - y1)
            )
        else:
            line_zone = LineZone(
                start=(start_x, start_y),
                end=(end_x, end_y)
            )
        
        line_zones.append(line_zone)
    
    class_filtering=True
    classes=[2,3,5,7]
    conf_thres=0.25
    stride=3
    


    if Cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    
    
    if not cap.isOpened():
        print('Could not open Cam or Video')
        return

    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, Cam_fps)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'Webcam info : {width}x{height} FPS:{Cam_fps}')
    
    output_filename = getNextFilename(base_name=output_filename, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))

    index = 1
    avrg_fps=0
    fps=0
    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        if index % stride == 0:
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
                smoother,
                roi,
                roi_points
                )

            current_time = time.time()
            fps = stride / (current_time - prev_time)
            avrg_fps+=fps
            prev_time = current_time
            
            
            cv2.putText(
                annotated_frame,
                f'FPS: {fps:.2f}', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
                )
            
            cv2.putText(
                annotated_frame, 
                f"IN Count: {in_count[0]} OUT Count: {out_count[0]}", 
                (80, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (125, 0, 255),
                2
                )
            
            cv2.imshow('cv2', annotated_frame)
            
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        index += 1

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(avrg_fps/index)
    
if __name__ == "__main__":
    main()
