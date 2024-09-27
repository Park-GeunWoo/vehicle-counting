import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import time
import numpy as np
import supervision as sv

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
            
from utils.file_utils import getNextFilename
from utils.roi_scaler import scale_roi
from utils.line_zones_scaler import scale_line_zones
from utils.print_args import print_args

from data.class_names import class_names
from data.count import in_count,out_count

from models.model_loader import load_model
from annotator.trace_annotator import TraceAnnotator
from annotator.LineZoneAnnotator import LineZoneAnnotator

from line_zone import LineZone,  check_line_crossing_multiple_zones

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


def main(
    weights,
    input,
    output='result',
    cam=False,
    v_filtering=False,
    conf_thres=0.25,
    stride=1,
    track_thres=0.55,
    track_buf=30,
    mm_thres=0.8,
    f_rate=10,
    min_frames=2,
    trace_len=10,
    width=1920,
    height=1080,
    video_fps=30,
    roi=False
    ):
    
    model = load_model(weights)
    
    tracker = sv.ByteTrack(
        track_activation_threshold=track_thres,
        lost_track_buffer=track_buf, #몇 프레임 동안 트래킹 할지
        minimum_matching_threshold=mm_thres,
        frame_rate=f_rate, #초당 몇 프레임 트래킹 할지
        minimum_consecutive_frames = min_frames
    )
    
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()
    trace_annotator = TraceAnnotator(trace_length=trace_len)
    line_annotator = LineZoneAnnotator()
    smoother = sv.DetectionsSmoother()
    
    input_path ="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/1_Input 동측_G87/16-19/alwa_20240529_185838_F.mp4"
    
    if cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print('Could not open Cam or Video')
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, video_fps)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if roi:
        roi_points=(577,377,1904,1042)
        roi_points = scale_roi(roi_points, width, height)
        x1,y1,x2,y2=roi_points
    else:
        roi_points=None
        
    
    line_zone_points=[
        (1897, 611,1161, 945),
        (925,805,1620,478),
        (729,612,1397,411)
        ]
    
    scaled_line_zones = scale_line_zones(line_zone_points, width, height)
    
    if v_filtering:
        classes=[2,3,5,7]

            
    line_zones = []
    for start_x, start_y, end_x, end_y in scaled_line_zones:
        if roi:
            line_zone=LineZone(
                start=(start_x-x1,start_y-y1),
                end=(end_x-x1,end_y-y1)
                )
        else:
            line_zone = LineZone(
                start=(start_x, start_y),
                end=(end_x, end_y)
            )
            
        line_zones.append(line_zone)
        
    print(f'info : {width}x{height} FPS:{video_fps}')
    
    output_filename = getNextFilename(base_name=output, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))

    global in_count, out_count
    
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

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--weights", nargs="+", type=str, default='yolov8n.pt')
    parser.add_argument("--input", type=str, default='video.mp4')
    parser.add_argument("--output", type=str, default='result')
    parser.add_argument("--cam", action="store_true")
    parser.add_argument("--v-filtering", action='store_true')
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--track-thres", type=float, default=0.55,help='track_activation_threshold')
    parser.add_argument("--track-buf",type=int,default=30)
    parser.add_argument("--mm-thres",type=float,default=0.8,help='minimum_matching_threshold')
    parser.add_argument("--f-rate",type=int,default=10,help='frame rate')
    parser.add_argument("--min-frames",type=int,default=2,help='minimum_consecutive_frames')
    parser.add_argument("--trace-len",type=int,default=10)
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--video-fps", type=int, default=30, help="Webcam FPS setting")
    parser.add_argument("--roi", action="store_true")
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return parser.parse_args()

if __name__ == "__main__":
    opt=parse_opt()
    main(**vars(opt))
