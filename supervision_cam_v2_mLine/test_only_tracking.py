import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import argparse
import cv2
import time
import subprocess
from datetime import datetime  # 시간을 포맷하기 위한 모듈
import torch
import subprocess
import numpy as np

from ultralytics import YOLO
import supervision as sv

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
            
from utils.print_args import print_args
from utils.file_utils import getNextFilename

from line_zone import LineZone,process_detections

from utils.roi_scaler import scale_roi
from utils.line_zones_scaler import scale_line_zones
from data.data_store import in_count,out_count,update

from models.model_loader import load_model

from annotator.annotator import TraceAnnotator,LineZoneAnnotator
from ultralytics import solutions


classes=[1,2,3,5,7]
def process_frame(
    frame,
    index, 
    model, 
    tracker,
    smoother,
    conf_thres, 
    box_annotator, 
    label_annotator,
    roi,
    roi_points
    ):
    global classes
    if roi:
        x_min, y_min, x_max, y_max = roi_points
        frame = frame[y_min:y_max, x_min:x_max]
    
    annotated_frame = frame.copy()

    results = model.predict(source=frame, conf=conf_thres)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections=detections[np.isin(detections.class_id,classes)]
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    
    if detections.tracker_id is None:
        return annotated_frame

    annotated_frame = box_annotator.annotate(annotated_frame,detections)

    return annotated_frame


def main(
    weights,
    input,
    output='result',
    cam=False,
    loc_name='Korea',
    v_filtering=False,
    conf_thres=0.25,
    iou=0.55,
    id='0',
    stride=1,
    track_thres=0.55,
    track_buf=30,
    mm_thres=0.8,
    f_rate=10,
    min_frames=3,
    trace_len=5,
    width=1920,
    height=1080,
    video_fps=30,
    roi=False
    ):
    model = YOLO("yolov8n.pt")

    tracker = sv.ByteTrack(
        track_activation_threshold=track_thres,
        lost_track_buffer=track_buf, #몇 프레임 동안 트래킹 할지
        minimum_matching_threshold=mm_thres,
        frame_rate=f_rate, #초당 몇 프레임 트래킹 할지
        minimum_consecutive_frames = min_frames
    )
    
    
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()
    smoother = sv.DetectionsSmoother()
    
    input_path ="input서측18.mp4"
    #print(cam)
    if cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print('Could not open Cam or Video')
        return
    print(f'asdfasdf{video_fps}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, video_fps)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x1,y1,x2,y2=0,0,0,0
    output_filename = getNextFilename(base_name=output, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    roi_points=None
    out = cv2.VideoWriter(output_filename, fourcc, 60.0, (width, height))
    video_fps=int(cap.get(cv2.CAP_PROP_FPS))

    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

        
    vid_info=f'Resolution:{width}x{height} FPS:{video_fps}'
    print(f'info:{vid_info}')

    global in_count, out_count
    
    index = 1
    avrg_fps=0
    fps=0
    prev_time = time.time()
    start_time=time.time()
    last_send_time = start_time
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if index % stride == 0:
            annotated_frame= process_frame(
                frame, 
                index,
                model,
                tracker,
                smoother,
                conf_thres,
                box_annotator,
                label_annotator,
                roi,
                roi_points
                )

            current_time = time.time()
            fps = stride / (current_time - prev_time)
            avrg_fps+=fps
            prev_time = current_time
            
            
            cv2.putText(
                annotated_frame,
                f'FPS: {fps:.1f}', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
                )
            
            cv2.putText(
                annotated_frame,
                f"Count: {in_count[0]}", 
                (80, 150), 
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

    end_time=time.time()
    
    print(f'Total Frames:{index} Average FPS:{int(avrg_fps/index)}, Total Time :{int(end_time-start_time)}s')
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default='yolov8n.pt')
    parser.add_argument("--input", type=str, default='video.mp4')
    parser.add_argument("--output", type=str, default='result')
    parser.add_argument("--cam", action="store_true")
    parser.add_argument("--loc-name", type=str,default='Korea',help='location_name')
    parser.add_argument("--v-filtering", action='store_true')
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou", type=float,default=0.65)
    parser.add_argument("--id", type=str,default='0')
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--track-thres", type=float, default=0.55,help='track_activation_threshold')
    parser.add_argument("--track-buf",type=int,default=60)
    parser.add_argument("--mm-thres",type=float,default=0.8,help='minimum_matching_threshold')
    parser.add_argument("--f-rate",type=int,default=20,help='frame rate')
    parser.add_argument("--min-frames",type=int,default=2,help='minimum_consecutive_frames')
    parser.add_argument("--trace-len",type=int,default=5)
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--video-fps", type=int, default=60, help="Webcam FPS setting")
    parser.add_argument("--roi", action="store_true")
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return parser.parse_args()

if __name__ == "__main__":
    opt=parse_opt()
    main(**vars(opt))
