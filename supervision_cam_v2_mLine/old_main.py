import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import time
import subprocess
from datetime import datetime
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
    line_zones,
    box_annotator, 
    label_annotator, 
    trace_annotator,
    line_annotator,
    roi,
    roi_points
    ):
    global classes
    
    if roi:
        x_min, y_min, x_max, y_max = roi_points
        frame = frame[y_min:y_max, x_min:x_max]
    
    annotated_frame = frame.copy()

    results = model(source=frame, conf=conf_thres)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections=detections[np.isin(detections.class_id,classes)]
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    
    if detections.tracker_id is None:
        return annotated_frame

    labels = process_detections(
        detections=detections,
        tracker=tracker,
        trace_annotator=trace_annotator,
        line_zones=line_zones
        )

    annotated_frame = trace_annotator.annotate(annotated_frame)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    annotated_frame = box_annotator.annotate(annotated_frame,detections)
    for line_zone in line_zones:
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)

    
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
    model = YOLO('yolov8n.pt')
    #model.export(format="engine", batch=8, workspace=4, int8=True, data="coco.yaml")
    #model=load_model('yolov8n.engine')
    
    tracker = sv.ByteTrack(
        track_activation_threshold=track_thres,
        lost_track_buffer=track_buf, #몇 프레임 동안 트래킹 할지
        minimum_matching_threshold=mm_thres,
        frame_rate=f_rate, #초당 몇 프레임 트래킹 할지
        minimum_consecutive_frames = min_frames
    )
    
    polygonzone_points=[
        [670,400],
        [1920,400],
        [1920,1080],
        [670,1080]
        ]
    line_zone_points=[

        (890,680,1760,480), #서측
        (1110,865,1840,510),
        (1667,990,1910,530)
        # (640,760,1510,480), #남측
        # (1010,980,1690,520), 
        # (1730,980,1810,530)
        # (860,760,1650,440), #동
        # (1220,990,1750,440),
        # (1740,990,1840,450)
        # (810,770,1700,530), #북
        # (1150,1050,1830,570),
        # (1720,1050,1900,590)
        ]
    roi_points=(670,350,1920,1080) #x1,y1,x2,y2
    
    scaled_line_zones = scale_line_zones(line_zone_points, width, height)
    
    trace_annotator = TraceAnnotator(trace_length=trace_len)
    line_annotator = LineZoneAnnotator()
    
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()
    smoother = sv.DetectionsSmoother()
    
    input_path ="1.mp4"
    #print(cam)
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
    
    x1,y1,x2,y2=0,0,0,0
    output_filename = getNextFilename(base_name=output, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if roi:
        roi_points = scale_roi(roi_points, width, height)
        x1,y1,x2,y2=roi_points
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (x2-x1, y2-y1))
    else:
        roi_points=None
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

            
    line_zones = []
    for start_x, start_y, end_x, end_y in scaled_line_zones:
        if roi:
            start_x-=x1
            start_y-=y1
            end_x-=x1
            end_y-=y1
        
        line_zone = LineZone(
                start=(start_x, start_y),
                end=(end_x, end_y)
            )
        line_zones.append(line_zone)
        
    vid_info=f'Resolution:{width}x{height} FPS:{video_fps}'
    print(f'info:{vid_info}')

    global in_count, out_count
    
    index = 1
    avrg_fps=0
    fps=0
    prev_time = time.time()
    start_time=time.time()

    # formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         # 데이터 전송
    # update(
    #     edge_id=id,
    #     location_name=loc_name,
    #     gps=None,
    #     time=formatted_time,
    #     count=in_count[0]
    # )
    # subprocess.run(["python", "./server/client.py"])
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # if 3100<=index and index<=3200:
        #     cv2.imwrite(f'{index}-1.jpg', frame)
        
        if index % stride == 0:
            annotated_frame= process_frame(
                frame=frame,
                index=index,
                model=model,
                tracker=tracker,
                smoother=smoother,
                conf_thres=conf_thres,
                line_zones=line_zones,
                box_annotator=box_annotator,
                label_annotator=label_annotator,
                trace_annotator=trace_annotator,
                line_annotator=line_annotator,
                roi=roi,
                roi_points=roi_points
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
            
            #print(f'\rframe {index}',end='')
            cv2.imshow('cv2', annotated_frame)
            
            out.write(annotated_frame)

        # # 1분마다 전송
        # if current_time - last_send_time >= 60:
        #     print(f"Sending data at frame {index}, Time: {current_time - start_time:.2f} seconds")
        #     formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     # 데이터 전송
        #     update(
        #         edge_id=id,
        #         location_name=loc_name,
        #         gps=None,
        #         time=formatted_time,
        #         count=in_count[0]
        #     )
            
        #     subprocess.run(["python", "./server/client.py"])
        #     last_send_time = current_time
                
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
    parser.add_argument("--weights", nargs="+", type=str, default='yolo11x.pt')
    parser.add_argument("--input", type=str, default='video.mp4')
    parser.add_argument("--output", type=str, default='result')
    parser.add_argument("--cam", action="store_true")
    parser.add_argument("--loc-name", type=str,default='Korea',help='location_name')
    parser.add_argument("--v-filtering", action='store_true')
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou", type=float,default=0.55)
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
    parser.add_argument("--video-fps", type=int, default=30, help="Webcam FPS setting")
    parser.add_argument("--roi", action="store_true")
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return parser.parse_args()

if __name__ == "__main__":
    opt=parse_opt()
    main(**vars(opt))
