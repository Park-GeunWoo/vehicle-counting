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

from server.client import send_data_to_server
from utils.print_args import print_args
from utils.file_utils import getNextFilename

from line_zone import LineZone,process_detections

from utils.roi_scaler import scale_roi
from utils.line_zones_scaler import scale_line_zones

from data.data_store import in_count,out_count,update

from annotator.annotator import TraceAnnotator,LineZoneAnnotator
from ultralytics import solutions

class YOLOTracker:
    def __init__(self, args):
        self.args = args
        
        self.cam = args.cam    
        self.roi = args.roi
        
        self.model = YOLO('yolov8n.pt')
        self.model.export(format="engine", dynamic=True, batch=1, workspace=4, half=True)
        self.model = YOLO('yolov8n.engine')

        self.tracker = sv.ByteTrack(
            track_activation_threshold=args.track_thres,
            lost_track_buffer=args.track_buf,
            minimum_matching_threshold=args.mm_thres,
            frame_rate=args.f_rate,
            minimum_consecutive_frames=args.min_frames
        )
        
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.smoother = sv.DetectionsSmoother()
        
        self.trace_annotator = TraceAnnotator(trace_length=args.trace_len)
        self.line_annotator = LineZoneAnnotator()
        
        self.input_path = '2.mp4'#args.input
        self.output_path = args.output
        
        self.width = args.width
        self.height = args.height
        self.video_fps = args.video_fps
        
        self.conf_thres = args.conf_thres

        self.line_zones = []
        self.roi_points = None
        
        self.interval=60


    def set_line_zones(self, line_zone_points):
        scaled_line_zones=scale_line_zones(line_zone_points,self.width,self.height)
        print(f'asdf{self.line_zones}')
        '''roi여부, roi 포인트가 있을때 라인존 스케일링 함수'''
        for start_x, start_y, end_x, end_y in scaled_line_zones:
            if self.roi and self.roi_points is not None:
                start_x-=self.roi_points[0]
                start_y-=self.roi_points[1]
                end_x-=self.roi_points[2]
                end_y-=self.roi_points[3]
            line_zone=LineZone(
                start=(start_x,start_y),
                end=(end_x,end_y)
            )
            self.line_zones.append(line_zone)



    def run(self):
        '''predict init'''
        
        if self.cam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.input_path)
            
        if not cap.isOpened():
            print('Could not open Cam or Video')
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.video_fps)
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps=int(cap.get(cv2.CAP_PROP_FPS))

        output_filename = getNextFilename(base_name=self.output_path, extension='mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
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
        
        #저장 해상도를 roi에 마춰 스케일링
        if self.roi:
            self.roi_points = scale_roi(self.roi_points, self.width, self.height)
            x1,y1,x2,y2=self.roi_points
            out = cv2.VideoWriter(output_filename, fourcc, self.video_fps, (x2-x1, y2-y1))
        else:
            self.roi_points=None
            out = cv2.VideoWriter(output_filename, fourcc, self.video_fps, (self.width, self.height))
        
        if not out.isOpened():
            print("Error: Could not open video writer.")
            return

        self.set_line_zones(line_zone_points)

        vid_info=f'Resolution:{self.width}x{self.height} FPS:{self.video_fps}'
        print(f'info:{vid_info}')
        
        index, avrg_fps, prev_time = 1, 0, time.time()#fps 타이머
        last_execution = time.time()#데이터 전송 타이머
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            if self.roi:
                x_min, y_min, x_max, y_max = self.roi_points
                frame = frame[y_min:y_max, x_min:x_max]
            
            annotated_frame = frame.copy()
            
            results = self.model(source=frame, conf=self.conf_thres)[0]
            
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[np.isin(detections.class_id, [1, 2, 3, 5, 7])]
            detections = self.tracker.update_with_detections(detections)
            detections = self.smoother.update_with_detections(detections)
            
            if detections.tracker_id is None:
                labels = process_detections(
                    detections=detections,
                    tracker=self.tracker,
                    trace_annotator=self.trace_annotator,
                    line_zones=self.line_zones
                )

                annotated_frame = self.trace_annotator.annotate(annotated_frame)
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
                annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
                
                for line_zone in self.line_zones:
                    annotated_frame = self.line_annotator.annotate(annotated_frame, line_zone)


            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            avrg_fps += fps
            prev_time = current_time

            #데이터 전송
            if time.time() - last_execution >= self.interval:
                send_data_to_server()
                last_execution = time.time()
                
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Count: {in_count[0]}", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 255), 2)
            
            cv2.imshow('cv2', annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            index += 1

        print(f'Total Frames: {index} Average FPS: {int(avrg_fps/index)}')
        cap.release()
        out.release()
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
    parser.add_argument("--video-fps", type=int, default=30.0, help="Webcam FPS setting")
    parser.add_argument("--roi", action="store_true")
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    tracker = YOLOTracker(opt)
    tracker.run()
