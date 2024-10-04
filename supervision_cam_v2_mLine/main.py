import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import time
import torch
import numpy as np
import supervision as sv
from supervision.draw.color import Color
from supervision.geometry.core import Position
from supervision.tracker.byte_tracker.basetrack import TrackState

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
            
from utils.print_args import print_args
from utils.file_utils import getNextFilename

from line_zone import LineZone,process_detections

from utils.roi_scaler import scale_roi
from utils.line_zones_scaler import scale_line_zones
from utils.gpu_usage import get_gpu_usage

from data.class_names import class_names
from data.count import in_count,out_count

from models.model_loader import load_model

from annotator.annotator import TraceAnnotator,LineZoneAnnotator

        
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
    #smoother,
    roi,
    roi_points
    ):
    
    if roi:
        x_min, y_min, x_max, y_max = roi_points
        frame = frame[y_min:y_max, x_min:x_max]
        
    # 추론 시간 측정을 위한 CUDA 이벤트 생성
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # 추론 시작
    with torch.no_grad():  # Gradient 연산 비활성화
        starter.record()  # 시작 시간 기록
        results = model(source=frame, conf=conf_thres)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        
        if detections.tracker_id is None:
            return frame

        labels = process_detections(
            detections=detections,
            tracker=tracker,
            trace_annotator=trace_annotator,
            line_zones=line_zones
            )
        
        gpu_util, mem_util, total_mem, used_mem, free_mem = get_gpu_usage()
        
        ender.record()  # 끝나는 시간 기록

    # GPU 연산이 끝날 때까지 대기
    torch.cuda.synchronize()
    # 추론 시간 계산 (milliseconds 단위)
    inference_time = starter.elapsed_time(ender) * 1e-3  # 초 단위로 변환
    


    annotated_frame = trace_annotator.annotate(frame)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    for line_zone in line_zones:
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)
    # 추론 시간 출력 (프레임에 표시)

    if gpu_util and mem_util:
        cv2.putText(
            annotated_frame,
            f'GPU Usage: {gpu_util}% | Mem: {used_mem}/{total_mem} MB',
            (10, 80),  # 텍스트 위치
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # 텍스트 크기
            (0, 200, 0),  # 텍스트 색상 (녹색)
            2  # 두께
        )
    
    return annotated_frame,inference_time


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
    min_frames=3,
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
    
    polygonzone_points=[
        [670,400],
        [1920,400],
        [1920,1080],
        [670,1080]
        # [450, 200],   # 왼쪽 상단
        # [1500,200],  # 오른쪽 상단
        # [1500,800], # 오른쪽 하단
        # [450,800]   # 왼쪽 하단
        ]
    line_zone_points=[
        (1897, 611,1161, 945),
        (925,805,1620,478),
        (729,612,1397,411)
        ]
    roi_points=(670,400,1920,1080) #x1,y1,x2,y2
    
    scaled_line_zones = scale_line_zones(line_zone_points, width, height)
    
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()
    trace_annotator = TraceAnnotator(trace_length=trace_len)
    line_annotator = LineZoneAnnotator()
    #smoother = sv.DetectionsSmoother()
    
    input_path ="C:/Users/USER/Desktop/alwa_20240529_185838_F.mp4"
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

    global in_count, out_count
    
    index = 1
    avrg_fps=0
    fps=0
    total_inf_time=0
    prev_time = time.time()
    start_time=time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if index % stride == 0:
            annotated_frame,inference_time= process_frame(
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
                #smoother,
                roi,
                roi_points
                )

            current_time = time.time()
            fps = stride / (current_time - prev_time)
            avrg_fps+=fps
            prev_time = current_time
            
            total_inf_time+=inference_time
            
            cv2.putText(
                annotated_frame,
                f'frame: {index} FPS: {fps:.1f} inf and track: {inference_time:.3f}s', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
                )
            
            cv2.putText(
                annotated_frame, 
                f"IN Count: {in_count[0]} OUT Count: {out_count[0]}", 
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
    
    print(f'Total Frames:{index} Average FPS:{int(avrg_fps/index)}, Total Time :{int(end_time-start_time)}s  Total Inf Time: {total_inf_time:.3f}')
    out.release()
    cap.release()
    cv2.destroyAllWindows()

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
