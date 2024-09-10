import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Dict

from utils import getNextFilename
from class_names import class_names
from supervision.tracker.byte_tracker.basetrack import TrackState


def main():
    weights = 'yolov5nu.pt'  # YOLO 모델 가중치
    model = YOLO(weights)
    model.to('cuda')  # GPU 사용

    Cam=False

    input_path="C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/8_Output 북측_G96/16-19/alwa_20240529_164029_F.mp4" #1

    output_filename = 'result'
    frame_batch = []
    conf_thres = 0.25
    batch_size = 4  # 배치 크기 (4프레임)

    if Cam:
        cap = cv2.VideoCapture(0)  # 웹캠 사용
    else:
        cap=cv2.VideoCapture(input_path)
        
    if not cap.isOpened():
        print('Could not open webcam')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f'Webcam resolution: {width}x{height}')
    
    output_filename = getNextFilename(base_name=output_filename, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (1280, 720))  # 해상도 변경
        frame_batch.append(frame)  # 배치에 프레임 추가

        if len(frame_batch) == batch_size:  # 배치 크기만큼 처리
            results = model.track(
                frame_batch,
                batch=batch_size,  # 배치 크기 설정
                conf=conf_thres,  # confidence 임계값
                iou=0.5,  # IOU 임계값
                tracker='bytetrack.yaml',
                persist=True
            )

            frame_batch = []  # 배치 초기화

            # 각 결과에 대해 처리
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    print("No objects detected")
                    continue

                # 결과 시각화
                annotated_frame = result.plot()

                # 결과 프레임 화면에 표시 및 저장
                cv2.imshow('YOLOv8 Tracking', annotated_frame)
                out.write(annotated_frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
