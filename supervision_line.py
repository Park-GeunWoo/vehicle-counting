import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.draw.color import Color
from supervision.assets import download_assets, VideoAssets

def getNextFilename(base_name="result", extension="mp4"):
    i = 1
    while True:
        filename = f"{base_name}{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

def callback(frame: np.ndarray, 
             index: int,
             model, 
             tracker, 
             smoother, 
             box_annotator,
             label_annotator,
             line_zone,
             line_annotator) -> np.ndarray:
        '''
        stride값 위치 video.py
        '''
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results) #Detection
        detections = detections[detections.confidence>0.5] #confidence threshold
        detections = tracker.update_with_detections(detections) #tracking
        detections = smoother.update_with_detections(detections)

        line_zone.trigger(detections)

        print(f'Frame {index}')
        #감지된 객체가 없을 때
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            print(f"No objects detected")
            return frame
        
        
        labels = [
            f"#{tracker_id} {class_name} {confidence:.2f}"
            for tracker_id, class_name, confidence
            in zip(detections.tracker_id,detections.data['class_name'], detections.confidence)
        ]
        
        # for tracker_id, class_name, confidence in (
        #         zip(detections.tracker_id, detections.data['class_name'], detections.confidence)):
        #     print(f"ID {tracker_id}, Class : {class_name}, Confidence : {confidence:.2f}%")

        for label in labels:
            print(label)


        annotated_frame = box_annotator.annotate(
            scene=frame,
            detections=detections)

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

        annotated_frame=line_annotator.annotate(
            frame=annotated_frame,
            line_counter=line_zone)


        cv2.putText(annotated_frame,
                    f"LineZone IN Count: {line_zone.in_count}",
                    (30, 100), #org
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, #font scale
                    (255, 0, 0), #color
                    2, #font thickness
                    cv2.LINE_AA)
        
        cv2.putText(annotated_frame,
                    f"LineZone OUT Count: {line_zone.out_count}",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

        return annotated_frame


def main():
    #download_assets(VideoAssets.VEHICLES_2)
    model = YOLO('yolov5xu.pt')
    
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    #start, end = sv.Point(x=0, y=1245), sv.Point(x=3840, y=1245) #vehicles.mp4
    start, end = sv.Point(x=648, y=759), sv.Point(x=1662, y=541) #alwa_20240529_185759_F.mp4
    line_zone = sv.LineZone(start=start, end=end)
    line_annotator=sv.LineZoneAnnotator(color=Color(0,255,255),text_scale=2)


    input_path='C:/Users/USER/Downloads/20240529_서평택IC사거리 교통량조사(76G)/3_Input 남측_G88/16-19/alwa_20240529_185759_F.mp4'
    target_path = getNextFilename(base_name="result", extension="mp4")

    sv.process_video(source_path=input_path, #input file
                     #source_path='vehicles.mp4',
                     target_path=target_path,
                     callback = lambda frame, index: callback(frame, 
                                                              index,
                                                              model,
                                                              tracker, 
                                                              smoother,
                                                              box_annotator, 
                                                              label_annotator,
                                                              line_zone, 
                                                              line_annotator)
                     )


if __name__ == '__main__':
    main()