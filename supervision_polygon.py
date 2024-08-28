import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.draw.color import Color

def getNextFilename(base_name="result", extension="mp4"):
    i = 1
    while True:
        filename = f"{base_name}{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

def createPolygonZones(polygon_points_list):
    return [sv.PolygonZone(polygon=np.array(points)) for points in polygon_points_list]

def createZoneAnnotators(polygon_zones, colors):
    return [
        sv.PolygonZoneAnnotator(zone=zone, color=color)
        for zone, color in zip(polygon_zones, colors)
    ]

def callback(frame: np.ndarray, 
             index: int, 
             model, 
             tracker,
             smoother,
             box_annotator, 
             label_annotator,
             polygon_zones,
             zone_annotators) -> np.ndarray:
        '''
        stride값 위치 video.py
        '''
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)  # Detection
        detections = detections[detections.confidence > 0.5]  # confidence threshold
        detections = tracker.update_with_detections(detections)  # tracking
        detections = smoother.update_with_detections(detections)

        for zone in polygon_zones:
            zone.current_count = np.sum(zone.trigger(detections))

        print(f'Frame {index}')
        #감지된 객체가 없을떄
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            print("No objects detected")
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
            scene=frame.copy(),
            detections=detections)

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)


        for annotator in zone_annotators:
            annotated_frame = annotator.annotate(scene=annotated_frame)


        return annotated_frame
    
def main():
    model = YOLO('yolov5xu.pt') #model
    
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    #ROI
    polygon_points_list = [
        [[730, 720], [980, 690], [1532, 920], [1301, 1065]],
        [[988, 687], [1194, 662], [1650, 815], [1532, 910]],
        [[1213, 660], [1371, 630], [1751, 734], [1650, 810]],
        [[632, 760], [1391, 628], [1827, 744], [941, 1050]]
    ]
    colors = [Color(255, 0, 0),
              Color(0, 255, 0),
              Color(0, 0, 255),
              Color(255,255,255)]

    
    polygon_zones = createPolygonZones(polygon_points_list)
    zone_annotators = createZoneAnnotators(polygon_zones, colors)


    target_path = getNextFilename(base_name="result", extension="mp4") #output filename

    sv.process_video(source_path='videos/alwa_20240529_185759_F.mp4', #inputfile
                     target_path=target_path,
                     callback = lambda frame, index: callback(frame,
                                                              index, 
                                                              model, 
                                                              tracker,
                                                              smoother,
                                                              box_annotator, 
                                                              label_annotator,
                                                              polygon_zones,
                                                              zone_annotators)
                     )

if __name__ == '__main__':
    main()