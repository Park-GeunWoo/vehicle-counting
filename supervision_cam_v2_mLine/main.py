import cv2
import time
import numpy as np
import supervision as sv
from utils.file_utils import getNextFilename
from models.model_loader import load_model
from tracking.trace_annotator import TraceAnnotator
from tracking.line_zone import LineZone, LineZoneAnnotator
from data.class_names import class_names

in_count = 0
out_count = 0
counted_tracker_ids = set()

def process_frame(frame, index, model, tracker, conf_thres, line_zones, box_annotator, label_annotator, trace_annotator, line_annotator, classes, smoother):
    global in_count, out_count
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is None:
        return frame

    detections = detections[np.isin(detections.class_id, classes)]
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
                check_line_crossing_multiple_zones(tracker_id, previous_coordinates, line_zone)

    for track in tracker.lost_tracks:
        if track.state == TrackState.Lost:
            predicted_coords = track.mean[:2]
            center_x, center_y = int(predicted_coords[0]), int(predicted_coords[1])
            trace_annotator.update_trace(track.external_track_id, (center_x, center_y), predicted=True)
            previous_coordinates = trace_annotator.trace_data.get(track.external_track_id)
            if previous_coordinates and len(previous_coordinates) > 2:
                for line_zone in line_zones:
                    check_line_crossing_multiple_zones(track.external_track_id, previous_coordinates, line_zone)

    for track in tracker.removed_tracks:
        trace_annotator.remove_trace(track.external_track_id)

    labels = [f"#{tracker_id} {class_names.get(class_id, 'Unknown')} {confidence:.2f}" for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)]
    
    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = trace_annotator.annotate(annotated_frame)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    for line_zone in line_zones:
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)

    cv2.putText(annotated_frame, f"IN Count: {in_count} OUT Count: {out_count}", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 255), 2)
    return annotated_frame

def main():
    model = load_model()
    tracker = sv.ByteTrack()
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()
    trace_annotator = TraceAnnotator()
    line_annotator = LineZoneAnnotator()
    smoother = sv.DetectionsSmoother()
    
    input_path = 'input_video.mp4'
    output_filename = 'result'
    line_zones = [LineZone(start=(80, 401), end=(1142, 844))]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print('Could not open webcam')
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = int(1000 / fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = getNextFilename(base_name=output_filename, extension='mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))

    index = 1
    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        if index % 2 == 0:
            annotated_frame = process_frame(frame, index, model, tracker, 0.25, line_zones, box_annotator, label_annotator, trace_annotator, line_annotator, [2, 3, 5, 7], smoother)
            current_time = time.time()
            fps = 2 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('cv2', annotated_frame)
            out.write(annotated_frame)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

        index += 1

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
