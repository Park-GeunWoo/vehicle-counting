import os
os.environ['YOLO_VERBOSE'] = 'False' #yolo quiet mode
from ultralytics import YOLO

def load_model(weights='yolov8n.pt'):
    return YOLO(str(weights))
