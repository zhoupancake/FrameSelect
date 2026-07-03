# FrameSelect weight configuration
YOLO_DET_ENGINE_PATH = r"./yolov8n-seg.engine"
YOLO_POSE_ENGINE_PATH = r'/yolov8n-pose.engine'

# YOLO detection configuration
YOLO_DET_TARGET_CLS = [0, 2, 5, 7]
YOLO_DET_CLASS_NAMES = {
    0: 'person',
    2: 'car',
    5: 'bus',
    7: 'truck'
}
YOLO_DET_IMGSZ = 640
YOLO_DET_CONF_THRESHOLD = 0.4
YOLO_DET_IOU_THRESHOLD = 0.65
YOLO_DET_TRACKER_CONFIG = 

#YOLO pose configuration




