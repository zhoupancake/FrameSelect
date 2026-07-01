# FrameSelect weight configuration
YOLO_ENGINE_PATH = r"/mnt/8T/zwl/HUAWEI/yolov8n.engine"


# YOLO configuration
YOLO_TARGET_CLS = [0, 2, 5, 7]
YOLO_CLASS_NAMES = {
    0: 'person',
    2: 'car',
    5: 'bus',
    7: 'truck'
}
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.4
YOLO_IOU_THRESHOLD = 0.65
YOLO_TRACKER_CONFIG = "bytetrack.yaml"

