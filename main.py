import os
from YOLO import YOLOv8nTracker
from config import *

import threading
import cv2
from pathlib import Path

# 导入已有类
from yolo_frontal_extractor import YOLOFrontalExtractor
from yolo_tracker import YOLOv8nTracker

def detect_person_in_first_frame(video_path, det_model_path="yolov8n.pt", device="cuda"):
    """使用检测模型检查首帧是否有人（仅用于决策）"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    # 用轻量检测（非跟踪）判断
    model = YOLO(det_model_path)
    results = model(frame, conf=0.3, verbose=False)  # 较低阈值
    if results[0].boxes is None:
        return False
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    return 0 in cls_ids  # COCO 中 person 的 class 为 0

def main():
    detector = YOLOv8nTracker(
        model_path=YOLO_ENGINE_PATH,
        device="cuda"
    )

    # ----- 批量处理 -----
    root_dir = None
    # 示例：处理编号为15的视频
    paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
             if name[:2].isdigit() and int(name[:2]) == 00]
    paths = sorted(paths, key=lambda s: int(os.path.basename(s)[:2]))
    print(f"待处理视频: {paths}")

    for idx, path in enumerate(paths):
        detector.process_video(
            video_path=path,
            output_dir=f'distractor_frames_00000',
            save_format='jpg',
            jpeg_quality=85,
            resize_scale=0.5
        )

if __name__ == "__main__":
    main()
    
