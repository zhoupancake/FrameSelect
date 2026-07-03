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
    video_path = r"/mnt/8T/zwl/HUAWEI/test_data/14_视频_街边雕塑_行人经过.mp4"
    output_dir_pose = r"/mnt/8T/zwl/HUAWEI/YOLO/final/pose_debug"
    output_dir_tracker = r"/mnt/8T/zwl/HUAWEI/YOLO/final/tracker_output"
    best_face_path = r"/mnt/8T/zwl/HUAWEI/YOLO/final/best_face.png"

    # 加载两个模型（设备可相同）
    device = "cuda"
    pose_extractor = YOLOFrontalExtractor(model_path="yolov8n-pose.pt", device=device)
    tracker = YOLOv8nTracker(model_path="yolov8n.engine", device=device)  # 或 .pt

    # 首帧检测决定运行模式
    has_person = detect_person_in_first_frame(video_path, det_model_path="yolov8n.pt", device=device)

    if has_person:
        print("[INFO] 首帧检测到人物，启动双线程：姿态估计 + 干扰物识别")
        # 线程 A：姿态估计（会保存调试图，并返回最佳帧）
        best_frame_holder = [None]  # 用于在线程间传递结果
        def pose_task():
            idx, frame = pose_extractor.process_video(video_path, debug_dir=output_dir_pose)
            best_frame_holder[0] = frame

        # 线程 B：干扰物识别
        def tracker_task():
            tracker.process_video(video_path, output_dir=output_dir_tracker)

        t1 = threading.Thread(target=pose_task)
        t2 = threading.Thread(target=tracker_task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 保存最佳正脸
        if best_frame_holder[0] is not None:
            cv2.imwrite(best_face_path, best_frame_holder[0])
            print(f"[SAVE] 最佳正脸已保存至 {best_face_path}")
    else:
        print("[INFO] 首帧未检测到人物，仅执行干扰物识别")
        tracker.process_video(video_path, output_dir=output_dir_tracker)

if __name__ == "__main__":
    main()
    