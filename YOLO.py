import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import threading
import os
import queue

from config import *


class YOLOv8nTracker:
    def __init__(self, model_path="yolov8n.engine", device="cuda"):
        self.model = YOLO(model_path)
        self.device = device
        try:
            self.model.to(device)
        except:
            pass
        self.main_track_id = None
        self.is_first_frame = True
        print(f"[INFO] YOLO 模型加载完成，设备: {device}")

    def _run_yolo_track(self, frame):
        """
        执行检测 + 跟踪，返回 (原始框, 类别, 置信度, track_id)
        """
        h, w = frame.shape[:2]
        # 缩放至模型输入尺寸
        resized = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
        
        results = self.model.track(resized, 
                                   conf=YOLO_CONF_THRESHOLD, 
                                   iou=YOLO_IOU_THRESHOLD,
                                   half=True,
                                   persist=True,  # 维持 ID
                                   tracker=YOLO_TRACKER_CONFIG,
                                   verbose=False)
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # 检查是否有 ID (跟踪器可能还没初始化)
        if boxes.id is None:
            return []

        # 坐标映射回原始尺寸
        scale_x = w / YOLO_IMGSZ
        scale_y = h / YOLO_IMGSZ
        detections = []
        
        # 提取数据
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cls = cls_ids[i]
            if cls not in YOLO_TARGET_CLS:
                continue
            x1, y1, x2, y2 = xyxy[i]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            conf = float(confs[i])
            track_id = track_ids[i]
            detections.append((x1, y1, x2, y2, cls, conf, track_id))
        return detections

    def select_main_character(self, detections, frame_shape):
        """
        选定主角：优先选择距离画面中心最近的行人 (class 0)
        """
        h, w = frame_shape[:2]
        center_x, center_y = w / 2, h / 2
        best_id = None
        best_dist = float('inf')

        for (x1, y1, x2, y2, cls, conf, track_id) in detections:
            if cls != 0:  # 只选行人作为主角，不选车
                continue
            # 计算框中心到画面中心距离
            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2
            dist = (box_cx - center_x) ** 2 + (box_cy - center_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def process_video(self, video_path, output_dir="distractor_detection",
                      save_format='jpg', jpeg_quality=85, resize_scale=0.5):
        """
        处理视频：只显示除主角外的干扰物
        """
        # 重置跟踪状态
        self.main_track_id = None
        self.is_first_frame = True

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息：总帧数 {total_frames}，分辨率 {int(cap.get(3))}x{int(cap.get(4))}")

        # 保存线程
        save_queue = queue.Queue(maxsize=64)
        stop_save = threading.Event()

        def saver():
            while not stop_save.is_set() or not save_queue.empty():
                try:
                    idx, img = save_queue.get(timeout=0.1)
                    if img is None:
                        continue
                    ext = 'jpg' if save_format == 'jpg' else 'png'
                    if ext == 'jpg':
                        cv2.imwrite(str(output_path / f"frame_{idx:06d}.jpg"), img,
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    else:
                        cv2.imwrite(str(output_path / f"frame_{idx:06d}.png"), img)
                except queue.Empty:
                    continue

        save_thread = threading.Thread(target=saver, daemon=True)
        save_thread.start()

        pbar = tqdm(total=total_frames, desc="处理帧")
        frame_idx = 0
        lost_frames = 0  # 主角跟丢计数

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            
            # 1. 执行检测 + 跟踪
            detections = self._run_yolo_track(frame)

            # 2. 首帧或主角丢失时重新选定
            if self.is_first_frame and detections:
                self.main_track_id = self.select_main_character(detections, frame.shape)
                self.is_first_frame = False
                if self.main_track_id is not None:
                    print(f"[INFO] 主角 ID 已选定: {self.main_track_id} (画面中心最近的行人)")

            # 3. 如果主角跟丢超过一定帧数，重置状态重新选
            if self.main_track_id is not None:
                found = any(t_id == self.main_track_id for (_, _, _, _, _, _, t_id) in detections)
                if not found:
                    lost_frames += 1
                    if lost_frames > 30:  # 跟丢超过 30 帧，重新选主角
                        print(f"[INFO] 主角丢失超过 30 帧，重新选定")
                        self.is_first_frame = True
                        self.main_track_id = None
                        lost_frames = 0
                else:
                    lost_frames = 0

            # 4. 绘制检测框（过滤掉主角）
            for (x1, y1, x2, y2, cls, conf, track_id) in detections:
                # 如果当前框是主角，则跳过不画（也不输出）
                if self.main_track_id is not None and track_id == self.main_track_id:
                    continue
                
                # 画干扰物的框（红色，便于区分）
                label = f"{CLASS_NAMES.get(cls, str(cls))} ID:{track_id} {conf:.2f}"
                color = (0, 0, 255)  # 红色框表示干扰物
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 5. 可选：在画面上标记主角位置（紫色虚框，方便调试）
            if self.main_track_id is not None:
                for (x1, y1, x2, y2, cls, conf, track_id) in detections:
                    if track_id == self.main_track_id:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)  # 紫色细框
                        cv2.putText(frame, "MAIN", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        break

            # 缩放保存
            if resize_scale != 1.0:
                new_h, new_w = int(h * resize_scale), int(w * resize_scale)
                frame = cv2.resize(frame, (new_w, new_h))

            save_queue.put((frame_idx, frame))
            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        stop_save.set()
        save_thread.join()
        print(f"完成！干扰物检测结果已保存至 {output_dir}")

