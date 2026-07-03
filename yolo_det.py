import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import os

class DetConfig:
    MODEL_PATH = r'./yolov8n.engine'
    DEVICE = 'cuda'
    TARGET_CLS = [0, 2, 5, 7]
    MAIN_CLS = [0, 15, 16]
    CLASS_NAMES = {
        0: 'person', 2: 'car', 5: 'bus', 7: 'truck',
        15: 'cat', 16: 'dog'
    }
    YOLO_IMGSZ = 640
    CONF_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.65
    TRACKER_CONFIG = "bytetrack.yaml"

# ===================== 检测跟踪器类 =====================
class YOLOv8nTracker:
    def __init__(self, model_path=DetConfig.MODEL_PATH, device=DetConfig.DEVICE):
        self.model = YOLO(model_path)
        self.device = device
        try:
            self.model.to(device)
        except:
            pass
        # 主体相关状态
        self.main_track_id = None
        self.is_first_frame = True
        self.lost_frames = 0
        self.has_main = False
        print(f"[INFO] YOLO 模型加载完成，设备: {device}")

    def _run_yolo_track(self, frame):
        """
        执行检测 + 跟踪，返回 (x1,y1,x2,y2,cls,conf,track_id)
        直接传入原始帧，由 model.track 内部处理 letterbox，坐标自动映射回原始尺寸
        """
        results = self.model.track(frame,
                                   conf=DetConfig.CONF_THRESHOLD,
                                   iou=DetConfig.IOU_THRESHOLD,
                                   half=True,
                                   persist=True,      # 维持 ID
                                   tracker=DetConfig.TRACKER_CONFIG,
                                   verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # 检查是否有跟踪 ID（跟踪器可能未初始化）
        if boxes.id is None:
            return []

        # 提取数据（坐标已是原始图像坐标系）
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        detections = []
        for i in range(len(xyxy)):
            cls = cls_ids[i]
            if cls not in DetConfig.TARGET_CLS:
                continue
            x1, y1, x2, y2 = xyxy[i].astype(int)
            conf = float(confs[i])
            track_id = track_ids[i]
            detections.append((x1, y1, x2, y2, cls, conf, track_id))
        return detections

    def select_main_character(self, detections, frame_shape):
        """
        从 detections 中选取属于 MAIN_CLS 且距离画面中心最近的目标
        返回其 track_id，若无则返回 None
        """
        h, w = frame_shape[:2]
        center_x, center_y = w / 2, h / 2
        best_id = None
        best_dist = float('inf')

        for (x1, y1, x2, y2, cls, conf, track_id) in detections:
            if cls not in DetConfig.MAIN_CLS:
                continue
            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2
            dist = (box_cx - center_x) ** 2 + (box_cy - center_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def process_video(self, video_path):
        """
        处理视频，返回每帧的检测结果列表。
        返回值: list of list, 外层按帧顺序，内层为该帧的检测框列表，
                每个框为 [x_rel, y_rel, w_rel, h_rel, cls, conf, track_id]
                所有坐标归一化到 [0,1]。
        若无主体，则返回所有目标；若有主体，则过滤掉主角 ID。
        """
        # 重置状态
        self.main_track_id = None
        self.is_first_frame = True
        self.lost_frames = 0
        self.has_main = False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频信息：总帧数 {total_frames}，分辨率 {width}x{height}")

        all_frames_results = []   # 存储每帧的结果
        pbar = tqdm(total=total_frames, desc="处理帧")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 1. 执行检测 + 跟踪
            detections = self._run_yolo_track(frame)

            # 2. 首帧或主角丢失时重新选定主角
            if self.is_first_frame and detections:
                # 尝试选择主体
                main_id = self.select_main_character(detections, frame.shape)
                if main_id is not None:
                    self.main_track_id = main_id
                    self.has_main = True
                    self.is_first_frame = False
                    self.lost_frames = 0
                    print(f"[INFO] 主角已选定，ID: {self.main_track_id}")
                else:
                    # 首帧无主体，放弃主体识别，后续不再尝试
                    self.has_main = False
                    self.is_first_frame = False
                    print("[INFO] 首帧未检测到主体，将显示所有目标（干扰物模式）")

            # 3. 如果已经选定主角，检查是否跟丢
            if self.has_main and self.main_track_id is not None:
                found = any(t_id == self.main_track_id for (_, _, _, _, _, _, t_id) in detections)
                if not found:
                    self.lost_frames += 1
                    if self.lost_frames > 30:
                        print(f"[INFO] 主角丢失超过 30 帧，重新选定")
                        # 重置状态，下一帧重新选择
                        self.is_first_frame = True
                        self.main_track_id = None
                        self.lost_frames = 0
                        # 注意：has_main 暂不置 False，等重新选定后再置 True
                else:
                    self.lost_frames = 0

            # 4. 构建当前帧的返回列表（过滤主角）
            frame_dets = []
            for (x1, y1, x2, y2, cls, conf, track_id) in detections:
                # 如果已选定主角且当前框是主角，则跳过
                if self.has_main and self.main_track_id is not None and track_id == self.main_track_id:
                    continue
                # 归一化坐标和尺寸
                x_rel = x1 / w
                y_rel = y1 / h
                w_rel = (x2 - x1) / w
                h_rel = (y2 - y1) / h
                frame_dets.append([x_rel, y_rel, w_rel, h_rel, cls, conf, track_id])

            all_frames_results.append(frame_dets)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        print(f"处理完成，共 {len(all_frames_results)} 帧")
        return all_frames_results


# ===================== 主程序 =====================
if __name__ == "__main__":
    # ----- 创建跟踪器 -----
    tracker = YOLOv8nTracker()

    # ----- 批量处理视频 -----
    root_dir = r"/mnt/8T/zwl/HUAWEI/test_data/"
    # 示例：处理编号为15的视频（可修改筛选条件）
    paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
             if name[:2].isdigit() and int(name[:2]) == 00]
    paths = sorted(paths, key=lambda s: int(os.path.basename(s)[:2]))
    print(f"待处理视频: {paths}")

    for video_path in paths:
        print(f"\n>>> 处理视频: {video_path}")
        results = tracker.process_video(video_path)
    