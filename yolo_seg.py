#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-seg + ByteTrack 动态干扰物分割（主体过滤版）
- 使用实例分割模型，输出每个干扰物的二值掩码
- 首帧自动选定主角（画面中心最近的目标类别在 MAIN_CLS 中）
- 后续帧过滤主角，只返回干扰物的掩码
- 若视频中无主体，则返回所有目标的掩码
- 返回值：list of list，每帧包含多个二值掩码 (numpy bool array, 尺寸与原始帧相同)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os

class DetConfig:
    MODEL_PATH = r'./yolov8n-seg.engine'
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

class YOLOv8SegTracker:
    def __init__(self, model_path=DetConfig.MODEL_PATH, device=DetConfig.DEVICE):
        self.model = YOLO(model_path)
        self.device = device
        try:
            self.model.to(device)
        except:
            pass
        # 主体跟踪状态
        self.main_track_id = None
        self.is_first_frame = True
        self.lost_frames = 0
        self.has_main = False
        print(f"[INFO] YOLO 分割模型加载完成，设备: {device}")

    def _run_yolo_track(self, frame):
        results = self.model.track(frame,
                                conf=DetConfig.CONF_THRESHOLD,
                                iou=DetConfig.IOU_THRESHOLD,
                                half=True,
                                persist=True,
                                tracker=DetConfig.TRACKER_CONFIG,
                                verbose=False)

        # 如果没有检测到任何目标，直接返回空
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        # 检查跟踪 ID 是否可用
        if boxes.id is None:
            return []

        # 提取检测框、类别、置信度、track_id
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        # 提取掩码
        masks = results[0].masks
        if masks is None:
            print("[WARN] 模型未返回掩码，请使用分割模型。")
            return []

        # masks.data 形状为 (N, H_mask, W_mask)
        mask_data = masks.data.cpu().numpy()   # dtype float32

        # 获取原始帧尺寸
        h_orig, w_orig = frame.shape[:2]

        # 如果掩码尺寸与原始帧不一致，则进行缩放
        if mask_data.shape[1] != h_orig or mask_data.shape[2] != w_orig:
            resized_masks = []
            for i in range(mask_data.shape[0]):
                # 保持浮点精度进行缩放，再二值化，保留更多边缘细节
                mask_float = mask_data[i].astype(np.float32)
                # cv2.resize 参数为 (width, height)
                mask_resized = cv2.resize(mask_float, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                # 阈值 0.5 转为二值 bool 数组
                binary_mask = mask_resized > 0.5
                resized_masks.append(binary_mask)
            binary_masks = np.stack(resized_masks, axis=0)   # shape (N, h_orig, w_orig)
        else:
            binary_masks = mask_data > 0.5

        # 构建检测结果列表，此时每个 mask 已经是原始尺寸的 bool 数组
        detections = []
        for i in range(len(xyxy)):
            cls = cls_ids[i]
            if cls not in DetConfig.TARGET_CLS:
                continue
            x1, y1, x2, y2 = xyxy[i].astype(int)
            conf = float(confs[i])
            track_id = track_ids[i]
            mask = binary_masks[i]   # 二维 bool 数组，尺寸 (h_orig, w_orig)
            detections.append((x1, y1, x2, y2, cls, conf, track_id, mask))

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

        for (x1, y1, x2, y2, cls, conf, track_id, mask) in detections:
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
        处理视频，返回每帧的掩码列表。
        返回值: list of list, 外层按帧顺序，内层为该帧所有干扰物的二值掩码 (bool numpy array)。
        掩码尺寸与原始帧相同。
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

        all_frames_masks = []   # 存储每帧的掩码列表
        pbar = tqdm(total=total_frames, desc="处理帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 1. 执行检测 + 跟踪（返回包含掩码的完整信息）
            detections = self._run_yolo_track(frame)

            # 2. 首帧或主角丢失时重新选定主角
            if self.is_first_frame and detections:
                main_id = self.select_main_character(detections, frame.shape)
                if main_id is not None:
                    self.main_track_id = main_id
                    self.has_main = True
                    self.is_first_frame = False
                    self.lost_frames = 0
                    print(f"[INFO] 主角已选定，ID: {self.main_track_id}")
                else:
                    # 首帧无主体，进入干扰物模式（所有目标均视为干扰物）
                    self.has_main = False
                    self.is_first_frame = False
                    print("[INFO] 首帧未检测到主体，将返回所有目标掩码（干扰物模式）")

            # 3. 主角跟丢检测
            if self.has_main and self.main_track_id is not None:
                found = any(t_id == self.main_track_id for (_, _, _, _, _, _, t_id, _) in detections)
                if not found:
                    self.lost_frames += 1
                    if self.lost_frames > 30:
                        print(f"[INFO] 主角丢失超过 30 帧，重新选定")
                        self.is_first_frame = True
                        self.main_track_id = None
                        self.lost_frames = 0
                else:
                    self.lost_frames = 0

            # 4. 构建当前帧的掩码列表（过滤主角）
            frame_masks = []
            for (x1, y1, x2, y2, cls, conf, track_id, mask) in detections:
                # 如果已选定主角且当前目标为主角，则跳过
                if self.has_main and self.main_track_id is not None and track_id == self.main_track_id:
                    continue
                # 保留干扰物的掩码（二值 bool 数组）
                frame_masks.append(mask)

            all_frames_masks.append(frame_masks)

            pbar.update(1)

        cap.release()
        pbar.close()
        print(f"处理完成，共 {len(all_frames_masks)} 帧")
        return all_frames_masks


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 创建分割跟踪器
    tracker = YOLOv8SegTracker()

    # 批量处理视频
    root_dir = r"/mnt/8T/zwl/HUAWEI/test_data/"
    # 示例：处理编号为00的视频（请根据实际修改筛选条件）
    paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
             if name[:2].isdigit() and int(name[:2]) == 14]
    paths = sorted(paths, key=lambda s: int(os.path.basename(s)[:2]))
    print(f"待处理视频: {paths}")

    for video_path in paths:
        print(f"\n>>> 处理视频: {video_path}")
        masks_per_frame = tracker.process_video(video_path)
        # 此处可以对 masks_per_frame 进行后续处理，例如保存为 .npz 或进行可视化
        # 示例：打印第一帧掩码数量和尺寸
        if masks_per_frame:
            first_frame_masks = masks_per_frame[0]
            print(f"第一帧共有 {len(first_frame_masks)} 个干扰物掩码")
            if first_frame_masks:
                print(f"掩码形状示例: {first_frame_masks[0].shape}")