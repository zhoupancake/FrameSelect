#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能视频处理器（多线程版）：
- 首帧检测主体，若主体为人：
    * 主线程：执行干扰物掩码提取（每帧输出掩码列表）
    * 子线程：执行姿态估计，寻找该主角的最佳正脸帧
- 若主体非人：仅执行掩码提取
"""

import os
import cv2
import numpy as np
import threading
from tqdm import tqdm
from ultralytics import YOLO

# ======================== 配置 ========================
class DetConfig:
    # ----- 分割 + 跟踪配置 -----
    SEG_MODEL_PATH = r'./yolov8n-seg.engine'   # 或 .pt
    DEVICE = 'cuda'
    TARGET_CLS = [0, 2, 5, 7]          # 需要检测的目标
    MAIN_CLS = [0, 15, 16]             # 可能成为主角的类别（0:人, 15:猫, 16:狗）
    CLASS_NAMES = {
        0: 'person', 2: 'car', 5: 'bus', 7: 'truck',
        15: 'cat', 16: 'dog'
    }
    YOLO_IMGSZ = 640
    CONF_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.65
    TRACKER_CONFIG = "bytetrack.yaml"
    
    # ----- 姿态估计配置 -----
    POSE_MODEL_PATH = 'yolov8n-pose.engine'   # 可改为 .pt
    POSE_CONF_THRESH = 0.5
    POSE_SYMMETRY_EPS = 1e-6

# ======================== 干扰物掩码提取器 ========================
class YOLOv8SegTracker:
    def __init__(self, model_path=DetConfig.SEG_MODEL_PATH, device=DetConfig.DEVICE):
        self.model = YOLO(model_path)
        self.device = device
        # 移除 self.model.to(device)  # 对于 .engine 无效
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
                                verbose=False,
                                device=self.device)   # 传入 device

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        if boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        masks = results[0].masks
        if masks is None:
            print("[WARN] 模型未返回掩码，请使用分割模型。")
            return []
        mask_data = masks.data.cpu().numpy()
        h_orig, w_orig = frame.shape[:2]
        if mask_data.shape[1] != h_orig or mask_data.shape[2] != w_orig:
            resized_masks = []
            for i in range(mask_data.shape[0]):
                mask_float = mask_data[i].astype(np.float32)
                mask_resized = cv2.resize(mask_float, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                binary_mask = mask_resized > 0.5
                resized_masks.append(binary_mask)
            binary_masks = np.stack(resized_masks, axis=0)
        else:
            binary_masks = mask_data > 0.5

        detections = []
        for i in range(len(xyxy)):
            cls = cls_ids[i]
            if cls not in DetConfig.TARGET_CLS:
                continue
            x1, y1, x2, y2 = xyxy[i].astype(int)
            conf = float(confs[i])
            track_id = track_ids[i]
            mask = binary_masks[i]
            detections.append((x1, y1, x2, y2, cls, conf, track_id, mask))
        return detections

    def select_main_character(self, detections, frame_shape):
        """返回 (主角track_id, 主角类别, 主角框)"""
        h, w = frame_shape[:2]
        center_x, center_y = w / 2, h / 2
        best_id = None
        best_cls = None
        best_box = None
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
                best_cls = cls
                best_box = (x1, y1, x2, y2)
        return best_id, best_cls, best_box

    def _process_video_mask_mode(self, video_path):
        """纯掩码提取模式（主线程调用，会重置内部状态）"""
        self.main_track_id = None
        self.is_first_frame = True
        self.lost_frames = 0
        self.has_main = False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_frames_masks = []
        pbar = tqdm(total=total_frames, desc="提取掩码")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self._run_yolo_track(frame)

            if self.is_first_frame and detections:
                main_id, main_cls, main_box = self.select_main_character(detections, frame.shape)
                if main_id is not None:
                    self.main_track_id = main_id
                    self.has_main = True
                    self.is_first_frame = False
                    self.lost_frames = 0
                    print(f"[INFO] 主角已选定，ID: {self.main_track_id}, 类别: {DetConfig.CLASS_NAMES.get(main_cls, 'unknown')}")
                else:
                    self.has_main = False
                    self.is_first_frame = False
                    print("[INFO] 首帧无主体，返回所有目标掩码（干扰物模式）")

            if self.has_main and self.main_track_id is not None:
                found = any(t_id == self.main_track_id for (_, _, _, _, _, _, t_id, _) in detections)
                if not found:
                    self.lost_frames += 1
                    if self.lost_frames > 30:
                        print("[INFO] 主角丢失超过30帧，重新选定")
                        self.is_first_frame = True
                        self.main_track_id = None
                        self.lost_frames = 0
                else:
                    self.lost_frames = 0

            frame_masks = []
            for (x1, y1, x2, y2, cls, conf, track_id, mask) in detections:
                if self.has_main and self.main_track_id is not None and track_id == self.main_track_id:
                    continue
                frame_masks.append(mask)

            all_frames_masks.append(frame_masks)
            pbar.update(1)

        cap.release()
        pbar.close()
        return all_frames_masks


# ======================== 正脸提取器 ========================
class YOLOFrontalExtractor:
    def __init__(self, model_path=DetConfig.POSE_MODEL_PATH, device=DetConfig.DEVICE):
        self.model = YOLO(model_path)
        self.device = device
        # 移除 self.model.to(device)  # 兼容 .engine
        print(f"[INFO] YOLO 姿态模型加载完成，设备: {device}")

    @staticmethod
    def compute_iou(box1, box2):
        """计算两个框的 IoU (box: x1,y1,x2,y2)"""
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def process_frame(self, frame, target_box=None):
        """
        处理单帧，若指定 target_box 则优先匹配该框内的主角
        返回: (symmetry_score, main_box, main_kps, status)
        """
        results = self.model(frame, verbose=False, device=self.device)   # 传入 device
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None, None, None, None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        keypoints = results[0].keypoints.data.cpu().numpy()  # [N, 17, 3]

        # ---- 选择目标人物 ----
        if target_box is not None:
            ious = [self.compute_iou(target_box, box) for box in boxes]
            best_idx = np.argmax(ious)
            if ious[best_idx] < 0.1:
                # 回退：中心距离
                t_cx = (target_box[0] + target_box[2]) / 2
                t_cy = (target_box[1] + target_box[3]) / 2
                dists = [((box[0]+box[2])/2 - t_cx)**2 + ((box[1]+box[3])/2 - t_cy)**2 for box in boxes]
                best_idx = np.argmin(dists)
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = np.argmax(areas)

        main_box = boxes[best_idx]
        main_kps = keypoints[best_idx]

        # ---- 提取五官 ----
        nose = main_kps[0]
        l_eye = main_kps[1]
        r_eye = main_kps[2]
        l_ear = main_kps[3]
        r_ear = main_kps[4]

        # ---- 置信度过滤 ----
        face_confs = [nose[2], l_eye[2], r_eye[2], l_ear[2], r_ear[2]]
        if min(face_confs) < DetConfig.POSE_CONF_THRESH:
            return float('inf'), main_box, main_kps, "Occluded/Profile"

        # ---- 对称性得分 ----
        d_left = np.linalg.norm(nose[:2] - l_eye[:2])
        d_right = np.linalg.norm(nose[:2] - r_eye[:2])
        symmetry_score = abs(d_left - d_right) / (d_left + d_right + DetConfig.POSE_SYMMETRY_EPS)

        return symmetry_score, main_box, main_kps, "Frontal/Valid"

    def process_video(self, video_path, target_box=None, debug_dir=None):
        """
        在整个视频中寻找最佳正脸帧，若指定 target_box 则跟踪该目标
        """
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[Debug] 姿态调试图保存至: {debug_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频 {video_path} 进行正脸提取")
            return -1, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        best_score = float('inf')
        best_frame_idx = -1
        best_frame = None

        for i in tqdm(range(total_frames), desc="搜索正脸", position=1, leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            score, box, kps, status = self.process_frame(frame, target_box=target_box)

            if score is not None and score < best_score:
                best_score = score
                best_frame_idx = i
                best_frame = frame.copy()

            # ---- 绘制调试图（可选） ----
            if debug_dir and box is not None:
                debug_img = frame.copy()
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                pts = {
                    "Nose": (kps[0], (0, 0, 255)),
                    "L-Eye": (kps[1], (0, 255, 255)),
                    "R-Eye": (kps[2], (0, 255, 255)),
                    "L-Ear": (kps[3], (255, 255, 0)),
                    "R-Ear": (kps[4], (255, 255, 0))
                }
                for name, (kp, color) in pts.items():
                    x, y, conf = kp
                    if conf > 0.3:
                        cv2.circle(debug_img, (int(x), int(y)), 5, color, -1)
                        cv2.putText(debug_img, name, (int(x)+5, int(y)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                color_text = (0, 255, 0) if status == "Frontal/Valid" else (0, 0, 255)
                cv2.putText(debug_img, f"Frame: {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(debug_img, f"Status: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
                cv2.putText(debug_img, f"Score: {score:.4f}" if score is not None else "Score: None",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

                cv2.imwrite(os.path.join(debug_dir, f"pose_debug_{i:04d}.jpg"), debug_img)

        cap.release()
        if best_frame is None:
            print("[Warning] 未检测到任何有效正脸！")
        else:
            print(f"[Done] 最佳正脸帧: 第 {best_frame_idx} 帧 (偏差分: {best_score:.4f})")
        return best_frame_idx, best_frame


# ======================== 智能融合处理器（多线程版） ========================
class SmartVideoProcessor:
    def __init__(self, seg_model_path=DetConfig.SEG_MODEL_PATH,
                 pose_model_path=DetConfig.POSE_MODEL_PATH,
                 device=DetConfig.DEVICE):
        self.device = device
        self.seg_tracker = YOLOv8SegTracker(seg_model_path, device)
        self.pose_extractor = None
        self.pose_model_path = pose_model_path

    def _get_pose_extractor(self):
        if self.pose_extractor is None:
            self.pose_extractor = YOLOFrontalExtractor(self.pose_model_path, self.device)
        return self.pose_extractor

    def process_video(self, video_path, debug_dir=None):
        """
        智能处理视频：
        - 首帧判定主体类别
        - 若为人 → 主线程提取掩码，子线程提取正脸，并行执行
        - 若非人 → 仅提取掩码
        返回字典：
            - 成功时包含 "mode", "masks" 等
            - 失败时包含 "error" 键
        """
        # ---- 第1步：读取首帧，做决策 ----
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {video_path}")
            return {"error": f"Failed to open video: {video_path}"}
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print(f"[ERROR] 无法读取首帧: {video_path}")
            return {"error": f"Failed to read first frame from: {video_path}"}

        detections = self.seg_tracker._run_yolo_track(first_frame)
        if not detections:
            print("[INFO] 首帧未检测到任何目标，进入干扰物模式（返回所有掩码）")
            masks = self.seg_tracker._process_video_mask_mode(video_path)
            return {"mode": "non_person", "masks": masks}

        main_id, main_cls, main_box = self.seg_tracker.select_main_character(detections, first_frame.shape)
        if main_id is None:
            print("[INFO] 首帧无符合 MAIN_CLS 的目标，进入干扰物模式")
            masks = self.seg_tracker._process_video_mask_mode(video_path)
            return {"mode": "non_person", "masks": masks}

        cls_name = DetConfig.CLASS_NAMES.get(main_cls, "unknown")
        print(f"[决策] 主角 ID: {main_id}, 类别: {cls_name}")

        # ---- 第2步：分支处理 ----
        if main_cls == 0:  # 人是类别 0
            print("[分支] 主角为人类 → 主线程执行掩码提取，子线程执行正脸检测")

            # 预先加载姿态模型（确保子线程可用）
            pose_ext = self._get_pose_extractor()

            # 准备子线程任务
            face_result = {"idx": -1, "frame": None}
            def face_task():
                idx, frame = pose_ext.process_video(
                    video_path,
                    target_box=main_box,
                    debug_dir=os.path.join(debug_dir, "pose_debug") if debug_dir else None
                )
                face_result["idx"] = idx
                face_result["frame"] = frame

            face_thread = threading.Thread(target=face_task)
            face_thread.start()

            # 主线程执行掩码提取
            masks = self.seg_tracker._process_video_mask_mode(video_path)

            # 等待子线程完成
            face_thread.join()

            # 返回综合结果
            return {
                "mode": "person",
                "masks": masks,
                "best_face_idx": face_result["idx"],
                "best_face_frame": face_result["frame"]
            }

        else:
            print(f"[分支] 主角为非人类 ({cls_name}) → 仅执行干扰物掩码提取")
            masks = self.seg_tracker._process_video_mask_mode(video_path)
            return {"mode": "non_person", "masks": masks}


# ======================== 主程序测试 ========================
if __name__ == "__main__":
    processor = SmartVideoProcessor(
        seg_model_path="./yolov8n-seg.engine",
        pose_model_path="./yolov8n-pose.engine",   # 可根据实际使用 .pt 或 .engine
        device="cuda"
    )

    video_path = None
    result = processor.process_video(video_path, debug_dir="./debug_output")

    # 检查错误
    if result is None or "error" in result:
        print(f"处理失败: {result.get('error', 'Unknown error') if result else 'result is None'}")
        exit(1)

    if result["mode"] == "person":
        print(f"掩码提取完成，共 {len(result['masks'])} 帧")
        print(f"最佳正脸帧索引: {result['best_face_idx']}")
        if result['best_face_frame'] is not None:
            cv2.imwrite("best_face.jpg", result['best_face_frame'])
    else:
        print(f"掩码提取完成（非人模式），共 {len(result['masks'])} 帧")
        if result['masks']:
            print(f"第一帧有 {len(result['masks'][0])} 个干扰物掩码")