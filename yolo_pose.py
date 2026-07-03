import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

class YOLOFrontalExtractor:
    def __init__(self, model_path="yolov8n-pose.pt", device="cuda"):
        self.model = YOLO(model_path)
        self.model.to(device)

    def process_frame(self, frame):
        """
        处理单帧：框出最大的人，提取五官，计算正脸得分
        """
        # verbose=False 关闭 YOLO 烦人的逐帧打印
        results = self.model(frame, verbose=False)
        
        # 如果画面里连一个人都没有
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None, None, None, None

        # 1. 提取所有人的检测框和关键点
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4]
        keypoints = results[0].keypoints.data.cpu().numpy()  # [N, 17, 3] (x, y, conf)

        # 2. 核心抗干扰逻辑：只选画面中 bounding box 面积最大的那个人（主角）
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = np.argmax(areas)
        
        main_box = boxes[best_idx]
        main_kps = keypoints[best_idx]

        # 3. 提取五官关键点 (COCO 格式索引)
        nose = main_kps[0]
        l_eye = main_kps[1]
        r_eye = main_kps[2]
        l_ear = main_kps[3]
        r_ear = main_kps[4]

        # 4. 遮挡与侧脸的绝对防线：利用置信度过滤
        # 如果这 5 个点里有任何一个点的置信度低于 0.5，说明脸转过去了，或者被挡住了
        face_confidences = [nose[2], l_eye[2], r_eye[2], l_ear[2], r_ear[2]]
        if min(face_confidences) < 0.5:
            return float('inf'), main_box, main_kps, "Occluded/Profile"

        # 5. 几何打分逻辑 (2D 欧式距离比例)
        # 计算鼻子到左右眼的距离
        d_left = np.linalg.norm(nose[:2] - l_eye[:2])
        d_right = np.linalg.norm(nose[:2] - r_eye[:2])
        
        # 使用比例差值而不是绝对像素差，这样人离镜头远近都不会影响打分
        # score 越接近 0 越对称
        symmetry_score = abs(d_left - d_right) / (d_left + d_right + 1e-6)

        return symmetry_score, main_box, main_kps, "Frontal/Valid"

    def process_video(self, video_path, debug_dir=None):
        """
        处理整个视频，输出每一帧的调试结果，并返回最好的一帧
        """
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[Debug] YOLO 检测过程图将保存至: {debug_dir}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        best_score = float('inf')
        best_frame_idx = -1
        best_frame = None

        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret: break
            
            score, box, kps, status = self.process_frame(frame)
            
            if score is not None:
                # 更新最佳帧
                if score < best_score:
                    best_score = score
                    best_frame_idx = i
                    best_frame = frame.copy()
                
                # ================= 绘制极其直观的调试图片 =================
                if debug_dir:
                    debug_img = frame.copy()
                    
                    # 1. 画出主角的 Bounding Box (绿色)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 2. 画出五官的 5 个点
                    # 颜色定义: 鼻(红), 左眼(黄), 右眼(黄), 左耳(青), 右耳(青)
                    pts = {
                        "Nose": (kps[0], (0, 0, 255)),
                        "L-Eye": (kps[1], (0, 255, 255)),
                        "R-Eye": (kps[2], (0, 255, 255)),
                        "L-Ear": (kps[3], (255, 255, 0)),
                        "R-Ear": (kps[4], (255, 255, 0))
                    }
                    
                    for name, (kp, color) in pts.items():
                        x, y, conf = kp
                        # 只画出置信度 > 0.3 的点
                        if conf > 0.3:
                            cv2.circle(debug_img, (int(x), int(y)), 5, color, -1)
                            cv2.putText(debug_img, name, (int(x)+5, int(y)-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    #3. 屏幕左上角打印该帧核心数据
                    color_text = (0, 255, 0) if status == "Frontal/Valid" else (0, 0, 255)
                    cv2.putText(debug_img, f"Frame: {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(debug_img, f"Status: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
                    cv2.putText(debug_img, f"Score: {score:.4f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
                    
                    cv2.imwrite(os.path.join(debug_dir, f"yolo_debug_{i:04d}.jpg"), debug_img)
                # ==========================================================

        cap.release()
        
        if best_frame is None:
            print("[Warning] 未检测到任何有效人脸！")
        else:
            print(f"[Done] YOLO 筛选出的最佳正脸为第 {best_frame_idx} 帧 (偏差分: {best_score:.4f})")
            
        return best_frame_idx, best_frame