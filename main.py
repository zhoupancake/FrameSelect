import os
from YOLO import YOLOv8nTracker
from config import *

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
    
