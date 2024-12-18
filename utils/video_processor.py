import cv2
from .blur_utils.py import blur_faces_and_plates

def process_video(input_path, output_path):
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对每帧进行模糊处理
        blurred_frame = blur_faces_and_plates(frame)
        out.write(blurred_frame)

    cap.release()
    out.release()
