import cv2
import numpy as np

# 视频路径和输出路径（假设在项目根目录下的 videos 文件夹）
video_path = "videos/SampleVideo.mp4"
output_path = "videos/output_counted_status.mp4"

# 目标像素坐标
target_x, target_y = 1535, 886

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Video not found: {video_path}")
    exit()

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"総フレーム数: {total_frames}, FPS: {fps}")

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

# 参数设置
threshold = 200
status = 0
Num_Car = 0
Num_Frame = 0

# 逐帧处理
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixel = frame_gray[target_y, target_x]
    print(f"フレーム {current_frame}, ピクセル値: {pixel}")

    if status == 0:
        if pixel < threshold:
            status = 1
    elif status == 1:
        if pixel < threshold:
            Num_Frame += 1
        else:
            if Num_Frame > 3:
                Num_Car += 1
                print(f"車を通行フレーム {current_frame}, 通過した車: {Num_Car}")
            status = 0
            Num_Frame = 0

    # 在视频帧上添加文字和圆圈
    cv2.putText(frame, f"Cars: {Num_Car}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(frame, (target_x, target_y), 10, (0, 0, 255), -1)

    # 写入处理后的帧
    out.write(frame)

# 释放资源
cap.release()
out.release()
print(f"完了：{Num_Car}台の自動車を検出しました")
