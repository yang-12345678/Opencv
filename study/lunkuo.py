import cv2 as cv
import numpy as np


def video_demo():
    """视频的加载与保存"""
    capture = cv.VideoCapture(0)
    while True:
        # ret是布尔值，当视屏读取到最后一帧或者是读取错误时返回False
        # frame是读取的视频每一帧的图像，是个三维矩阵。
        ret, frame = capture.read()
        # 倒转（变回正常）摄像头
        frame = cv.flip(frame, 1)
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        r2 = cv.Canny(frame, 32, 128)
        cv.imshow("Video", r2)
        # 延迟播放下一帧的间隔时间（ms），在特定时间内按下，返回该键的ASCII值，否则返回-1
        c = cv.waitKey(50)
        if c == 27:  # esc
            break


video_demo()