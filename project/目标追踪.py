# -*- coding: utf-8 -*-
# Date: 2021/06/14

import cv2

list = [0, 0, 0, 0, 0, 0, 0, 0]

tracker = cv2.TrackerCSRT_create()

bbox = (287, 23, 86, 320)

capture = cv2.VideoCapture(0)
x = 0
while True:
    # ret是布尔值，当视屏读取到最后一帧或者是读取错误时返回False
    # frame是读取的视频每一帧的图像，是个三维矩阵。

    ret, frame = capture.read()
    # 倒转（变回正常）摄像头
    frame = cv2.flip(frame, 1)


    if x == 0:
        bbox = cv2.selectROI(frame, False)
        ok = tracker.init(frame, bbox)
        x += 1

    ok, bbox = tracker.update(frame)

    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("one", frame)
    c = cv2.waitKey(50)
    if c == 27:  # esc
        break
