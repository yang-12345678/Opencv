import cv2 as cv
import numpy as np


def color_tracking_demo():
    """用inRange（）做颜色追踪"""
    capture = cv.VideoCapture("C:\\Users\\yang\\Desktop\\文件\\111_1.mp4")
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        if ret is False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 要追踪的颜色下界（hsv）
        lower_hsv = np.array([18, 50, 50])
        # 要追踪的颜色上界
        upper_hsv = np.array([35, 255, 255])

        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("video", frame)
        cv.imshow("mask", dst)

        c = cv.waitKey(40)
        if c == 27:
            break
    cv.destroyAllWindows()


# print("-----------hello world----------")

# 读取图片
# src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")

# 创建窗口
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
# cv.imshow("input image", src)

color_tracking_demo()

cv.waitKey()

cv.destroyAllWindows()
