import cv2 as cv
import numpy as np


def detect_circles_demo(image):
    """霍夫圆检测"""
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(dst, cv.COLOR_BGRA2GRAY)
    # 输入， 方法, 最小不长， 误差，  边缘提取的低值，  高值
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], [0, 0, 255], 2)
        cv.circle(image, (i[0], i[1]), 2, [255, 0, 0], 2)
    cv.imshow("image", image)

print("-----------hello world----------")

# 读取图片
src = cv.imread("smarties.png")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)

detect_circles_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()