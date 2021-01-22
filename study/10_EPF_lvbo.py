import cv2 as cv
import numpy as np


def bi_demo(image):
    """双边模糊"""
    # 边缘像素差异大的会保留
    # 颜色差异要取大一点，表示小的差异要模糊掉，去掉噪声
    # 空间差异要取小一点，核的大小就会小，主要差异保留
    dst = cv.bilateralFilter(image, 0, 80, 10)
    # 过滤时周围每个像素领域的直径的  0
    cv.imshow('bi_demo', dst)


def shift_demo(image):
    """均值迁移"""
    # 可能会过度模糊
    # 图像 空间窗的半径  色彩窗的半径
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow('shift_demo', dst)


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\lena.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)
shift_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
