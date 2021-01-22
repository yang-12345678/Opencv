import cv2 as cv
import numpy as np


def color_space_demo(image):
    """转换色彩空间"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)

    '''
        H（Hue）：色调，取值范围是 [0,179] ，它用来限制某一个颜色的彩色光谱范围；
        S（Saturation）：饱和度，取值范围是 [0,255] ,它用来限制颜色的深度，值越大颜色越深；
        V（Value）：色值，取值范围是 [0,255] ,它用来限制像素的亮度，值越大像素越亮。
    '''

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb", ycrcb)


print("-----------hello world----------")

# 读取图片
src = cv.imread("yang.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)

# 分离颜色通道
# split 是一个比较耗时的工作，只有真正需要他是才使用，  b = image[:, :, 0]  尽量使用索引
# b, g, r = cv.split(src)
# cv.imshow("blue", b)
# cv.imshow("green", g)
# cv.imshow("red", r)
#
# cv.merge([b, g, r])
#
# src[:, :, 2] = 0

# cv.imshow("change image", src)
color_space_demo(src)
cv.waitKey()

cv.destroyAllWindows()
