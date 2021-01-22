import cv2 as cv
import numpy as np

'''像素运算时要求图片shape相同'''


def add_demo(m1, m2):
    """像素相加"""
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)


def subtract_demo(m1, m2):
    """像素相减"""
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


def multiply_demo(m1, m2):
    """像素相乘"""
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)


def divide_demo(m1, m2):
    """像素相除"""
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)


def logic_demo(m1, m2):
    """逻辑运算"""
    dst1 = cv.bitwise_and(m1, m1)  # 与
    dst2 = cv.bitwise_or(m1, m2)  # 或
    # not用于像素翻转，对单个图像操作
    cv.imshow("and_demo", dst1)
    cv.imshow("or_demo", dst2)


def contrast_brightness(image, c, b):
    """对比度增强"""
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    # cv.imshow("blank", blank)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("con_bri_demo", dst)


def others(m1, m2):
    """均值和方差"""
    M1 = cv.mean(m1)
    M2 = cv.mean(m2)
    # cv.meanStdDev()  同时返回均值和方差
    print(M1)
    print(M2)


print("-----------hello world----------")

# 读取图片
src1 = cv.imread("C:\\Users\\Administrator\\Desktop/vision\\LinuxLogo.jpg")
src2 = cv.imread("C:\\Users\\Administrator\\Desktop/vision\\WindowsLogo.jpg")

print(src1.shape)
print(src2.shape)

# 创建窗口
cv.namedWindow("image1", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("image1", src1)
cv.imshow("image2", src2)

# add_demo(src1, src2)
# subtract_demo(src1, src2)
# multiply_demo(src1, src2)
# divide_demo(src1, src2)
# logic_demo(src1, src2)
contrast_brightness(src2, 1.2, 10)

others(src1, src2)

cv.waitKey(0)

cv.destroyAllWindows()
