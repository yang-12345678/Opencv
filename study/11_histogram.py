import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_demo(image):
    # image.ravel()  将多维数组降为一维数组
    # hist(要统计的数据， bin(条形数), 颜色， density是否用密度形式显示， range, bottom(y轴的其实位置），histtype（线条类型） )
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def image_hist_demo(image):
    """绘制三个通道"""
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        '''一、images（输入图像）参数必须用方括号括起来。
           二、计算直方图的通道。
           三、Mask（掩膜），一般用None，表示处理整幅图像。
           四、histSize，表示这个直方图分成多少份（即多少个直方柱）。
           五、range，直方图中各个像素的值，[0.0, 256.0]表示直方图能表示像素值从0.0到256的像素。'''

        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\20200718201538.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)
image_hist_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
