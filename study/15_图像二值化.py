import cv2 as cv
import numpy as np

'''图像二值化：图像二值化（ Image Binarization）就是将图像上的像素点的灰度值设置为0或255，
   也就是将整个图像呈现出明显的黑白效果的过程。
   在数字图像处理中，二值图像占有非常重要的地位，
   图像的二值化使图像中数据量大为减少，从而能凸显出目标的轮廓。'''


def threshold_demo(image):
    """全局阈值"""
    # 将图像转换为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    # ret 布尔值，代表是否读到图像
    # binary 目标图像
    # gray 输入图像
    # 0  阈值   255  最大值   二值化类型有5种
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("%s" % ret)
    cv.imshow("binary", binary)


def local_threshold_demo(image):
    """局部阈值"""
    gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    # 输入图像  最大灰度值  模式   邻域大小（用来计算阈值的区域大小）。 c 常量 这个值是一个从平均值中减去的整数，使我们可以微调我们的阈值
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 16)
    cv.imshow("local_demo", binary)


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)

cv.waitKey(0)

cv.destroyAllWindows()
