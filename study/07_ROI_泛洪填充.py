import cv2 as cv
import numpy as np

'''ROI: Region of Interest'''


def fill_color_demo(image):
    """指定颜色填充"""

    # 从一个点开始附近像素点，填充成新的颜色，直到封闭区域内的所有像素点都被填充新颜色为止
    copy_image = image.copy()

    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    # 遮罩层的（h+2， W+2）， np.uint8都是固定的
    cv.floodFill(copy_image, mask, (30, 30), (0, 0, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    # floodFill(1.操作的图像, 2.掩模, 3.起始像素值，4.填充的颜色, 5.填充颜色的低值， 6.填充颜色的高值, 7.填充的方法)
    # 100：(30, 30) 起始尺寸的像素值减100范围内都填充
    # 50: (30, 30) 起始尺寸的像素值加50范围内都填充
    # 最后一个参数flag：填充的方法
    cv.imshow("fill_color_demo", copy_image)


def fill_binary(image):
    """指定位置填充"""
    image = np.zeros([400, 400, 3], np.uint8)  # 创建了一个黑色的图像
    image[100:300, 100:300, :] = 255

    cv.imshow("fill_binary", image)

    mask = np.ones([402, 402], np.uint8)
    mask[101:301, 101:301] = 0  # 填充只会穿过mask的零值区域
    cv.floodFill(image, mask, (200, 200), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill_binary", image)


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)

# fill_color_demo(src)
fill_binary(src)

# face = src[40:300, 100:400]
#
# gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
# backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# src[40:300, 100:400] = backface
# cv.imshow("face", src)

cv.waitKey(0)

cv.destroyAllWindows()
