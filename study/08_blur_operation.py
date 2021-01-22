import cv2 as cv
import numpy as np

"""
   图像噪声是指存在于图像数据中的不必要的或多余的干扰信息。
   噪声的存在严重影响了遥感图像的质量，因此在图像增强处理和分类处理之前，必须予以纠正
"""


def blur_demo(image):
    """均值模糊"""
    # 用来去噪声 （随机噪声）
    dst = cv.blur(image, (1, 7))  # (1, 5) 一行5列的卷积核
    cv.imshow("blur_demo", dst)


def median_blur_demo(image):
    """中值模糊"""
    # 平滑椒盐噪声
    dst = cv.medianBlur(image, 7)
    cv.imshow("median_blur_demo", dst)


def custom_blur_demo(image):
    """自定义模糊"""
    kernel = np.ones([5, 5], np.float32) / 25  # 自定义的卷积核
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")
src1 = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\253cdbfd8f5e026b.jpg")
# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)
cv.imshow("input image1", src1)

blur_demo(src1)
# median_blur_demo(src1)
# custom_blur_demo(src1)

cv.waitKey(0)
cv.destroyAllWindows()
