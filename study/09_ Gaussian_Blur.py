import cv2 as cv
import numpy as np

"""
   把卷积核换成高斯核（简单来说，方框不变，将原来每个方框的值是相等的，
   现在里面的值是符合高斯分布的，方框中心的值最大，其余方框根据距离中心元素的距离递减，
   构成一个高斯小山包。原来的求平均数现在变成求 加权平均数，全就是方框里的值）
"""


def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):
    """加噪声"""
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow("noise_image", image)


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\253cdbfd8f5e026b.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)

# 高斯模糊对高斯噪声有抑制作用
dst = cv.GaussianBlur(src, (0, 0), 15)
cv.imshow("Gaussian_blur", dst)

gaussian_noise(src)


cv.waitKey(0)

cv.destroyAllWindows()