import cv2 as cv
# import numpy as np


print("-----------hello world----------")

# 读取图片
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")

# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 加载图像
cv.imshow("input image", src)

cv.waitKey(0)

cv.destroyAllWindows()
