import cv2 as cv
import numpy as np

open_cv = cv.imread("opencv.png")
# 指定颜色空间
hsv = cv.cvtColor(open_cv, cv.COLOR_BGR2HSV)
# 蓝色
min_blue = np.array([110, 50, 50])
max_blue = np.array([130, 255, 255])
# 确定蓝色的区域
mask = cv.inRange(hsv, min_blue, max_blue)
# 通过掩码控制的按位与运算，锁定蓝色的区域
blue = cv.bitwise_and(open_cv, open_cv, mask=mask)
cv.imshow("blue", blue)

# 绿色
min_grenn = np.array([50, 50, 50])
max_green = np.array([70, 255, 255])

mask = cv.inRange(hsv, min_grenn, max_green)
green = cv.bitwise_and(open_cv, open_cv, mask=mask)
cv.imshow("green", green)

# 红色
min_red = np.array([0, 50, 50])
max_red = np.array([30, 255, 255])
mask = cv.inRange(hsv, min_red, max_red)
red = cv.bitwise_and(open_cv, open_cv, mask=mask)
cv.imshow("red", red)
cv.waitKey(0)
cv.destroyAllWindows()
