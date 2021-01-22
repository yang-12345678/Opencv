# -*- coding=utf-8 -*-
import cv2 as cv


# 边缘检测述算法
def edge_image(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    cv.imshow("边缘", edge_output)
    dst = cv.bitwise_and(image, image, mask=edge_output)
    cv.imshow("color边缘", dst)


src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")
cv.imshow("input", src)
edge_image(src)
cv.waitKey(0)
cv.destroyAllWindows()