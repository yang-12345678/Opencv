import cv2 as cv


# 图像梯度：索贝尔算子
def sobel_image(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)  # x方向导数
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)  # y方向导数
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("X", gradx)  # 颜色变化在水平分层
    cv.imshow("Y", grady)  # 颜色变化在垂直分层
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("x&y", gradxy)


src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=226249496,1262278109&fm=26&gp=0.jpg")
cv.imshow("input", src)
sobel_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
