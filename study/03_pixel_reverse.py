import cv2 as cv
import numpy as np


def access_pixels(image):
    """像素反转"""
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width: %s, height: %s, channels: %s" % (width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv

    cv.imshow("pixels_demo", image)


def creat_image():
    """用数组创图像"""
    image = np.zeros([400, 400, 3], np.uint8)
    print(image)
    image[:, :, 0] = np.ones([400, 400]) * 255

    # cv.imshow("new image", image)
    #
    # image = np.zeros([400, 400, 1], np.uint8)
    # image[:, :, 0] = np.ones([400, 400]) * 127
    # cv.imshow("new image", image)

    m1 = np.ones((3, 3), np.float32)
    m1.fill(122.388)
    print(m1)

    m2 = m1.reshape((1, 9))
    print(m2)


def inverse(image):
    """内置API像素翻转"""
    dst = cv.bitwise_not(image)

    cv.imshow("inverse input", dst)


print("-----------hello world----------")
src = cv.imread("000.jpg")  # blue green red
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# t1 = cv.getTickCount()

access_pixels(src)

# t2 = cv.getTickCount()
# time = (t2 - t1) / cv.getTickFrequency()
# print("time: %f ms" % (time * 1000))

cv.waitKey(0)

cv.destroyAllWindows()
