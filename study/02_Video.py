import cv2 as cv
import numpy as np


def video_demo():
    """视频的加载与保存"""
    capture = cv.VideoCapture(0)
    while True:
        # ret是布尔值，当视屏读取到最后一帧或者是读取错误时返回False
        # frame是读取的视频每一帧的图像，是个三维矩阵。
        ret, frame = capture.read()
        # 倒转（变回正常）摄像头
        frame = cv.flip(frame, 1)

        cv.imshow("Video", frame)
        # 延迟播放下一帧的间隔时间（ms），在特定时间内按下，返回该键的ASCII值，否则返回-1
        c = cv.waitKey(50)
        if c == 27:  # esc
            break


def get_image_info(image):
    """图像参数"""
    print(type(image))  # 类型
    print(image.shape)  # 形状 （长，宽，通道数目）
    print(image.size)  # 像素数
    print(image.dtype)  # 像素值类型  一般都是uint8
    data = np.array(image)
    print(data)


print("-----------hello world----------")
# 图像读取
src = cv.imread("C:\\Users\\Administrator\\Desktop\\vision\\u=2882008312,3095180572&fm=26&gp=0.jpg")
# 创建窗口
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.WINDOW_NORMAL 可手动调节大小
cv.imshow("input image", src)

video_demo()
get_image_info(src)

cv.waitKey()

cv.destroyAllWindows()
