# -*- coding=GBK -*-
import cv2 as cv


# 人脸检测
def face_image():
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("../project/cascades/haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 1)  # 第二个参数是移动距离，第三个参数是识别度，越大识别读越高
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 后两个参数，一个是颜色，一个是边框宽度
    cv.imshow("output", src)


src = cv.imread("face.jfif")
cv.imshow("input", src)
face_image()
cv.waitKey(0)
cv.destroyAllWindows()

# -*- coding=GBK -*-
# import cv2 as cv
#
#
# # 摄像头人脸检测
# def face_image(src):
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     face_detector = cv.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
#     faces = face_detector.detectMultiScale(gray, 1.02, 5)  # 第二个参数是移动距离，第三个参数是识别度，越大识别读越高
#     for x, y, w, h in faces:
#         cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 后两个参数，一个是颜色，一个是边框宽度
#     cv.imshow("结果", src)
#
#
# capture = cv.VideoCapture(0)
# while True:
#     ret, frame = capture.read()
#     frame = cv.flip(frame, 1)
#     face_image(frame)
#     if cv.waitKey(10) & 0xFF == ord('q'):  # 键盘输入q退出窗口，不按q点击关闭会一直关不掉 也可以设置成其他键。
#         break
# face_image()
# cv.waitKey(0)
# cv.destroyAllWindows()