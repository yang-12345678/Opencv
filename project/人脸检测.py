import cv2 as cv


def detect():
    # 创建人脸检测对象
    face_cascade = cv.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')
    # 创建眼睛检测对象
    eye_cascade = cv.CascadeClassifier('../cascades/haarcascade_eye.xml')
    # 连接内置摄像头
    video = cv.VideoCapture(0)
    while True:
        # 读取当前帧
        ret, frame = video.read()
        # 调转摄像头
        frame = cv.flip(frame, 1)
        # 转换为灰度图像
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 检测人脸 返回列表 每个元素都是(x, y, w, h)表示矩形的左上角和宽高
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 在人脸上画矩形
        for (x, y, w, h) in faces:
            # 画矩形 在frame图片上画， 传入左上角和右下角坐标 矩形颜色 和线条宽度
            image = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 在脸上检测眼睛   (40, 40)是设置最小尺寸，再小的部分会不检测
            eyes = eye_cascade.detectMultiScale(image, 1.03, 5, 0, (40, 40))
            # 在眼睛上画矩形
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv.imshow("camera", frame)
        # 按esc退出
        if cv.waitKey(50) == 27:
            break
    video.release()
    cv.destroyAllWindows()


detect()
