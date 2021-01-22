# coding:utf-8

import cv2 as cv
import os
import numpy as np



def face_collect():
    # 获取XML文件，加载人脸检测器
    face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
    video = cv.VideoCapture(0)
    name = input("请输入采集人的信息(格式：编号+姓名)：")
    os.mkdir("./data/{}".format(name))
    count = 0
    while True:
        # 读取当前帧
        ret, frame = video.read()
        # 调转摄像头
        frame = cv.flip(frame, 1)

        # 转换为灰度图像
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 调用方法detectMultiScale检测人脸
        # 待检测图像（通常为灰度），前后两次扫描中窗口缩放比例，
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(5, 5)
        )

        # 画矩形 在frame图片上画， 传入左上角和右下角坐标 矩形颜色 和线条宽度
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv.resize(gray[y:y + h, x:x + w], (200, 200))
            cv.imwrite("./data/{}/{}.pgm".format(name, str(count)), f)
            count += 1

        cv.imshow("dect", frame)
        if cv.waitKey(50) == 27:
            break
    video.release()
    cv.destroyAllWindows()

def read_images():
    x, y = [], []
    path = "./data/"
    # 列出data文件夹下的所有内容  yang....
    image_file = os.listdir(path)
    # print(type(image_file))    列表类型

    # 得到所有形如./data/yang/  ...  路径  zhao qian sun li
    image_files = [path + i for i in image_file]
    for file in image_files:
        # ./data/yang路径下所有内容，即为图片
        images = os.listdir(file)

        label = file.split("/")[-1][:1]
        for i in images:
            # ./data/yang/{}.pgm
            img = cv.imread(file + "/" + i, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (200, 200))
            # x列表中存每个图片的信息
            # y列表中存每个图片对应的标签
            x.append(np.asarray(img, dtype=np.uint8))
            y.append(int(label))

    y = np.asarray(y, dtype=np.int32)
    return x, y




def face_rec():
    # 获取数据
    x, y = read_images()

    # 生成特征脸识别器的实例模型
    # model = cv.face.EigenFaceRecognizer_create()
    # fisherfaces算法的模型
    # model = cv.face.FisherFaceRecognizer_create()
    # LBPH算法的模型
    model = cv.face.LBPHFaceRecognizer_create()
    """
    Eigenfaces和Fisherfaces 预测时候产生0到20000的评分
        低于4000 5000 的评分都是相当可靠的
    LBPH 产生评分不同，低于60是可靠的 高于80是不可靠的
    """

    # 训练模型
    model.train(np.asarray(x), np.asarray(y))

    # 开摄像头
    camera = cv.VideoCapture(0)
    # 加载检测人脸对象
    face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
    while True:
        # 读取当前帧
        ret, frame = camera.read()
        frame = cv.flip(frame, 1)
        # 当前帧下检测人脸
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(5, 5)
        )
        for (x, y, w, h) in faces:
            # 画出人脸
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi = gray[x: x + w, y: y + h]
            try:
                # 更改大小
                roi = cv.resize(roi, (200, 200))
                # 进行预测\
                global count
                global label
                params = model.predict(roi)
                # params是一个二元素列表, 第一个元素是预测结果，第二个元素是得分情况
                # 在图像上写预测结果  添加的文字， 左上角坐标
                # 1.2是字体大小 2是粗细
                if params[1] < 70:
                    label = params[0]
                    frame = cv.putText(frame, str("yang"), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    count += 1

                # print(params)
            except Exception:
                pass
        cv.imshow("detect face", frame)
        if cv.waitKey(50) == 27:
            break

if __name__ == '__main__':
    # 调用摄像头 采集人脸照片数据
    # face_collect()

    # 检测人脸
    face_rec()