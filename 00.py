# keep calm and carry on
# frame_nms.py
import cv2
import numpy as np
from time import sleep
from tkinter import *
from tkinter import messagebox
import time
import json
import requests
import base64


def storePic(img):
    date = time.strftime('%Y%m%d %H%M%S')
    fileName = "./photos/" + date + ".jpg"
    cv2.imwrite(fileName, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print("保存成功")


def Exit():
    response = messagebox.askokcancel("exit", "请问您真的要退出么？")
    if response:
        master.destroy()


def twiceDetect(img):
    global z
    global twiceCheck
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    B1 = img[:, :, 0] / 255
    G1 = img[:, :, 1] / 255
    R1 = img[:, :, 2] / 255
    minValue = np.array(
        np.where(R1 <= G1, np.where(G1 <= B1, R1, np.where(R1 <= B1, R1, B1)), np.where(G1 <= B1, G1, B1)))
    sumValue = R1 + G1 + B1
    # HSI中S分量计算公式
    S = np.array(np.where(sumValue != 0, (1 - 3.0 * minValue / sumValue), 0))
    Sdet = (255 - R) / 20
    SThre = ((255 - R) * sThre / redThre)
    # 判断条件
    fireImg = np.array(
        np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S > 0, np.where(S > Sdet, np.where(
            S >= SThre, 255, 0), 0), 0), 0), 0), 0))

    gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
    gray_fireImg[:, :, 0] = fireImg
    meBImg = cv2.medianBlur(gray_fireImg, 5)
    kernel = np.ones((5, 5), np.uint8)
    ProcImg = cv2.dilate(meBImg, kernel)
    # 绘制矩形框
    contours, _ = cv2.findContours(ProcImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ResImg = img.copy()
    twiceCheck = twiceCheck + 1
    if len(contours):
        # 获取矩形的左上角坐标(x,y)，以及矩形的宽和高w、h
        x, y, w, h = cv2.boundingRect(contours[0])
        l_top = (x, y)
        r_bottom = (x + w, y + h)
        cv2.rectangle(ResImg, l_top, r_bottom, (255, 0, 0), 2)
        cv2.imshow("RESULT", ResImg)
        z = z + 1
        twiceCheck = 0


def detect(image):
    global thirdDe
    global aiCheck
    data = {'image': base64.b64encode(cv2.imencode('.png', image)[1]).decode()}
    response = requests.post(request_url, data=json.dumps(data))
    content = response.json()
    print(content)
    results = content['results']
    print(content)
    aiCheck = aiCheck + 1
    if results:
        result = results[0]['location']
        score = results[0]['score']
        if score > 0.4:
            thirdDe = thirdDe + 1
            print(thirdDe)
        text = 'fire' + '   ' + str(score)
        x, y, w, h = result['left'], result['top'], result['width'], result['height']
        cv2.rectangle(image, (x, y), (x + w, y + h), (128, 128, 0), 2)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)
        aiCheck = 0
    cv2.imshow('aaaaa', image)


def py_cpu_nms(dets, thresh):
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = y1 + dets[:, 3]
    x2 = x1 + dets[:, 2]

    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]

    # keep为最后保留的边框
    keep = []

    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


class Detector(object):
    i = 0

    def __init__(self, name='my_video', frame_num=10, k_size=7):

        self.name = name

        self.nms_threshold = 0.5

        self.time = 1 / frame_num

        self.es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))

    def catch_video(self, video_index=0, k_size=7,
                    iterations=3, threshold=20, bias_num=1,
                    min_area=360, show_test=True, enhance=True):
        global twiceCheck
        global aiCheck
        global z
        if not bias_num > 0:
            raise Exception('bias_num must > 0')

        if isinstance(video_index, str):
            is_camera = False
        else:
            is_camera = True

        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')
        global hsv_detect
        frame_num = 0
        previous = []
        i = 0
        while cap.isOpened():
            catch, frame = cap.read()  # 读取每一帧图片
            copy = frame.copy()
            if not catch:
                raise Exception('Unexpected Error.')
            if frame_num < bias_num:
                value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                previous.append(value)
                frame_num += 1
            raw = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.absdiff(gray, previous[0])
            gray = cv2.medianBlur(gray, k_size)
            ret, mask = cv2.threshold(
                gray, threshold, 255, cv2.THRESH_BINARY)
            if enhance:
                mask = cv2.dilate(mask, self.es, iterations)
                mask = cv2.erode(mask, self.es, iterations)

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounds = self.nms_cnts(cnts, mask, min_area)
            if len(bounds):
                bound = bounds[0]
                x, y, w, h = bound
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                i = i + 1
                hsv_detect = frame[y:y + h, x:x + w]
            # cv2.cvtColor(frame[y + 10:y + h, x:x + w], cv2.COLOR_BGR2HSV)
            # cv2.imshow("detect", frame[y - 50:y + h + 50, x - 50:x + w + 50])

            # cv2.cvtColor(hsv_detect, cv2.COLOR_BGR2HSV)
            # ma = cv2.inRange(hsv_detect, lower_hsv, upper_hsv)
            # print(ma)
            # hsv = cv2.bitwise_and(hsv_detect, hsv_detect, mask=ma)
            # cv2.imshow("fras", hsv)
            if i > 15:
                twiceDetect(img=hsv_detect)
            if twiceCheck > 10:
                z = 0
                i = 0
                twiceCheck = 0
            if z > 10:
                detect(image=copy)
            if thirdDe > 10:
                print("发生火灾，请注意预防")
                storePic(copy)
            if aiCheck > 50:
                z = 0
                i = 0
                twiceCheck = 0
            if not is_camera:
                sleep(self.time)
            cv2.imshow(self.name, frame)  # 在window上显示图片
            if show_test:
                cv2.imshow(self.name + '_frame', mask)  # 边界
            value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            previous = self.pop(previous, value)
            cv2.waitKey(10)

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

            if show_test and cv2.getWindowProperty(self.name + '_frame', cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()

    def nms_cnts(self, cnts, mask, min_area):

        bounds = [cv2.boundingRect(
            c) for c in cnts if cv2.contourArea(c) > min_area]

        if len(bounds) == 0:
            return []

        scores = [self.calculate(b, mask) for b in bounds]

        bounds = np.array(bounds)

        scores = np.expand_dims(np.array(scores), axis=-1)

        keep = py_cpu_nms(np.hstack([bounds, scores]), self.nms_threshold)

        return bounds[keep]

    def calculate(self, bound, mask):

        x, y, w, h = bound

        area = mask[y:y + h, x:x + w]

        pos = area > 0 + 0

        score = np.sum(pos) / (w * h)

        return score

    def pop(self, l, value):

        l.pop(0)
        l.append(value)
        return l


def startDetect():
    detector = Detector()
    detector.catch_video("111.avi", bias_num=2, iterations=3,
                         k_size=5, show_test=True, enhance=False)  # 第一个参数可以是数字（


if __name__ == "__main__":
    access_token = '24.553b930866afcc491abdfe60a38806b4.2592000.1621752026.282335-24046764'
    request_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/detection/littlefiredetect" + "?access_token=" + access_token
    thirdDe = 0
    z = 0
    twiceCheck = 0
    aiCheck = 0
    redThre = 115  # 115~135红色分量阈值
    sThre = 60  # 55~65饱和度阈值
    win = Tk
    entry_acc = Entry
    master = Tk()
    master.title("欢迎")
    master.geometry("450x400+500+200")
    canvas = Canvas(master, height=130, width=440)
    image3 = PhotoImage(file="welcome.gif")
    canvas.create_image(0, 0, anchor='nw', image=image3)
    canvas.grid(row=0, column=0, columnspan=2)
    Label(text="亲爱管理员\n"
               "这里是烟火防控\n"
               "请选择你的操作：", font="微软雅黑 14", justify=LEFT).grid(row=1, column=0, columnspan=2, sticky='w')
    Button(master, text="开始检测", font="微软雅黑 14", relief="solid", command=startDetect).grid(sticky='w', row=3,
                                                                                          column=0,
                                                                                          padx=10, pady=20)
    Button(master, text="退出", font="微软雅黑 14", relief="solid").grid(sticky='e', row=3, column=1,
                                                                   padx=20,
                                                                   pady=20)
    master.mainloop()
