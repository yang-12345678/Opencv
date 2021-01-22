# -*- coding=GBK -*-
import cv2 as cv
import numpy as np


# 模版匹配
def template_image():
    tpl = cv.imread("C://2.jpg")
    target = cv.imread("C://1.jpg")
    cv.imshow("模板", tpl)
    cv.imshow("原图", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("匹配" + np.str(md), target)


template_image()
cv.waitKey(0)
cv.destroyAllWindows()