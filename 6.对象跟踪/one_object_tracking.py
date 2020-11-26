import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
import math
from collections import deque

from imutils.video_capture import playVideo
from imutils.video_capture import captureCamera

# 区域颜色范围
lower = (15, 0, 100)
upper = (35, 155, 255)
# 中心点队列
ptSize = 30
pts = deque(maxlen=ptSize)

def captureFunc(frame, frameIndex):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)

    cnts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
        maxContour = max(cnts, key=cv2.contourArea)

        M = cv2.moments(maxContour)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        center = (cX, cY)

        if cv2.contourArea(maxContour) > 500:
            # 计算最大轮廓的旋转边界框，返回矩形格式 ((中心点横坐标， 中心点纵坐标), (矩形宽度, 矩形高度), 旋转角度)
            rect = cv2.minAreaRect(maxContour)
            box = cv2.boxPoints(rect).round().astype(np.int32)
            cv2.drawContours(frame, [box], -1, (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # 将该帧的寻找结果加入队列
    pts.appendleft(center)

    count = len(pts)
    # 将队列中的中心点列表连接起来
    for i in range(1, count):
        # 消失的部分不画
        if pts[i - 1] is None or pts[i] is None:
            continue
        # 连线宽度由新到旧变小
        thickness = round(math.sqrt(ptSize / (i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow('Frame', frame)
    return frame

captureCamera('http://192.168.0.101:4747/video', captureFunc=captureFunc)