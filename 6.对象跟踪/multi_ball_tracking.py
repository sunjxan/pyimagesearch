import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
import math
from collections import deque

import imutils
from imutils.video_capture import playVideo
from imutils.video_capture import captureCamera
from imutils.centroid_tracker import CentroidTracker

# 区域颜色范围
greenLower = (15, 100, 60)
greenUpper = (35, 160, 120)

# 创建质心追踪器
centroidTracker = CentroidTracker()

def captureFunc(frame, frameIndex):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)

    cnts, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []

    for cnt in cnts:
        M = cv2.moments(cnt)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        center = (cX, cY)

        if cv2.contourArea(cnt) > 500:
            centroids.append(center)
            # 计算最大轮廓的旋转边界框，返回矩形格式 ((中心点横坐标， 中心点纵坐标), (矩形宽度, 矩形高度), 旋转角度)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).round().astype(np.int32)
            cv2.drawContours(frame, [box], -1, (0, 255, 255), 2)
            cv2.circle(frame, center, 4, (0, 255, 0), -1)

    # 更新质心列表
    objects = centroidTracker.update(centroids)

    for objectID, centroid in objects.items():
        # 消失的不画
        if centroidTracker.disappeared[objectID] == 0:
            cv2.putText(frame, "ID {}".format(objectID), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    return frame

playVideo("../1.入门/OpenCV高级项目/vtest.avi", captureFunc=captureFunc)
# captureCamera(0, captureFunc=captureFunc)