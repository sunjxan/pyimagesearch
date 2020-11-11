import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import cv2
import numpy as np

from imutils.video_capture import playVideo

def find_targets(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    found = 0
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, .02 * peri, True)
        count = len(approx)
        # 确保近似轮廓为大致矩形
        if count < 4 or count > 6:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # 宽高大致相等
        aspectRatio = w / h
        # 原始面积
        area = cv2.contourArea(cnt)
        # 凸包面积
        hullArea = cv2.contourArea(cv2.convexHull(cnt))
        # 坚固度 =原始面积/凸包面积，坚固度要接近1
        solidity = area / hullArea

        conditions = []
        conditions.append(w >= 25 and h >= 25)
        conditions.append(aspectRatio >= .8 and aspectRatio <= 1.2)
        conditions.append(solidity > .9)

        # 所以条件都满足
        if all(conditions):
            found += 1
            M = cv2.moments(cnt)
            cX = round(M["m10"] / M["m00"])
            cY = round(M["m01"] / M["m00"])
            cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)
            cv2.line(output, (round(cX - w * .15), cY), (round(cX + w * .15), cY), (0, 0, 255), 2)
            cv2.line(output, (cX, round(cY - h * .15)), (cX, round(cY + h * .15)), (0, 0, 255), 2)
    if found == 0:
        status = "No Targets"
    elif found == 1:
        status = "1 Target Acquired"
    else:
        status = "{} Targets Acquired".format(found)
    cv2.putText(output, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    return output

def captureFunc(frame, frameIndex):
    output = find_targets(frame)
    cv2.imshow("Output", output)

def endFunc():
    cv2.destroyWindow("Output")

playVideo("drone.avi", fps=20.0, winname="Input", captureFunc=captureFunc, endFunc=endFunc)