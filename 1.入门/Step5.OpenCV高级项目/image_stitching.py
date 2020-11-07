import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

import imutils

imageA = cv2.imread("imageA.png")
imageB = cv2.imread("imageB.png")
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# 创建拼接器
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch([imageA, imageB])

if status == 0:
    cv2.imshow("Image A ({} x {})".format(imageA.shape[1], imageA.shape[0]), imageA)
    cv2.imshow("Image B ({} x {})".format(imageB.shape[1], imageB.shape[0]), imageB)

    # 四周加上10像素的黑色边框，保证可以从四个方向进行腐蚀
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    # 得到目标区域轮廓
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(cnts, key=cv2.contourArea)
    # 得到目标区域外接矩形
    x, y, w, h = cv2.boundingRect(maxContour)
    boundingRect = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.rectangle(boundingRect, (x, y), (x + w - 1, y + h - 1), 255, -1)
    # 预览外接矩形内的拼接结果
    cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
    cv2.waitKey(1000)

    # 外接矩形减去目标区域
    sub = cv2.subtract(boundingRect, thresh)
    # 腐蚀外接矩形，每次向内缩减1像素，直到完全在目标区域内部
    while cv2.countNonZero(sub) > 0:
        boundingRect = cv2.erode(boundingRect, None)
        sub = cv2.subtract(boundingRect, thresh)
        cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
        cv2.waitKey(30)

    # 得到新的矩形轮廓
    cnts, hier = cv2.findContours(boundingRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(cnts, key=cv2.contourArea)
    nX, nY, nW, nH = cv2.boundingRect(maxContour)

    left = nX
    right = nX + nW
    top = nY
    bottom = nY + nH

    # 1. 分别从左右两个方向对内部矩形进行膨胀，找到满足条件的最大矩形
    while left > x:
        left = left - 1
        boundingRect[top:bottom, left] = 255
        # 预览外接矩形内的拼接结果
        cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
        cv2.waitKey(30)
        sub = cv2.subtract(boundingRect, thresh)
        if cv2.countNonZero(sub) > 0:
            boundingRect[top:bottom, left] = 0
            # 预览外接矩形内的拼接结果
            cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
            cv2.waitKey(30)
            left = left + 1
            break
    while right < x + w:
        right = right + 1
        boundingRect[top:bottom, right - 1] = 255
        # 预览外接矩形内的拼接结果
        cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
        cv2.waitKey(30)
        sub = cv2.subtract(boundingRect, thresh)
        if cv2.countNonZero(sub) > 0:
            boundingRect[top:bottom, right - 1] = 0
            # 预览外接矩形内的拼接结果
            cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
            cv2.waitKey(30)
            right = right - 1
            break

    # 2. 分别从上下两个方向对内部矩形进行膨胀，找到满足条件的最大矩形
    while top > y:
        top = top - 1
        boundingRect[top, left:right] = 255
        # 预览外接矩形内的拼接结果
        cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
        cv2.waitKey(30)
        sub = cv2.subtract(boundingRect, thresh)
        if cv2.countNonZero(sub) > 0:
            boundingRect[top, left:right] = 0
            # 预览外接矩形内的拼接结果
            cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
            cv2.waitKey(30)
            top = top + 1
            break
    while bottom < y + h:
        bottom = bottom + 1
        boundingRect[bottom - 1, left:right] = 255
        # 预览外接矩形内的拼接结果
        cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
        cv2.waitKey(30)
        sub = cv2.subtract(boundingRect, thresh)
        if cv2.countNonZero(sub) > 0:
            boundingRect[bottom - 1, left:right] = 0
            # 预览外接矩形内的拼接结果
            cv2.imshow("View", cv2.bitwise_and(stitched, stitched, mask=boundingRect))
            cv2.waitKey(30)
            bottom = bottom - 1
            break

    cv2.destroyWindow("View")

    # 获得没有黑色区域的拼接结果
    stitched = stitched[top:bottom, left:right]

    cv2.imshow("stitched ({} x {})".format(stitched.shape[1], stitched.shape[0]), stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("[INFO] image stitching failed ({})".format(status))