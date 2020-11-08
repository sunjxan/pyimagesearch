import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

import imutils
from imutils.transform import four_point_transform
from imutils.contours import sort_contours

image = cv2.imread("digits.jpg")
image = imutils.resize(image, height=500)
cv2.imshow("Input", image)
cv2.waitKey()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200)

cnts, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# 找到显示屏
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, .02 * peri, True)
    if len(approx) == 4:
        displayCnt = cnt
        break

# 透视变换
warped = four_point_transform(gray, displayCnt.squeeze())
output = four_point_transform(image, displayCnt.squeeze())

# 使用大津算法确定的阈值进行阈值化，将显示的内容凸显出来
ret, thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 使用形态学闭运算操作，让每个数字的各个区域连在一起
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)

cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digitCnts = []
for cnt in cnts:
    x, y, w, h =  cv2.boundingRect(cnt)
    # 根据单个数字显示区域宽高比确定轮廓
    if 2 * w > h and 1.5 * w < h:
        digitCnts.append(np.array([(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)]))

# 将数字轮廓从左到右排列
digitCnts = sort_contours(digitCnts)
# 展示数字轮廓
img_digits = output.copy()
cv2.drawContours(img_digits, digitCnts, -1, (0, 255, 0), 2)
cv2.imshow("Digits", img_digits)
cv2.waitKey()

# 查找表，短线排序：先上下排序，后左右排序
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# 找出数字的短线部分
img_lines = output.copy()
digits = []
for cnt in digitCnts:
    x, y, w, h = cv2.boundingRect(cnt)
    # 设置短线宽度
    lineWidth = .2 * w

    # 七段小短线在数字轮廓中的区域
    segments = np.array([
        (0, 0, w, lineWidth),  # top
        (0, 0, lineWidth, h / 2),  # top-left
        (w - lineWidth, 0, lineWidth, h / 2),  # top-right
        (0, h / 2 - lineWidth / 2, w, lineWidth),  # center
        (0, h / 2, lineWidth, h / 2),  # bottom-left
        (w - lineWidth, h / 2, lineWidth, h / 2),  # bottom-right
        (0, h - lineWidth, w, lineWidth)  # bottom
    ]).round().astype(np.int32)
    on = [0] * len(segments)
    # 找到每个数字的短线区域
    for index, (xS, yS, wS, hS) in enumerate(segments):
        xS += x
        yS += y
        area = (wS - 1) * (hS - 1)
        count = cv2.countNonZero(thresh[yS:yS+hS, xS:xS+wS])
        # 如果短线区域内有显示的部分占0.5以上，就当作显示
        if count / area > .5:
            on[index] = 1
            img_lines[yS:yS+hS, xS:xS+wS] = [0, 255, 0]
    # 按短线分布在查找表中找出数字
    on = tuple(on)
    if on in DIGITS_LOOKUP:
        digits.append(DIGITS_LOOKUP[on])

# 打印数字结果
print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Lines", img_lines)
cv2.waitKey()
cv2.destroyAllWindows()