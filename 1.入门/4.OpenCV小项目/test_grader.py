import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import cv2
import numpy as np

from imutils.transform import four_point_transform
from imutils.contours import sort_contours

image = cv2.imread("omr_test_01.png")
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey()
cv2.destroyAllWindows()

cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
docCnt = None

for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, .02 * peri, True)

    if len(approx) == 4:
        docCnt = approx
        break

print("SETP 2: Find contours of paper")
cv2.drawContours(image, [docCnt], -1, (0, 0, 255), 2)
cv2.imshow("Outline", image)
cv2.waitKey()
cv2.destroyAllWindows()

print("STEP 3: Apply perspective transform")
paper = four_point_transform(original, docCnt.squeeze())
warped = four_point_transform(gray, docCnt.squeeze())
ret, thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Paper", paper)
cv2.imshow("Thresh", thresh)
cv2.waitKey()
cv2.destroyAllWindows()

print("STEP 4: Find contours of bubbles")
cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
questionCnts = []

for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / h

    if w >= 20 and h >= 20 and ar >= .9 and ar <= 1.1:
        questionCnts.append(cnt)

output = paper.copy()
cv2.drawContours(output, questionCnts, -1, (0, 0, 255), 2)
cv2.imshow("Questions", output)
cv2.waitKey()
cv2.destroyAllWindows()

print("STEP 5: Find checked bubbles")
questionCnts = sort_contours(questionCnts, "top-to-bottom")
count = len(questionCnts)

output = paper.copy()
for start in range(0, count, 5):
    rowCnts = questionCnts[start:start+5]
    rowCnts = sort_contours(rowCnts)
    maxI = maxV = 0
    for index in range(0, 5):
        mask = np.zeros(thresh.shape, np.uint8)
        cv2.drawContours(mask, rowCnts, index, (255, 255, 255), -1)
        img = cv2.bitwise_and(thresh, thresh, mask=mask)
        value = cv2.countNonZero(img)
        if value > maxV:
            maxV = value
            maxI = index
    cv2.drawContours(output, rowCnts, maxI, (0, 0, 255), 2)
    print("Row {}: {}".format(round(start / 5), chr(ord('A') + maxI)))

cv2.imshow("Exam", output)
cv2.waitKey()
cv2.destroyAllWindows()