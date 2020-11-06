import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append('../..')

import cv2
import numpy as np

image = cv2.imread("hand_01.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

ret, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
maxContour = max(cnts, key=cv2.contourArea)

left, top = maxContour.argmin(axis=0)[0]
right, bottom = maxContour.argmax(axis=0)[0]

extLeft = tuple(maxContour[left, 0])
extRight = tuple(maxContour[right, 0])
extTop = tuple(maxContour[top, 0])
extBottom = tuple(maxContour[bottom, 0])

cv2.drawContours(image, [maxContour], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBottom, 8, (255, 255, 0), -1)

cv2.imshow("Image", image)
cv2.waitKey()
cv2.destroyAllWindows()