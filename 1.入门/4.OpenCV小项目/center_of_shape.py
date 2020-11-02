import os
os.environ['DISPLAY'] = 'windows:0'

import cv2
import numpy as np

image = cv2.imread("shapes_and_colors.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)

cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for index, cnt in enumerate(cnts):
    M = cv2.moments(cnt)
    cX = round(M["m10"] / M["m00"])
    cY = round(M["m01"] / M["m00"])
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, str(index + 1), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(400)