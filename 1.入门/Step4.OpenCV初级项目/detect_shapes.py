import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

from imutils.shape_detector import ShapeDetector

image = cv2.imread("shapes_and_colors.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sd = ShapeDetector()
for cnt in cnts:
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    cX = round(M["m10"] / M["m00"])
    cY = round(M["m01"] / M["m00"])
    cv2.putText(image, sd.detect(cnt), (cX - 10, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(500)
cv2.destroyAllWindows()