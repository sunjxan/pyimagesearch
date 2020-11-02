import os
os.environ['DISPLAY'] = 'windows:0'

import cv2
import numpy as np

image = cv2.imread("shapes.png")
lowerb = np.array([0] * 3)
upperb = np.array([15] * 3)
mask = cv2.inRange(image, lowerb, upperb)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", mask)
cv2.waitKey()

for cnt in cnts:
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(400)