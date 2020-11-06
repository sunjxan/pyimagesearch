import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append('../..')

import cv2
import numpy as np

import imutils

image = cv2.imread("pill_01.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 20, 100)

cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts):
    maxContour = max(cnts, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(maxContour)
    imageROI = image[y:y+h, x:x+w]
    maskROI = mask[y:y+h, x:x+w]
    imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)

    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate(imageROI, angle)
        cv2.imshow("Rotated (Problematic)", rotated)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate_bound(imageROI, angle)
        cv2.imshow("Rotated (Bound)", rotated)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    maxContour[:, :, 0] -= x
    maxContour[:, :, 1] -= y
    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate_contour(imageROI, maxContour, angle, 1.0)
        cv2.imshow("Rotated (Contour)", rotated)
        cv2.waitKey(100)
    cv2.destroyAllWindows()