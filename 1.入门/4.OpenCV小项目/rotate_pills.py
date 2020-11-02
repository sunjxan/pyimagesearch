import os
os.environ['DISPLAY'] = 'windows:0'

import cv2
import math
import numpy as np

image = cv2.imread("pill_01.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 20, 100)

cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts):
    maxContour = max(cnts, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(maxContour)
    imageROI = image[y:y+h, x:x+w]
    maskROI = mask[y:y+h, x:x+w]
    imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)

    for angle in np.arange(0, 360, 15):
        center = w / 2, h / 2
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(imageROI, M, (w, h))
        cv2.imshow("Rotated (Problematic)", rotated)
        cv2.waitKey(100)


    def getContourRotationMatrix2D(contour, angle, scale):
        M = cv2.getRotationMatrix2D((0, 0), angle, scale)
        contour = contour.squeeze()
        contour = np.matmul(M[:, :2], contour.T).T
        minX, minY = np.min(contour, axis=0)
        maxX, maxY = np.max(contour, axis=0)
        M[:, 2] = [-minX, -minY]
        nW = math.ceil(maxX - minX + 1)
        nH = math.ceil(maxY - minY + 1)
        return M, (nW, nH)

    keyContour = maxContour
    keyContour[:, :, 0] -= x
    keyContour[:, :, 1] -= y
    for angle in np.arange(0, 360, 15):
        M, (nW, nH) = getContourRotationMatrix2D(keyContour, angle, 1.0)
        rotated = cv2.warpAffine(imageROI, M, (nW, nH))
        cv2.imshow("Rotated (Correct)", rotated)
        cv2.waitKey(100)