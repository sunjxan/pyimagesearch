import os
os.environ['DISPLAY'] = 'windows:0'

import cv2
import math
import numpy as np

image = cv2.imread("pill_01.png")
h, w, d = image.shape

center = w / 2, h / 2
for angle in np.arange(0, 360, 15):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imshow("Rotated (Problematic)", rotated)
    cv2.waitKey(100)

def getBoundRotationMatrix2D(bound, angle, scale):
    w, h = bound
    M = cv2.getRotationMatrix2D((0, 0), angle, scale)
    points = np.array([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
    points = np.matmul(M[:, :2], points.T).T
    minX, minY = np.min(points, axis=0)
    maxX, maxY = np.max(points, axis=0)
    M[:, 2] = [-minX, -minY]
    nW = math.ceil(maxX - minX + 1)
    nH = math.ceil(maxY - minY + 1)
    return M, (nW, nH)

for angle in np.arange(0, 360, 15):
    M, (nW, nH) = getBoundRotationMatrix2D((w, h), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (nW, nH))
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(100)
