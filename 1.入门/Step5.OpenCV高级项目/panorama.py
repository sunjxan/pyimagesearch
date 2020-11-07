import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

import imutils
from imutils.Stitcher import Stitcher

imageA = cv2.imread("imageA.png")
imageB = cv2.imread("imageB.png")
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

stitcher = Stitcher()
result, vis = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey()
cv2.destroyAllWindows()