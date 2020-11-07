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

cv2.imshow("Image A ({} x {})".format(imageA.shape[1], imageA.shape[0]), imageA)
cv2.imshow("Image B ({} x {})".format(imageB.shape[1], imageB.shape[0]), imageB)
cv2.imshow("Keypoint Matches ({} x {})".format(vis.shape[1], vis.shape[0]), vis)
cv2.imshow("Result ({} x {})".format(result.shape[1], result.shape[0]), result)
cv2.waitKey()
cv2.destroyAllWindows()