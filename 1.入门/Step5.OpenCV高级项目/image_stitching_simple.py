import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

import imutils

imageA = cv2.imread("imageA.png")
imageB = cv2.imread("imageB.png")
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# 创建拼接器
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch([imageA, imageB])

if status == 0:
    cv2.imshow("Image A ({} x {})".format(imageA.shape[1], imageA.shape[0]), imageA)
    cv2.imshow("Image B ({} x {})".format(imageB.shape[1], imageB.shape[0]), imageB)
    cv2.imshow("stitched ({} x {})".format(stitched.shape[1], stitched.shape[0]), stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
	print("[INFO] image stitching failed ({})".format(status))