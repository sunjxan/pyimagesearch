import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

from imutils.Stitcher import Stitcher

print("[INFO] loading images...")
images = []
index = 0
while True:
    image = cv2.imread("image_stitch_{}.jpg".format(index))
    if image is None:
        break
    images.append(image)
    index += 1

print("[INFO] stitching images...")
# 创建拼接器
stitcher = Stitcher()
stitched = stitcher.stitch(images, showMatches=True)

if stitched is not None:
    stitched = stitcher.removeBlackBorder(stitched, showAnimate=True, winname="Stitched")
    cv2.imshow("Stitched", stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("[INFO] image stitching failed")