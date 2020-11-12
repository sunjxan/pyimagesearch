import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

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
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

if status == 0:
    stitched = Stitcher().removeBlackBorder(stitched, showAnimate=True, winname="Stitched")
    cv2.imshow("Stitched", stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("[INFO] image stitching failed ({})".format(status))