import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append('../..')

import cv2
import numpy as np

boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

image = cv2.imread("color_detection_blue_version.jpg")
for lower, upper in boundaries:
    lowerb = np.array(lower, dtype=np.uint8)
    upperb = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(image, lowerb, upperb)
    output = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(500)
cv2.destroyAllWindows()