import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

image = cv2.imread("color_detection_blue_version.jpg")

plt.subplot(2, 3, 1)
plt.imshow(image[..., (2, 1, 1)])

for index, (lower, upper) in enumerate(boundaries):
    lowerb = np.array(lower, dtype=np.uint8)
    upperb = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(image, lowerb, upperb)
    output = cv2.bitwise_and(image, image, mask=mask)
    plt.subplot(2, 3, index + 2)
    plt.imshow(output[..., (2, 1, 0)])
plt.show()