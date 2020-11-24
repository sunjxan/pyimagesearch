import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

image = cv2.imread("jp.png")

# 边框
border =  cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_ISOLATED)
plt.subplot(2, 3, 1)
plt.imshow(border[..., (2, 1, 0)])
plt.title('BORDER_ISOLATED')

border =  cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 255))
plt.subplot(2, 3, 2)
plt.imshow(border[..., (2, 1, 0)])
plt.title('BORDER_CONSTANT')

border =  cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REPLICATE)
plt.subplot(2, 3, 3)
plt.imshow(border[..., (2, 1, 0)])
plt.title('BORDER_REPLICATE')

border =  cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_WRAP)
plt.subplot(2, 3, 4)
plt.imshow(border[..., (2, 1, 0)])
plt.title('BORDER_WRAP')

border =  cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT)
plt.subplot(2, 3, 5)
plt.imshow(border[..., (2, 1, 0)])
plt.title('BORDER_REFLECT')

border =  cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)
plt.subplot(2, 3, 6)
plt.imshow(border[..., (2, 1, 0)])
plt.title('BORDER_REFLECT_101')
plt.show()

# 融合
background = np.ones(image.shape, dtype=np.uint8) * 255
mixed = cv2.addWeighted(image, .5, background, .5, 0)
plt.subplot(1, 2, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(mixed[..., (2, 1, 0)])
plt.title('Mixed')
plt.show()