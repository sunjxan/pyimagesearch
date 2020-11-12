import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

image = cv2.imread("tetris_blocks.png")

# Sobel算子
sobelX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0)
sobelX = np.abs(sobelX).round().clip(0, 255).astype(np.uint8)
plt.subplot(2, 2, 1)
plt.imshow(sobelX[..., (2, 1, 0)])
plt.title('Sobel X')

sobelY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1)
sobelY = np.abs(sobelY).round().clip(0, 255).astype(np.uint8)
plt.subplot(2, 2, 2)
plt.imshow(sobelY[..., (2, 1, 0)])
plt.title('Sobel Y')

# Scharr算子
scharrX = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=1, dy=0)
scharrX = np.abs(scharrX).round().clip(0, 255).astype(np.uint8)
plt.subplot(2, 2, 3)
plt.imshow(scharrX[..., (2, 1, 0)])
plt.title('Scharr X')

scharrY = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=0, dy=1)
scharrY = np.abs(scharrY).round().clip(0, 255).astype(np.uint8)
plt.subplot(2, 2, 4)
plt.imshow(scharrY[..., (2, 1, 0)])
plt.title('Scharr Y')
plt.show()

# Laplacian 算子
laplac = cv2.Laplacian(image, ddepth=cv2.CV_32F)
laplac = np.abs(laplac).round().clip(0, 255).astype(np.uint8)
plt.imshow(laplac)
plt.title('Laplacian')
plt.show()