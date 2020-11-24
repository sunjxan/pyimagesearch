import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

image = cv2.imread("tetris_blocks.png")
plt.subplot(2, 2, 1)
plt.imshow(image[..., (2, 1, 0)])
plt.title('Image')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Gray')

edged = cv2.Canny(gray, 30, 150)
plt.subplot(2, 2, 3)
plt.imshow(edged, cmap='gray')
plt.title('Edged')

# 默认将灰度值高的部分作为255，THRESH_BINARY_INV设置相反
ret, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
plt.subplot(2, 2, 4)
plt.imshow(thresh, cmap='gray')
plt.title('Thresh')
plt.show()

cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()
for cnt in cnts:
    cv2.drawContours(output, [cnt], -1, (240, 0, 159), 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(300)

text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey()
cv2.destroyAllWindows()

# 腐蚀：大体上缩小一圈，可以让细小的突起轮廓消失
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
plt.subplot(2, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title('Eroded')

# 膨胀：大体上扩大一圈，可以让细小的缝隙连成一体
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
plt.subplot(2, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Dilated')

# 形态学操作
mask = thresh.copy()
# 开运算：先腐蚀iterations次后膨胀iterations次
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=5)
plt.subplot(2, 2, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Morph Open')

mask = thresh.copy()
# 闭运算：先膨胀iterations次后腐蚀iterations次
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=5)
plt.subplot(2, 2, 4)
plt.imshow(mask, cmap='gray')
plt.title('Morph Close')
plt.show()

mask = thresh.copy()
output = cv2.bitwise_not(image, mask=mask)
plt.subplot(2, 2, 1)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Bitwise Not')

mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
plt.subplot(2, 2, 2)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Bitwise And')

mask = thresh.copy()
output = cv2.bitwise_or(image, image, mask=mask)
plt.subplot(2, 2, 3)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Bitwise Or')

mask = thresh.copy()
output = cv2.bitwise_xor(image, image, mask=mask)
plt.subplot(2, 2, 4)
plt.imshow(output[..., (2, 1, 0)])
plt.title('Bitwise Xor')
plt.show()