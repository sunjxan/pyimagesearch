import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

image = cv2.imread("barcode_01.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Scharr运算符（ksize=-1）构造水平和垂直方向上灰度图像的梯度幅度表示
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# Scharr运算符的x梯度中减去Scharr运算符的y梯度
# 通过执行该减法运算，我们剩下的图像区域具有高水平梯度和低垂直梯度
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("Gradient", gradient)
cv2.waitKey()

# 平均滤波
blurred = cv2.blur(gradient, (9, 9))
cv2.imshow("Blurred", blurred)
cv2.waitKey()
# 阈值处理
ret, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresh", thresh)
cv2.waitKey()

# 形态学操作，先闭运算，后开运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey()
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, None, iterations=4)
cv2.imshow("Opened", opened)
cv2.waitKey()

# 找出最大轮廓
cnts, hier = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
maxContour = max(cnts, key=cv2.contourArea)
mask = np.zeros(thresh.shape, np.uint8)
cv2.drawContours(mask, [maxContour], -1, 255, -1)
cv2.imshow("Max Contour", cv2.bitwise_and(image, image, mask=mask))
cv2.waitKey()

# 计算最大轮廓的旋转边界框
rect = cv2.minAreaRect(maxContour)
box = cv2.boxPoints(rect).round().astype(np.int32)
mask = np.zeros(thresh.shape, np.uint8)
cv2.drawContours(mask, [box], -1, 255, -1)
cv2.imshow("Box", cv2.bitwise_and(image, image, mask=mask))
cv2.waitKey()

cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey()
cv2.destroyAllWindows()