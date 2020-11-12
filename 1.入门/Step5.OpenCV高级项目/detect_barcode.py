import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

def detect_barcode(image):
    image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 2, 1)
    plt.imshow(image[..., (2, 1, 0)])
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray")
    plt.show()

    # 使用Scharr运算符（ksize=-1）构造水平和竖直方向上灰度图像的梯度幅度表示，并取绝对值
    gradX = np.abs(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
    gradY = np.abs(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1))

    # 因为条形码区域水平和竖直方向梯度相差大，所以相减
    # Scharr运算符的x梯度绝对值中减去Scharr运算符的y梯度绝对值，并取绝对值
    gradient = np.abs(cv2.subtract(gradX, gradY))

    # 取整到[0, 255]，使具有高水平梯度绝对值的图像区域亮度高
    gradX = gradX.round().clip(0, 255).astype(np.uint8)
    # 取整到[0, 255]，使具有高竖直梯度绝对值的图像区域亮度高
    gradY = gradY.round().clip(0, 255).astype(np.uint8)
    # 取整到[0, 255]，使具有水平梯度绝对值和竖直梯度绝对值相差大的图像区域亮度高
    gradient = gradient.round().clip(0, 255).astype(np.uint8)

    plt.subplot(1, 3, 1)
    plt.imshow(gradX, cmap='gray')
    plt.title("Gradient X")
    plt.subplot(1, 3, 2)
    plt.imshow(gradY, cmap='gray')
    plt.title("Gradient Y")
    plt.subplot(1, 3, 3)
    plt.imshow(gradient, cmap='gray')
    plt.title("Gradient Difference")
    plt.show()

    # 平均滤波
    blurred = cv2.blur(gradient, (5, 5))

    # 阈值化
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作，先闭运算，后开运算
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None, iterations=4)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, None, iterations=4)

    plt.subplot(2, 2, 1)
    plt.imshow(blurred, cmap='gray')
    plt.title("Blurred")
    plt.subplot(2, 2, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresh")
    plt.subplot(2, 2, 3)
    plt.imshow(closed, cmap='gray')
    plt.title("Closed")
    plt.subplot(2, 2, 4)
    plt.imshow(opened, cmap='gray')
    plt.title("Opened")
    plt.show()

    plt.subplot(2, 2, 1)
    plt.imshow(image[..., (2, 1, 0)])
    plt.title("Original")

    # 找出最大轮廓
    cnts, hier = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    maxContour = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.bitwise_and(image, image, mask=mask)[..., (2, 1, 0)])
    plt.title("Max Contour")

    # 计算最大轮廓的旋转边界框，返回矩形格式 ((中心点横坐标， 中心点纵坐标), (矩形宽度, 矩形高度), 旋转角度)
    rect = cv2.minAreaRect(maxContour)
    # 计算边界框的四个顶点坐标
    box = cv2.boxPoints(rect).round().astype(np.int32)
    mask = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask, [box], -1, 255, -1)

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.bitwise_and(image, image, mask=mask)[..., (2, 1, 0)])
    plt.title("Box")

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    plt.subplot(2, 2, 4)
    plt.imshow(image[..., (2, 1, 0)])
    plt.title("Result")
    plt.show()

    # 返回矩形四个顶点坐标
    return box

detect_barcode("barcode_01.jpg")

detect_barcode("barcode_02.jpg")

detect_barcode("barcode_03.jpg")