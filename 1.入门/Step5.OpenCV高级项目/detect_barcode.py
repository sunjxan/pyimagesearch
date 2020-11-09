import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

def detect_barcode(image):
    image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", image)
    cv2.imshow("Gray", gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

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

    cv2.imshow("Gradient X", gradX)
    cv2.imshow("Gradient Y", gradY)
    cv2.imshow("Gradient Difference", gradient)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 平均滤波
    blurred = cv2.blur(gradient, (5, 5))

    # 阈值化
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作，先闭运算，后开运算
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None, iterations=4)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, None, iterations=4)

    cv2.imshow("Blurred", blurred)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Closed", closed)
    cv2.imshow("Opened", opened)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 找出最大轮廓
    cnts, hier = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    maxContour = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)
    cv2.imshow("Max Contour", cv2.bitwise_and(image, image, mask=mask))

    # 计算最大轮廓的旋转边界框，返回矩形格式 ((中心点横坐标， 中心点纵坐标), (矩形宽度, 矩形高度), 旋转角度)
    rect = cv2.minAreaRect(maxContour)
    # 计算边界框的四个顶点坐标
    box = cv2.boxPoints(rect).round().astype(np.int32)
    mask = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask, [box], -1, 255, -1)
    cv2.imshow("Box", cv2.bitwise_and(image, image, mask=mask))

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 返回矩形四个顶点坐标
    return box

detect_barcode("barcode_01.jpg")

detect_barcode("barcode_02.jpg")

detect_barcode("barcode_03.jpg")