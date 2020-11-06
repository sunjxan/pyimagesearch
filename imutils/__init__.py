import cv2
import numpy as np
import math

def resize(image, **kwargs):
    h, w = image.shape[:2]
    if "width" in kwargs:
        ratio = kwargs["width"] / w
    elif "height" in kwargs:
        ratio = kwargs["height"] / h
    else:
        return image
    return cv2.resize(image, (round(w * ratio), round(h * ratio)))

def rotate(image, angle, scale=1.0):
    h, w, d = image.shape
    center = round(w / 2), round(h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))

def rotate_bound(image, angle, scale=1.0):
    h, w, d = image.shape
    M = cv2.getRotationMatrix2D((0, 0), angle, scale)
    pts = np.array([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
    pts = np.matmul(M[:, :2], pts.T).T
    minX, minY = np.min(pts, axis=0)
    maxX, maxY = np.max(pts, axis=0)
    M[:, 2] = [-minX, -minY]
    nW = math.ceil(maxX - minX + 1)
    nH = math.ceil(maxY - minY + 1)
    return cv2.warpAffine(image, M, (nW, nH))

def rotate_contour(image, contour, angle, scale=1.0):
    M = cv2.getRotationMatrix2D((0, 0), angle, scale)
    contour = contour.squeeze()
    contour = np.matmul(M[:, :2], contour.T).T
    minX, minY = np.min(contour, axis=0)
    maxX, maxY = np.max(contour, axis=0)
    M[:, 2] = [-minX, -minY]
    nW = math.ceil(maxX - minX + 1)
    nH = math.ceil(maxY - minY + 1)
    return cv2.warpAffine(image, M, (nW, nH))