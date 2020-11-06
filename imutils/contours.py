import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [(cv2.boundingRect(cnt), cnt) for cnt in cnts]
    boundingBoxes = sorted(boundingBoxes, key=lambda x: x[0][i], reverse=reverse)
    return list(map(lambda x: x[1], boundingBoxes))