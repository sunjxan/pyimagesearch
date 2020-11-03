import os
os.environ['DISPLAY'] = 'windows:0'

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

def draw_contours(image, cnt, index):
    M = cv2.moments(cnt)
    cX = round(M["m10"] / M["m00"])
    cY = round(M["m01"] / M["m00"])
    cv2.putText(image, "#{}".format(index + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image

image = cv2.imread("image_01.png")
accumEdged = np.zeros(image.shape[:2], dtype=np.uint8)

for chan in cv2.split(image):
    chan = cv2.medianBlur(chan, 11)
    edged = cv2.Canny(chan, 50, 200)
    accumEdged = cv2.bitwise_or(accumEdged, edged)

cv2.imshow("Edged Map", accumEdged)

cnts, _ = cv2.findContours(accumEdged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
orig = image.copy()

for index, cnt in enumerate(cnts):
    orig = draw_contours(orig, cnt, index)

cv2.imshow("Unsorted", orig)

cnts = sort_contours(cnts, method="top-to-bottom")

for index, cnt in enumerate(cnts):
    draw_contours(image, cnt, index)

cv2.imshow("Sorted", image)
cv2.waitKey()
